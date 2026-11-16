"""
db.py / schema.sql 的自動測試。不需要 pytest：
    python test_db.py
全部通過會印出「全部通過」。每個測試用獨立的暫存資料庫，不會動到你真正的 bus_status.db。
"""

import json
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import db
import seed_demo

T0 = datetime(2026, 11, 2, 8, 0, 0, tzinfo=timezone.utc)


def fresh():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    return path


def seats_file(n=4):
    """測試用的暫存座位座標檔，不依賴你資料夾裡真正的 seats_config.json。"""
    path = os.path.join(tempfile.mkdtemp(), "seats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({f"A{i:02d}": [0, 0, 10, 10] for i in range(1, n + 1)}, f)
    return path


def result(total=4, occupied_ids=(), person=None):
    ids = [f"A{i:02d}" for i in range(1, total + 1)]
    seats = {i: ("occupied" if i in occupied_ids else "empty") for i in ids}
    return {"occupied_seats": seats, "occupied_count": len(occupied_ids), "total_seats": total,
            "person_count": len(occupied_ids) if person is None else person}


def setup(conn, total=4):
    db.init_db(conn)
    db.add_route(conn, "E01")
    return db.add_bus(conn, "KKA-1234", "E01", total, source="video")


def test_init_is_idempotent_and_keeps_data():
    path = fresh()
    with db.get_conn(path) as c:
        setup(c)
    with db.get_conn(path) as c:
        db.init_db(c); db.init_db(c)
        assert c.execute("SELECT COUNT(*) FROM buses").fetchone()[0] == 1


def test_status_flow_and_remaining_computed_by_view():
    with db.get_conn(fresh()) as c:
        bus = setup(c)
        s = db.get_bus_status(c, bus_id=bus, now=T0)
        assert s["has_data"] is False and s["remaining_seats"] is None and s["is_stale"] is True   # 還沒收到資料
        db.update_bus_status(c, bus, result(4, ["A01", "A02"]), now=T0)
        s = db.get_bus_status(c, plate="KKA-1234", now=T0 + timedelta(seconds=3))
        assert (s["total_seats"], s["occupied_count"], s["remaining_seats"]) == (4, 2, 2)
        assert s["status_text"] == "尚有座位" and s["is_stale"] is False and s["occupied_seats"]["A01"] == "occupied"
        db.update_bus_status(c, bus, result(4, ["A01", "A02", "A03"]), now=T0 + timedelta(seconds=1))
        assert db.get_bus_status(c, bus_id=bus, now=T0)["status_text"] == "座位有限"
        db.update_bus_status(c, bus, result(4, ["A01", "A02", "A03", "A04"]), now=T0 + timedelta(seconds=2))
        assert db.get_bus_status(c, bus_id=bus, now=T0)["status_text"] == "已滿座"
        assert c.execute("SELECT COUNT(*) FROM bus_status").fetchone()[0] == 1         # 一台車永遠只有一列目前狀態
        assert db.get_bus_status(c, bus_id=bus, now=T0 + timedelta(seconds=60))["is_stale"] is True   # 太久沒更新 -> 過期


def test_history_is_sparse():
    with db.get_conn(fresh()) as c:
        bus = setup(c)
        same = result(4, ["A01"])
        assert db.update_bus_status(c, bus, same, now=T0) is True                              # 第一筆一定記
        for i in range(1, 50):                                                                  # 之後 49 次佔用數都沒變
            assert db.update_bus_status(c, bus, same, now=T0 + timedelta(seconds=i)) is False
        assert db.update_bus_status(c, bus, result(4, ["A01", "A02"]), now=T0 + timedelta(seconds=50)) is True   # 改變 -> 記
        assert db.update_bus_status(c, bus, result(4, ["A01", "A02"]), now=T0 + timedelta(seconds=111)) is True  # 沒變但超過 60 秒 -> 記
        assert c.execute("SELECT COUNT(*) FROM bus_status_log").fetchone()[0] == 3
        hist = db.get_history(c, bus, since_minutes=10, now=T0 + timedelta(minutes=2))
        assert [h["occupied_count"] for h in hist] == [1, 2, 2] and hist[0]["remaining_seats"] == 3
        assert db.prune_log(c, keep_days=30, now=T0 + timedelta(days=40)) == 3


def test_bad_data_rejected():
    with db.get_conn(fresh()) as c:
        bus = setup(c)
        for bad, exc in [
            (result(5, ["A01"]), ValueError),                                                   # 總座位數對不上
            ({**result(4, ["A01"]), "occupied_count": 3}, ValueError),                         # 字典跟數字不一致
        ]:
            try:
                db.update_bus_status(c, bus, bad, now=T0); raise AssertionError("應該要拒絕")
            except exc:
                pass
        try:
            db.update_bus_status(c, 999, result(4), now=T0); raise AssertionError
        except KeyError:
            pass
        assert c.execute("SELECT COUNT(*) FROM bus_status").fetchone()[0] == 0                 # 壞資料一筆都沒寫進去


def test_database_level_guards():
    with db.get_conn(fresh()) as c:
        bus = setup(c)
        def expect_integrity(sql, params=()):
            try:
                c.execute(sql, params); raise AssertionError("應該要被資料庫擋住：" + sql[:50])
            except sqlite3.IntegrityError:
                pass
        # 繞過 Python 檢查直接寫 SQL，資料庫自己也要擋
        expect_integrity("INSERT INTO bus_status VALUES (?, 99, 0, '{}', 'x')", (bus,))           # 佔用數 > 總座位數（trigger）
        expect_integrity("INSERT INTO buses (plate, route_id, total_seats) VALUES ('KKA-1234','E01',4)")   # 車牌重複
        expect_integrity("INSERT INTO buses (plate, route_id, total_seats) VALUES ('X-1','E99',4)")        # 路線不存在（外鍵）
        expect_integrity("INSERT INTO buses (plate, route_id, total_seats) VALUES ('X-2','E01',0)")        # 座位數 <= 0
        expect_integrity("INSERT INTO buses (plate, route_id, total_seats, source) VALUES ('X-3','E01',4,'abc')")
        try:
            db.add_bus(c, "X-4", "E01", 4, source="abc"); raise AssertionError
        except ValueError:
            pass


def test_sql_injection_is_inert():
    with db.get_conn(fresh()) as c:
        setup(c)
        evil = "x' OR '1'='1"
        assert db.get_bus_status(c, plate=evil) is None                                         # 不會因此撈出全部
        assert db.list_bus_status(c, route_id="E01'; DROP TABLE buses; --") == []
        assert c.execute("SELECT COUNT(*) FROM buses").fetchone()[0] == 1


def test_routes_and_filtering():
    SEATS = seats_file()
    path = fresh()
    made = seed_demo.seed(path, SEATS, "AAA-1234", "E01")
    assert "AAA-1234" in made and len(made) == 3
    with db.get_conn(path) as c:
        routes = {r["route_id"]: r["bus_count"] for r in db.list_routes(c)}
        assert routes == {"E01": 1, "E02": 1, "E25": 1}                                        # 一條路線一台車
        assert [b["plate"] for b in db.list_bus_status(c, route_id="E25")] == ["DEMO-E25-1"]
        assert all(b["has_data"] is False for b in db.list_bus_status(c))                       # 還沒有任何推論資料
        c.execute("UPDATE buses SET is_active = 0 WHERE plate = 'DEMO-E25-1'")
        assert len(db.list_bus_status(c, route_id="E25")) == 0
        assert len(db.list_bus_status(c, route_id="E25", include_inactive=True)) == 1
        assert db.route_exists(c, "E02") and not db.route_exists(c, "E99")
    seed_demo.seed(path, SEATS, "AAA-1234", "E01")                                # 重複執行不會重複新增
    with db.get_conn(path) as c:
        assert c.execute("SELECT COUNT(*) FROM buses").fetchone()[0] == 3


def test_simulation_writes_valid_data():
    SEATS = seats_file()
    path = fresh()
    made = seed_demo.seed(path, SEATS, "AAA-1234", "E01")
    sim = {p: v for p, v in made.items() if p.startswith("DEMO-")}
    state = {}
    for _ in range(30):
        seed_demo.simulate_step(path, sim, state)
    with db.get_conn(path) as c:
        for b in db.list_bus_status(c):
            if b["source"] == "simulated":
                assert b["has_data"] and 0 <= b["occupied_count"] <= b["total_seats"]
                assert b["remaining_seats"] == b["total_seats"] - b["occupied_count"]
                assert len(b["occupied_seats"]) == b["total_seats"]


def test_ensure_bus_registers_updates_and_resets_on_seat_change():
    with db.get_conn(fresh()) as c:
        db.init_db(c)
        a = db.ensure_bus(c, "KKA-1234", "E01", 4, seats_config="a.json", source="video")
        assert db.ensure_bus(c, "KKA-1234", "E01", 4, seats_config="a.json", source="video") == a     # 重複呼叫不新增
        assert c.execute("SELECT COUNT(*) FROM buses").fetchone()[0] == 1
        db.update_bus_status(c, a, result(4, ["A01", "A02"]), now=T0)
        b = db.ensure_bus(c, "KKA-1234", "E02", 4, seats_config="a.json", source="video")             # 只改路線
        assert b == a and db.get_bus_status(c, bus_id=a, now=T0)["route_id"] == "E02"
        assert db.get_bus_status(c, bus_id=a, now=T0)["has_data"] is True                              # 座位數沒變，狀態保留
        db.ensure_bus(c, "KKA-1234", "E02", 6, seats_config="b.json", source="video")                  # 座位數改變
        s = db.get_bus_status(c, bus_id=a, now=T0)
        assert s["total_seats"] == 6 and s["has_data"] is False                                        # 舊狀態清掉，不留對不上的資料
        db.update_bus_status(c, a, result(6, ["A01"]), now=T0)                                         # 之後能正常寫入新座位數


def test_taiwan_time_field():
    with db.get_conn(fresh()) as c:
        bus = setup(c)
        db.update_bus_status(c, bus, result(4, ["A01"]), now=T0)                                       # UTC 08:00
        s = db.get_bus_status(c, bus_id=bus, now=T0)
        assert s["updated_at"] == "2026-11-02T08:00:00Z" and s["updated_at_tw"] == "2026-11-02 16:00:00"
        assert db.get_bus_status(c, bus_id=setup_second(c), now=T0)["updated_at_tw"] is None            # 沒資料就是 None


def setup_second(conn):
    return db.add_bus(conn, "KKA-9999", "E01", 4)


def test_concurrent_writer_and_readers():
    path = fresh()
    with db.get_conn(path) as c:
        bus = setup(c, total=16)
    errors, stop = [], threading.Event()

    def writer():
        try:
            for i in range(200):
                with db.get_conn(path) as c:
                    db.update_bus_status(c, bus, result(16, [f"A{j:02d}" for j in range(1, 1 + i % 17)]), now=T0 + timedelta(seconds=i))
        except Exception as e:  # noqa: BLE001
            errors.append(("writer", e))
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                with db.get_conn(path) as c:
                    s = db.get_bus_status(c, bus_id=bus, now=T0)
                    if s["has_data"]:
                        assert s["remaining_seats"] == 16 - s["occupied_count"]
        except Exception as e:  # noqa: BLE001
            errors.append(("reader", e))

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in threads]; [t.join(30) for t in threads]
    assert not errors, errors


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("通過:", t.__name__)
    print(f"全部通過（{len(tests)} 項），SQLite 版本 {sqlite3.sqlite_version}")
