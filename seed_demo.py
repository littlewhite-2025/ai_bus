"""
建立資料庫、寫入路線與車輛，方便前端 / API 在「只有一支真實影片」的情況下，也能開發多台車、多條路線的畫面。

注意：除了那台接真實影片的車（source='video'）以外，其他車都是 source='simulated' 的示範資料，
車牌、座位數、佔用變化全部是假的，只用來測畫面，不要拿去當成果展示成真實數據。API 回傳會帶
source 欄位，前端可以據此標示「示範資料」。

用法：
    python seed_demo.py                       # 建庫 + 寫入路線/車輛，然後離開
    python seed_demo.py --simulate            # 之後持續讓模擬車的佔用數隨機變動（Ctrl+C 結束）
    python seed_demo.py --db bus_status.db --seats-config seats_config.json --real-plate AAA-1234 --real-route E01
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from db import (
    DEFAULT_DB_PATH,
    DEFAULT_ROUTES,
    add_bus,
    add_route,
    get_bus_id,
    get_conn,
    init_db,
    update_bus_status,
)

ROUTES = list(DEFAULT_ROUTES)

# (車牌, 路線, 座位數)：全部是編的示範資料。
# 目前每條路線一台車（組長確認）：E01 是接真實影片的那台，E02、E25 用示範車。座位數用 16，跟「16 席座況字典」一致。
SIMULATED_BUSES = [
    ("DEMO-E02-1", "E02", 16),
    ("DEMO-E25-1", "E25", 16),
]


def make_seat_ids(total: int) -> list[str]:
    # 示範車的座位編號：左側 A、右側 B 各一半（奇數座位數時 A 多一個）。
    a = (total + 1) // 2
    return [f"A{i:02d}" for i in range(1, a + 1)] + [f"B{i:02d}" for i in range(1, total - a + 1)]


def random_result(total: int, occupied: int) -> dict:
    ids = make_seat_ids(total)
    taken = set(random.sample(ids, occupied))
    return {
        "occupied_seats": {sid: ("occupied" if sid in taken else "empty") for sid in ids},
        "occupied_count": occupied,
        "total_seats": total,
        "person_count": occupied + random.randint(0, 3),  # 偶爾多幾個站著的人
    }


def seed_routes_and_demo_buses(conn) -> dict:
    # 寫入路線與示範車（可重複執行），回傳 {車牌: (bus_id, 座位數)}。main.py 的 --demo-buses 也用這個。
    for route in ROUTES:
        add_route(conn, route)
    buses = {}
    for plate, route, total in SIMULATED_BUSES:
        if get_bus_id(conn, plate) is None:
            add_bus(conn, plate, route, total, source="simulated")
        buses[plate] = (get_bus_id(conn, plate), total)
    return buses


def load_simulated(conn) -> dict:
    # 資料庫裡所有 source='simulated' 的車：{車牌: (bus_id, 座位數)}。
    rows = conn.execute("SELECT plate, bus_id, total_seats FROM buses WHERE source = 'simulated'").fetchall()
    return {r["plate"]: (r["bus_id"], r["total_seats"]) for r in rows}


def simulate_forever(db_path: str, interval: float = 2.0, stop_event=None) -> None:
    state: dict = {}
    while not (stop_event is not None and stop_event.is_set()):
        try:
            with get_conn(db_path) as conn:
                sim = load_simulated(conn)
            simulate_step(db_path, sim, state)
        except Exception as exc:
            print(f"[demo] 模擬更新失敗：{exc}")
        if stop_event is not None:
            stop_event.wait(interval)
        else:
            time.sleep(interval)


def seed(db_path: str, seats_config: str, real_plate: str, real_route: str) -> dict:
    with get_conn(db_path) as conn:
        init_db(conn)
        buses = {}
        add_route(conn, real_route)
        cfg = Path(seats_config)
        if cfg.exists():
            real_total = len(json.loads(cfg.read_text(encoding="utf-8")))
            if get_bus_id(conn, real_plate) is None:
                add_bus(conn, real_plate, real_route, real_total, seats_config=seats_config, source="video")
            buses[real_plate] = (get_bus_id(conn, real_plate), real_total)
        else:
            print(f"找不到座位座標檔 {seats_config}，略過真實車輛（{real_plate}）。")

        buses.update(seed_routes_and_demo_buses(conn))
    return buses


def simulate_step(db_path: str, sim_buses: dict, state: dict) -> None:
    with get_conn(db_path) as conn:
        for plate, (bus_id, total) in sim_buses.items():
            current = state.get(plate, random.randint(0, total))
            current = min(total, max(0, current + random.randint(-3, 3)))
            state[plate] = current
            update_bus_status(conn, bus_id, random_result(total, current))


def main():
    parser = argparse.ArgumentParser(description="建立示範用的多車資料庫")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--seats-config", default="seats_config.json", help="真實影片那台車的座位座標檔")
    parser.add_argument("--real-plate", default="AAA-1234", help="真實影片那台車的車牌（暫定）")
    parser.add_argument("--real-route", default="E01", choices=ROUTES, help="真實影片那台車的路線（暫定）")
    parser.add_argument("--simulate", action="store_true", help="持續讓模擬車的佔用數變動，Ctrl+C 結束")
    parser.add_argument("--interval", type=float, default=2.0, help="--simulate 每隔幾秒更新一次")
    args = parser.parse_args()

    buses = seed(args.db, args.seats_config, args.real_plate, args.real_route)
    print(f"資料庫：{args.db}，共 {len(buses)} 台車：")
    for plate, (bus_id, total) in buses.items():
        print(f"  bus_id={bus_id}  {plate}  座位數 {total}")

    if args.simulate:
        sim = {p: v for p, v in buses.items() if p.startswith("DEMO-")}
        state: dict = {}
        print(f"開始模擬 {len(sim)} 台示範車（假資料），每 {args.interval} 秒更新一次，Ctrl+C 結束")
        try:
            while True:
                simulate_step(args.db, sim, state)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("已停止")


if __name__ == "__main__":
    main()
