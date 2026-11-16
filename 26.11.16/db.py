# 多台公車 / 多條路線 / 座位狀態的 SQLite 資料層。結構定義在 schema.sql（唯一來源）。

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from bus_status import status_text

DEFAULT_DB_PATH = "bus_status.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# 超過這麼多秒沒有更新，API 就標記這台車的資料「過期」（跟 main.py 的 STALE_AFTER_SECONDS 一致）
DEFAULT_STALE_SECONDS = 15.0

VALID_SOURCES = ("camera", "video", "simulated")

# 組長指定的路線
DEFAULT_ROUTES = ("E01", "E02", "E25")

# 台灣沒有日光節約時間，固定 UTC+8 即可，不需要 tzdata
TAIWAN_TZ = timezone(timedelta(hours=8))


# 時間
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """datetime -> 'YYYY-MM-DDTHH:MM:SSZ'（UTC）"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# 連線 / 初始化
def connect(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")    # SQLite 預設不檢查外鍵，要每條連線自己打開
    conn.execute("PRAGMA busy_timeout = 5000")  # 被別人鎖住時最多等 5 秒，不要直接報錯
    conn.execute("PRAGMA journal_mode = WAL")   # 一邊寫一邊讀不互相擋（記憶體資料庫會自動忽略）
    return conn


@contextmanager
def get_conn(path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    # 用 with 開一條連線：正常結束自動 commit，發生例外自動 rollback，最後一定關閉。
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
   # 建立資料表（可以重複執行，不會動到既有資料）。
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys = ON")


# 路線、車輛
def add_route(conn: sqlite3.Connection, route_id: str, name: str = "") -> None:
    # 新增路線；已經存在就保持原樣（不會蓋掉已經填好的名稱）。
    conn.execute("INSERT OR IGNORE INTO routes (route_id, name) VALUES (?, ?)", (route_id, name))


def add_bus(
    conn: sqlite3.Connection,
    plate: str,
    route_id: str,
    total_seats: int,
    seats_config: Optional[str] = None,
    source: str = "camera",
) -> int:
    if source not in VALID_SOURCES:
        raise ValueError(f"source 只能是 {VALID_SOURCES}，收到 {source!r}")
    cur = conn.execute(
        "INSERT INTO buses (plate, route_id, total_seats, seats_config, source) VALUES (?, ?, ?, ?, ?)",
        (plate, route_id, total_seats, seats_config, source),
    )
    return int(cur.lastrowid)


def ensure_bus(
    conn: sqlite3.Connection,
    plate: str,
    route_id: str,
    total_seats: int,
    seats_config: Optional[str] = None,
    source: str = "camera",
) -> int:
    if source not in VALID_SOURCES:
        raise ValueError(f"source 只能是 {VALID_SOURCES}，收到 {source!r}")
    add_route(conn, route_id)
    row = conn.execute(
        "SELECT bus_id, route_id, total_seats, seats_config, source FROM buses WHERE plate = ?", (plate,)
    ).fetchone()
    if row is None:
        return add_bus(conn, plate, route_id, total_seats, seats_config, source)

    if (row["route_id"], row["total_seats"], row["seats_config"], row["source"]) != (
        route_id, total_seats, seats_config, source,
    ):
        conn.execute(
            "UPDATE buses SET route_id = ?, total_seats = ?, seats_config = ?, source = ? WHERE bus_id = ?",
            (route_id, total_seats, seats_config, source, row["bus_id"]),
        )
        if row["total_seats"] != total_seats:
            conn.execute("DELETE FROM bus_status WHERE bus_id = ?", (row["bus_id"],))
    return int(row["bus_id"])


def route_exists(conn: sqlite3.Connection, route_id: str) -> bool:
    return conn.execute("SELECT 1 FROM routes WHERE route_id = ?", (route_id,)).fetchone() is not None


def get_bus_id(conn: sqlite3.Connection, plate: str) -> Optional[int]:
    row = conn.execute("SELECT bus_id FROM buses WHERE plate = ?", (plate,)).fetchone()
    return int(row["bus_id"]) if row else None


# 寫入狀態
def update_bus_status(
    conn: sqlite3.Connection,
    bus_id: int,
    result: dict,
    now: Optional[datetime] = None,
    log_every_seconds: float = 60.0,
) -> bool:
    bus = conn.execute("SELECT total_seats FROM buses WHERE bus_id = ?", (bus_id,)).fetchone()
    if bus is None:
        raise KeyError(f"沒有 bus_id = {bus_id} 這台車")

    seats = result["occupied_seats"]
    occupied = int(result["occupied_count"])
    if int(result["total_seats"]) != bus["total_seats"]:
        raise ValueError(
            f"推論結果的總座位數是 {result['total_seats']}，但這台車登記的是 {bus['total_seats']}。"
            f"請確認 seats_config 跟 buses.total_seats 一致。"
        )
    if sum(1 for v in seats.values() if v == "occupied") != occupied or len(seats) != bus["total_seats"]:
        raise ValueError("occupied_seats 的內容跟 occupied_count / total_seats 對不起來")

    now = now or utc_now()
    stamp = to_iso(now)

    conn.execute(
        """
        INSERT INTO bus_status (bus_id, occupied_count, person_count, occupied_seats, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(bus_id) DO UPDATE SET
            occupied_count = excluded.occupied_count,
            person_count   = excluded.person_count,
            occupied_seats = excluded.occupied_seats,
            updated_at     = excluded.updated_at
        """,
        (bus_id, occupied, int(result.get("person_count", 0)), json.dumps(seats, ensure_ascii=False), stamp),
    )

    last = conn.execute(
        "SELECT occupied_count, recorded_at FROM bus_status_log WHERE bus_id = ? ORDER BY log_id DESC LIMIT 1",
        (bus_id,),
    ).fetchone()
    need_log = (
        last is None
        or last["occupied_count"] != occupied
        or (now - parse_iso(last["recorded_at"])).total_seconds() >= log_every_seconds
    )
    if need_log:
        conn.execute(
            "INSERT INTO bus_status_log (bus_id, occupied_count, recorded_at) VALUES (?, ?, ?)",
            (bus_id, occupied, stamp),
        )
    return need_log


def prune_log(conn: sqlite3.Connection, keep_days: int = 30, now: Optional[datetime] = None) -> int:
    cutoff = to_iso((now or utc_now()) - timedelta(days=keep_days))
    return conn.execute("DELETE FROM bus_status_log WHERE recorded_at < ?", (cutoff,)).rowcount


# 讀取（給 API 用）
def _row_to_status(row: sqlite3.Row, now: datetime, stale_seconds: float) -> dict:
    has_data = row["updated_at"] is not None
    age = (now - parse_iso(row["updated_at"])).total_seconds() if has_data else None
    occupied = row["occupied_count"]
    return {
        "bus_id": row["bus_id"],
        "plate": row["plate"],
        "route_id": row["route_id"],
        "route_name": row["route_name"],
        "source": row["source"],                       # 'simulated' 的就是示範資料
        "is_active": bool(row["is_active"]),
        "total_seats": row["total_seats"],
        "occupied_count": occupied,
        "remaining_seats": row["remaining_seats"],
        "status_text": status_text(occupied, row["total_seats"]) if has_data else None,
        "person_count": row["person_count"],
        "occupied_seats": json.loads(row["occupied_seats"]) if has_data else None,
        "updated_at": row["updated_at"],                # UTC，格式 2026-11-02T08:00:00Z
        "updated_at_tw": (                              # 同一個時間換成台灣時間（UTC+8），給人直接看
            parse_iso(row["updated_at"]).astimezone(TAIWAN_TZ).strftime("%Y-%m-%d %H:%M:%S") if has_data else None
        ),
        "has_data": has_data,
        "is_stale": (not has_data) or age > stale_seconds,
    }


def list_bus_status(
    conn: sqlite3.Connection,
    route_id: Optional[str] = None,
    include_inactive: bool = False,
    now: Optional[datetime] = None,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
) -> List[dict]:
    sql = "SELECT * FROM v_bus_status WHERE 1 = 1"
    params: list = []
    if route_id is not None:
        sql += " AND route_id = ?"
        params.append(route_id)
    if not include_inactive:
        sql += " AND is_active = 1"
    sql += " ORDER BY route_id, plate"
    now = now or utc_now()
    return [_row_to_status(r, now, stale_seconds) for r in conn.execute(sql, params)]


def get_bus_status(
    conn: sqlite3.Connection,
    bus_id: Optional[int] = None,
    plate: Optional[str] = None,
    now: Optional[datetime] = None,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
) -> Optional[dict]:
    if (bus_id is None) == (plate is None):
        raise ValueError("bus_id 和 plate 要二選一")
    col, val = ("bus_id", bus_id) if bus_id is not None else ("plate", plate)
    row = conn.execute(f"SELECT * FROM v_bus_status WHERE {col} = ?", (val,)).fetchone()
    return _row_to_status(row, now or utc_now(), stale_seconds) if row else None


def list_routes(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute(
        """
        SELECT r.route_id, r.name, COUNT(b.bus_id) AS bus_count
        FROM routes r LEFT JOIN buses b ON b.route_id = r.route_id AND b.is_active = 1
        GROUP BY r.route_id ORDER BY r.route_id
        """
    )
    return [dict(r) for r in rows]


def get_history(
    conn: sqlite3.Connection,
    bus_id: int,
    since_minutes: int = 60,
    limit: int = 1000,
    now: Optional[datetime] = None,
) -> List[Dict]:
    since = to_iso((now or utc_now()) - timedelta(minutes=since_minutes))
    rows = conn.execute(
        """
        SELECT l.recorded_at, l.occupied_count, b.total_seats - l.occupied_count AS remaining_seats
        FROM bus_status_log l JOIN buses b ON b.bus_id = l.bus_id
        WHERE l.bus_id = ? AND l.recorded_at >= ?
        ORDER BY l.recorded_at LIMIT ?
        """,
        (bus_id, since, limit),
    )
    return [dict(r) for r in rows]
