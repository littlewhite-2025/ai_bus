"""
建立資料庫、寫入路線與車輛，方便前端 / API 在「只有一支真實影片」的情況下，也能開發多台車、多條路線的畫面

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

from db import DEFAULT_DB_PATH, add_bus, add_route, get_bus_id, get_conn, init_db, update_bus_status

ROUTES = ["E01", "E02", "E25"]

# (車牌, 路線, 座位數): 全部是編的示範資料
SIMULATED_BUSES = [
    ("DEMO-E01-2", "E01", 24),
    ("DEMO-E02-1", "E02", 30),
    ("DEMO-E25-1", "E25", 36),
    ("DEMO-E25-2", "E25", 36),
]


def make_seat_ids(total: int) -> list[str]:
    # 示範車的座位編號
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


def seed(db_path: str, seats_config: str, real_plate: str, real_route: str) -> dict:
    #建庫並寫入路線 / 車輛，回傳 {plate: (bus_id, total_seats)}
    with get_conn(db_path) as conn:
        init_db(conn)
        for route in ROUTES:
            add_route(conn, route)

        buses = {}
        cfg = Path(seats_config)
        if cfg.exists():
            real_total = len(json.loads(cfg.read_text(encoding="utf-8")))
            if get_bus_id(conn, real_plate) is None:
                add_bus(conn, real_plate, real_route, real_total, seats_config=seats_config, source="video")
            buses[real_plate] = (get_bus_id(conn, real_plate), real_total)
        else:
            print(f"找不到座位座標檔 {seats_config}，略過真實車輛（{real_plate}）。")

        for plate, route, total in SIMULATED_BUSES:
            if get_bus_id(conn, plate) is None:
                add_bus(conn, plate, route, total, source="simulated")
            buses[plate] = (get_bus_id(conn, plate), total)
    return buses


def simulate_step(db_path: str, sim_buses: dict, state: dict) -> None:
    # 讓每台模擬車的佔用數隨機 +-0~3 席，寫進資料庫
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
