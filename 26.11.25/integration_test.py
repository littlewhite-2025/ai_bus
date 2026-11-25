"""
確認座況 JSON 持續產生→ 後端收到→ API 讀得到→ 寫進 SQLite 資料庫

這會先啟動後端(main.py)，等它載入模型、算出第一筆結果，然後連續取樣一段時間(預設30秒)，每秒打一次 API，把每次的座況印在終端機，最後會逐項檢查，每一項印 PASS / FAIL，並把完整過程存成 integration_report.txt

使用方式:
    python integration_test.py --video assets/videos/bus.mp4 --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json

    預設會用獨立的埠(8765) 和獨立的資料庫(integration_test.db)、後端與 bus_status.db

如果後端已經啟動好了:
    python integration_test.py --no-launch --base-url http://127.0.0.1:8000 --db bus_status.db --plate AAA-1234

"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bus_status import STATUS_AVAILABLE, STATUS_FULL, STATUS_LIMITED, status_text

REQUIRED_KEYS = {
    "bus_id": int, "plate": str, "route_id": str, "route_name": str, "source": str, "is_active": bool,
    "total_seats": int, "occupied_count": int, "remaining_seats": int, "status_text": str,
    "person_count": int, "occupied_seats": dict, "updated_at": str, "updated_at_tw": str,
    "has_data": bool, "is_stale": bool,
}
LEGACY_KEYS = {"occupied_seats", "occupied_count", "total_seats", "person_count"}
VALID_STATUS = {STATUS_AVAILABLE, STATUS_LIMITED, STATUS_FULL}


# 小工具
def http_get_json(url: str, timeout: float = 5.0) -> Tuple[object, float]:
    # GET 一個JSON，回傳(內容, 花了幾毫秒)
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body), (time.perf_counter() - start) * 1000


def iso_to_epoch(text: str) -> float:
    return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))


def seat_string(seats: Dict[str, str]) -> str:
    # 'A[..##....] B[........]'，# = 有人，. = 空位；純 ASCII 避免中文字寬讓終端機排版歪掉。
    parts = []
    for prefix in sorted({sid[:1] for sid in seats}):
        ids = sorted(k for k in seats if k[:1] == prefix)
        parts.append(f"{prefix}[{''.join('#' if seats[k] == 'occupied' else '.' for k in ids)}]")
    return " ".join(parts)


def validate_bus_json(d: dict) -> List[str]:
    # 檢查一筆公車狀態 JSON 的格式與內部一致性，回傳問題清單(空=沒問題)
    problems = []
    for key, typ in REQUIRED_KEYS.items():
        if key not in d:
            problems.append(f"缺少欄位 {key}")
        elif d[key] is None:
            problems.append(f"{key} 是 null（已經有資料的車不該有空值）")
        elif not isinstance(d[key], typ):
            problems.append(f"{key} 型態不對（{type(d[key]).__name__}，應為 {typ.__name__}）")
    if problems:
        return problems

    seats, total, occ = d["occupied_seats"], d["total_seats"], d["occupied_count"]
    if len(seats) != total:
        problems.append(f"座位字典有 {len(seats)} 筆，但 total_seats = {total}")
    if any(v not in ("empty", "occupied") for v in seats.values()):
        problems.append("座位字典裡出現 empty / occupied 以外的值")
    if sum(1 for v in seats.values() if v == "occupied") != occ:
        problems.append(f"字典裡有人的座位數 != occupied_count（{occ}）")
    if d["remaining_seats"] != total - occ:
        problems.append(f"remaining_seats = {d['remaining_seats']}，但 total - occupied = {total - occ}")
    if d["status_text"] not in VALID_STATUS:
        problems.append(f"status_text 不是三種合法值之一：{d['status_text']!r}")
    elif d["status_text"] != status_text(occ, total):
        problems.append(f"status_text {d['status_text']!r} 跟門檻規則算出的 {status_text(occ, total)!r} 不一致")
    try:
        iso_to_epoch(d["updated_at"])
    except ValueError:
        problems.append(f"updated_at 格式不對：{d['updated_at']!r}")
    return problems


class Checks:
    def __init__(self):
        self.items: List[Tuple[str, str, str]] = []

    def add(self, name: str, ok: Optional[bool], detail: str = "") -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.items.append((status, name, detail))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.items if s == "FAIL")

    def lines(self) -> List[str]:
        return [f"[{s}] {n}" + (f" — {d}" if d else "") for s, n, d in self.items]


# 後端
class Backend:
    # 用subprocess啟動後端，並把它印出來的每一行收進 self.lines(同時印出重要的幾種)
    def __init__(self, cmd: List[str], verbose: bool):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env,
        )
        self.lines: List[str] = []
        self.started = time.time()
        self._verbose = verbose
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for raw in self.proc.stdout:
            line = raw.rstrip()
            self.lines.append(line)
            important = line.startswith(("[main]", "[SeatDetector]", "[demo]")) or any(
                w in line for w in ("Traceback", "Error", "ERROR")
            )
            if self._verbose or important:
                print(f"    [backend] {line}")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# SQLite
def read_db(db_path: str, bus_id: int) -> Tuple[Optional[dict], List[dict]]:
    """讀 bus_status 那一列、以及歷史紀錄。只做 SELECT。"""
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM bus_status WHERE bus_id = ?", (bus_id,)).fetchone()
        status = dict(row) if row else None
        if status:
            status["occupied_seats"] = json.loads(status["occupied_seats"])
        log = [dict(r) for r in conn.execute(
            "SELECT recorded_at, occupied_count FROM bus_status_log WHERE bus_id = ? ORDER BY log_id", (bus_id,))]
        return status, log
    finally:
        conn.close()


# 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="MP4 連續推論 + FastAPI + SQLite 整合測試")
    ap.add_argument("--video", default="assets/videos/bus.mp4")
    ap.add_argument("--model", default="runs/detect/seat_detector-3/weights/best.pt")
    ap.add_argument("--seats-config", default="seats_config.json")
    ap.add_argument("--plate", default="AAA-1234")
    ap.add_argument("--route", default="E01")
    ap.add_argument("--port", type=int, default=8765, help="測試用的埠，預設 8765，避免撞到平常開發用的 8000")
    ap.add_argument("--db", default="integration_test.db", help="測試用的資料庫檔，每次執行都會先刪掉重建")
    ap.add_argument("--duration", type=float, default=30.0, help="取樣多久（秒），預設 30")
    ap.add_argument("--interval", type=float, default=1.0, help="每隔幾秒取樣一次，預設 1")
    ap.add_argument("--refresh-seconds", type=float, default=0.5, help="後端每隔幾秒推論一次")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--startup-timeout", type=float, default=180.0, help="等後端載入模型並算出第一筆結果的上限（秒）")
    ap.add_argument("--max-age", type=float, default=5.0, help="資料「新鮮度」門檻（秒）：中位數超過就判定不即時")
    ap.add_argument("--max-gap", type=float, default=10.0, help="連續兩次資料更新之間最長允許幾秒")
    ap.add_argument("--no-demo", action="store_true", help="不啟動其他路線的示範車")
    ap.add_argument("--no-launch", action="store_true", help="不啟動後端，直接測已經在跑的（要搭配 --base-url）")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--backend-script", default="main.py")
    ap.add_argument("--report", default="integration_report.txt")
    ap.add_argument("--verbose-backend", action="store_true", help="把後端所有輸出都印出來（預設只印 [main] 與錯誤）")
    args = ap.parse_args()

    base = args.base_url or f"http://127.0.0.1:{args.port}"
    checks = Checks()
    samples: List[dict] = []
    t0_wall = datetime.now()
    out_lines: List[str] = []

    def say(text: str = "") -> None:
        print(text)
        out_lines.append(text)

    say("=" * 100)
    say("AI 辨識（MP4 連續推論）× FastAPI × SQLite 整合測試")
    say(f"時間：{t0_wall:%Y-%m-%d %H:%M:%S}    後端：{base}    資料庫：{args.db}")
    say("=" * 100)

    # 影片資訊
    if not args.no_launch:
        try:
            import cv2

            cap = cv2.VideoCapture(args.video)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                say(f"影片：{args.video}  {size}，{fps:.1f} fps，{n} 幀（約 {n / fps:.0f} 秒）" if fps
                    else f"影片：{args.video}  {size}")
            else:
                say(f"警告：打不開影片 {args.video}")
            cap.release()
        except ImportError:
            pass

    backend: Optional[Backend] = None
    bus: Optional[dict] = None
    bus_id: Optional[int] = None
    try:
        # 啟動後端
        if not args.no_launch:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(args.db + suffix)
                except FileNotFoundError:
                    pass
            script = args.backend_script
            if not os.path.isabs(script) and not os.path.exists(script):
                script = str(Path(__file__).with_name(script))
            cmd = [sys.executable, "-u", script, "--model", args.model, "--seats-config", args.seats_config,
                   "--video", args.video, "--plate", args.plate, "--route", args.route, "--db", args.db,
                   "--host", "127.0.0.1", "--port", str(args.port), "--refresh-seconds", str(args.refresh_seconds)]
            if not args.no_demo:
                cmd.append("--demo-buses")
            if args.conf is not None:
                cmd += ["--conf", str(args.conf)]
            if args.imgsz:
                cmd += ["--imgsz", str(args.imgsz)]
            say(f"啟動後端：{' '.join(cmd)}")
            backend = Backend(cmd, args.verbose_backend)

        # 等後端回應
        say("等待後端啟動（第一次載入模型可能要一段時間）...")
        deadline = time.time() + args.startup_timeout
        healthy = False
        while time.time() < deadline:
            if backend and not backend.alive():
                break
            try:
                data, _ = http_get_json(base + "/healthz", timeout=2)
                healthy = data.get("status") == "ok"
                if healthy:
                    break
            except Exception:  # noqa: BLE001 - 還沒起來，繼續等
                pass
            time.sleep(0.5)
        if healthy:
            detail = ""
        elif backend and not backend.alive():
            detail = "後端程式已經結束了，請看上面 [backend] 的錯誤訊息"
        else:
            detail = f"{args.startup_timeout:.0f} 秒內沒有回應"
        checks.add("後端啟動並回應 /healthz", healthy, detail)
        if not healthy:
            return finish(checks, backend, args, say, out_lines, None, None)

        # 等第一筆推論結果出現
        first_data_after = None
        while time.time() < deadline:
            if backend and not backend.alive():
                break
            try:
                buses, _ = http_get_json(base + "/api/buses", timeout=3)
                bus = next((b for b in buses if b["plate"] == args.plate), None)
                if bus and bus.get("has_data"):
                    first_data_after = time.time() - (backend.started if backend else time.time())
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        checks.add("第一筆 AI 推論結果出現（模型載入 + 第一次推論）", first_data_after is not None,
                   f"約 {first_data_after:.0f} 秒" if first_data_after is not None else
                   f"{args.startup_timeout:.0f} 秒內 {args.plate} 一直沒有資料（影片打不開？模型路徑錯？）")
        if first_data_after is None:
            return finish(checks, backend, args, say, out_lines, None, None)

        # 清單API
        routes, _ = http_get_json(base + "/api/routes")
        route_ids = {r["route_id"] for r in routes}
        expected = {args.route} if args.no_demo else {"E01", "E02", "E25", args.route}
        checks.add("路線清單 /api/routes", expected <= route_ids, f"收到 {sorted(route_ids)}")
        buses, _ = http_get_json(base + "/api/buses")
        bus = next((b for b in buses if b["plate"] == args.plate), None)
        checks.add("公車清單 /api/buses 含目標公車",
                   bus is not None and bus["route_id"] == args.route and bus["source"] in ("video", "camera"),
                   f"共 {len(buses)} 台；{args.plate} 路線 {bus['route_id'] if bus else '?'}、來源 {bus['source'] if bus else '?'}")
        if bus is None:
            return finish(checks, backend, args, say, out_lines, None, None)
        bus_id = bus["bus_id"]
        say(f"目標公車：bus_id={bus_id}  {args.plate}  路線 {args.route}  座位數 {bus['total_seats']}")
        say("-" * 100)
        say(f"開始連續取樣 {args.duration:.0f} 秒（每 {args.interval:g} 秒一次）。座位：# = 有人，. = 空位")

        # 連續取樣
        start = time.time()
        timings: List[float] = []
        while time.time() - start < args.duration:
            if backend and not backend.alive():
                break
            try:
                d, ms = http_get_json(f"{base}/api/buses/{bus_id}")
                timings.append(ms)
            except Exception as exc:  # noqa: BLE001
                samples.append({"error": str(exc), "t": time.time() - start})
                say(f"  t={time.time() - start:5.1f}s  取樣失敗：{exc}")
                time.sleep(args.interval)
                continue
            now = time.time()
            age = now - iso_to_epoch(d["updated_at"]) if d.get("updated_at") else float("inf")
            samples.append({"t": now - start, "json": d, "age": age, "ms": ms})
            say(f"  t={now - start:5.1f}s | {datetime.now():%H:%M:%S} | occupied {d['occupied_count']:>2}/{d['total_seats']}"
                f" | remaining {d['remaining_seats']:>2} | {seat_string(d['occupied_seats'])}"
                f" | updated {d['updated_at_tw'][11:]} (age {age:4.1f}s) | {d['status_text']}")
            time.sleep(max(0.0, args.interval - ms / 1000))
        say("-" * 100)

        # 逐項檢查
        good = [s for s in samples if "json" in s]
        checks.add("連續取樣成功（API 每次都有回應）", len(good) == len(samples) and len(good) >= 3,
                   f"成功 {len(good)} / {len(samples)} 次")

        problems = [(s["t"], p) for s in good for p in validate_bus_json(s["json"])]
        checks.add("座況 JSON 格式與內部一致性（欄位、型態、字典筆數、剩餘數、狀態文字）", not problems,
                   f"{len(good)} 筆全部通過" if not problems
                   else f"t={problems[0][0]:.0f}s：{problems[0][1]}（共 {len(problems)} 項問題）")

        stamps = sorted({s["json"]["updated_at"] for s in good})
        gaps = [iso_to_epoch(b) - iso_to_epoch(a) for a, b in zip(stamps, stamps[1:])]
        final_age = good[-1]["age"] if good else float("inf")
        # 光有「更新過幾次」不夠：還要確認取樣結束時最新的一筆仍然很新，否則中途停掉也會被誤判成通過
        continuous = len(stamps) >= 3 and (max(gaps) if gaps else 0) <= args.max_gap and final_age <= args.max_age
        checks.add("連續推論：資料持續更新（updated_at 不斷往前，直到取樣結束都還在更新）", continuous,
                   (f"{len(stamps)} 個不同的更新時間，平均每 {statistics.mean(gaps):.1f} 秒更新一次，"
                    f"最長間隔 {max(gaps):.0f} 秒，取樣結束時最新一筆已是 {final_age:.1f} 秒前") if gaps
                   else f"只看到 {len(stamps)} 個更新時間，資料沒有在更新")

        ages = [s["age"] for s in good]
        med_age = statistics.median(ages) if ages else float("inf")
        checks.add(f"即時性：資料新鮮度中位數 <= {args.max_age:g} 秒", med_age <= args.max_age,
                   f"中位數 {med_age:.1f} 秒，最大 {max(ages):.1f} 秒" if ages else "沒有資料")

        fresh_ratio = sum(1 for s in good if not s["json"]["is_stale"]) / len(good) if good else 0
        checks.add("資料沒有過期（is_stale 為 false 的比例 >= 90%）", fresh_ratio >= 0.9, f"{fresh_ratio:.0%}")

        if args.db and os.path.exists(args.db):
            synced, detail = False, "一直無法在同一瞬間取得一致的三份資料"
            for _ in range(10):
                a, _ = http_get_json(base + "/api/bus")           # 後端記憶體裡的 AI 結果
                row, _log = read_db(args.db, bus_id)                 # SQLite 裡那一列
                b, _ = http_get_json(base + "/api/bus")
                if a["occupied_seats"] != b["occupied_seats"]:       # 取樣中間剛好更新了，重來
                    time.sleep(0.3)
                    continue
                if row is None:
                    detail = "資料庫裡沒有這台車的 bus_status 紀錄"
                    break
                synced = (row["occupied_seats"] == a["occupied_seats"] and row["occupied_count"] == a["occupied_count"]
                          and row["person_count"] == a["person_count"])
                detail = (f"後端 JSON 與資料庫一致：已佔用 {row['occupied_count']}、人數 {row['person_count']}、"
                          f"{len(row['occupied_seats'])} 席座位字典，更新時間 {row['updated_at']}") if synced \
                    else "後端 JSON 與資料庫內容不一樣"
                break
            checks.add("座況 JSON 寫入 SQLite（後端 /api/bus 的 JSON == 資料庫 bus_status 那一列）", synced, detail)

            _row, log = read_db(args.db, bus_id)
            observed = {s["json"]["occupied_count"] for s in good}
            logged = {r["occupied_count"] for r in log}
            checks.add("歷史紀錄 bus_status_log 有寫入，且涵蓋取樣期間看到的所有佔用數",
                       bool(log) and observed <= logged,
                       f"共 {len(log)} 筆；取樣期間佔用數出現過 {sorted(observed)}，資料庫歷史有 {sorted(logged)}")
        else:
            checks.add("座況 JSON 寫入 SQLite", None, "沒有指定可讀的 --db，略過")
            checks.add("歷史紀錄 bus_status_log", None, "沒有指定可讀的 --db，略過")

        legacy_a, _ = http_get_json(base + "/api/bus")
        legacy_b, _ = http_get_json(base + "/api/bus-status")
        checks.add("舊端點相容（/api/bus 四個欄位、/api/bus-status 含狀態文字）",
                   set(legacy_a) == LEGACY_KEYS and {"remaining_seats", "status_text"} <= set(legacy_b),
                   f"/api/bus 欄位 {sorted(legacy_a)}")

        if args.no_demo:
            checks.add("示範車標示為 simulated 且持續更新", None, "已用 --no-demo 略過")
        else:
            buses, _ = http_get_json(base + "/api/buses")
            demos = [b for b in buses if b["plate"] != args.plate]
            ok = len(demos) >= 2 and all(b["source"] == "simulated" and b["has_data"] and not b["is_stale"] for b in demos)
            checks.add("示範車標示為 simulated 且持續更新", ok,
                       ", ".join(f"{b['route_id']}:{b['plate']}({b['source']})" for b in demos))

        if timings:
            checks.add("API 回應時間（單次最大 <= 2 秒）", max(timings) <= 2000,
                       f"平均 {statistics.mean(timings):.0f} ms，最大 {max(timings):.0f} ms")

        if backend:
            video_line = next((l for l in backend.lines if "使用影片來源" in l), None)
            checks.add("後端確實讀取 MP4 影片", video_line is not None, video_line or "後端沒有印出使用影片來源")
            bad = [l for l in backend.lines if any(w in l for w in ("Traceback", "推論發生錯誤", "寫入資料庫失敗", "登記車輛失敗"))]
            checks.add("後端執行期間沒有例外 / 推論錯誤 / 寫入資料庫失敗", not bad, bad[0] if bad else "乾淨")
        else:
            checks.add("後端確實讀取 MP4 影片", None, "沒有由本程式啟動後端，看不到它的輸出")

        occs = [s["json"]["occupied_count"] for s in good]
        if occs:
            say(f"參考：取樣期間已佔用數範圍 {min(occs)} ~ {max(occs)}（影片內容有變化時才會有範圍）")
        last = good[-1]["json"] if good else None
        db_info = read_db(args.db, bus_id) if args.db and os.path.exists(args.db) else None
        return finish(checks, backend, args, say, out_lines, last, db_info)
    finally:
        if backend:
            backend.stop()


def finish(checks, backend, args, say, out_lines, last_json, db_info) -> int:
    say()
    say("=" * 100)
    say("整合測試結果")
    say("=" * 100)
    for line in checks.lines():
        say(line)
    passed = sum(1 for s, _, _ in checks.items if s == "PASS")
    skipped = sum(1 for s, _, _ in checks.items if s == "SKIP")
    say("-" * 100)
    say(f"通過 {passed} 項、失敗 {checks.failed} 項、略過 {skipped} 項 —— "
        + ("整合測試通過" if checks.failed == 0 else "整合測試未通過"))

    report = list(out_lines)
    if last_json:
        report += ["", "最後一筆 API 回傳（GET /api/buses/{bus_id}）：", json.dumps(last_json, ensure_ascii=False, indent=2)]
    if db_info:
        row, log = db_info
        report += ["", "SQLite bus_status（目前狀態）：",
                   json.dumps({k: v for k, v in (row or {}).items() if k != "occupied_seats"}, ensure_ascii=False, indent=2),
                   f"SQLite bus_status_log（歷史紀錄，共 {len(log)} 筆，最後 10 筆）：",
                   *[f"  {r['recorded_at']}  occupied_count={r['occupied_count']}" for r in log[-10:]]]
    if backend:
        keep = [l for l in backend.lines
                if l.startswith(("[main]", "[SeatDetector]", "[demo]")) or "Traceback" in l or "Error" in l]
        report += ["", "後端輸出（節錄）：", *["  " + l for l in keep[-40:]]]
    try:
        Path(args.report).write_text("\r\n".join(report) + "\r\n", encoding="utf-8-sig")
        say(f"完整報告已存檔：{args.report}")
    except OSError as exc:
        say(f"報告存檔失敗：{exc}")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
