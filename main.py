"""
main.py

座位偵測的 FastAPI 服務（由 server.py 改來，原本的 server.py 可以不用了）。

跟 server.py 比，這一版：
    1. 新增 /api/bus-status：總座位數、已佔用數、剩餘數、狀態文字
       （尚有座位 / 座位有限 / 已滿座）、16 席狀態字典
    2. 偵測改走 StreamDetector：固定座標 + 匈牙利配對 + 5 幀中滿 3 幀的時間防抖，
       API 回傳的座位狀態不會因為單幀漏偵測而閃爍
    3. /api/bus 保留，格式不變（前端目前打的就是這個），前端不用改
    4. 如果推論很久都沒有產生新結果（攝影機斷線、推論掛掉），不再一直回傳
       舊資料，改回 503，讓前端能顯示「連線異常」而不是看到凍結的畫面

啟動：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt \
        --seats-config seats_config.json --camera 0

    # 沒有攝影機時，用公車影片當來源（播完會自動從頭循環，適合前端串接測試／展示）：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt \
        --seats-config seats_config.json --video assets/videos/bus.mp4

    # 先用靜態圖片測試整條路：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt \
        --seats-config seats_config.json --image assets/pictures/main/main2.png

    # 或用 uvicorn 啟動（參數就吃下面 CONFIG 的預設值）：
    uvicorn main:app --host 0.0.0.0 --port 8000

注意：stream_test.py 只是在終端機印結果的測試腳本，不會開 API，前端接不到它。
要讓前端有資料，必須讓這支 main.py 一直開著（前端 .env 的 VITE_API_BASE_URL 指向它）。

驗證：
    curl -s http://localhost:8000/healthz
    curl -s http://localhost:8000/api/bus-status | python3 -m json.tool
    # 或直接開瀏覽器：http://localhost:8000/docs（FastAPI 自動產生的互動式文件）

多車 / 多路線 API（資料存在 SQLite，db.py + schema.sql）：
    GET /api/routes                        路線清單（E01 / E02 / E25）與各路線的車輛數
    GET /api/routes/{route_id}/buses       某條路線上的公車與目前座位狀態
    GET /api/buses                         公車清單（可加 ?route_id=E01）
    GET /api/buses/{bus_id}                單一公車：總座位數、已佔用數、剩餘數、狀態文字、各座位狀態、最後更新時間
    GET /api/buses/{bus_id}/history        最近一段時間的佔用數紀錄（?minutes=60）
    接影片 / 攝影機的那台車用 --plate、--route 指定（預設 AAA-1234 / E01）；
    其他路線沒有真實資料，加 --demo-buses 會產生「示範資料」的車（資料庫裡 source = simulated）。

/api/bus-status 回傳範例：
    {
      "total_seats": 16,
      "occupied_count": 5,
      "remaining_seats": 11,
      "status_text": "尚有座位",
      "occupied_seats": {"A01": "empty", "A02": "occupied", ...}
    }
"""

import argparse
import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import db
from bus_status import build_bus_status
from seat_matcher import SeatDetector, load_seats_config
from stream_detector import StreamDetector

# 預設值：直接 `uvicorn main:app` 啟動（沒有經過 argparse）時會用這組。
# 用 `python main.py --...` 啟動的話，下面 __main__ 區塊會覆寫這個 dict。
CONFIG = dict(
    seats_config="seats_config.json",
    model="runs/detect/seat_detector-3/weights/best.pt",
    video=None,       # 影片路徑（播完自動循環）；有值時優先於 image / camera
    image=None,       # 靜態圖片路徑；有值時優先於 camera
    camera=0,         # 攝影機編號；video / image 都沒給時才用
    conf=0.4,
    imgsz=None,       # 推論解析度；遠處的人抓不到時調大（960、1280），預設 640
    person_class_name="person",
    score_thresh=0.15,
    # 每隔多少秒推論一次。防抖是以「推論次數」計算的，5 幀中滿 3 幀 =
    # 約 3 次推論，所以座位狀態翻轉大約要 3 * refresh_seconds 秒。
    refresh_seconds=0.5,
    window=5,
    min_votes=3,
    db_path=db.DEFAULT_DB_PATH,   # SQLite 資料庫檔
    plate="AAA-1234",             # 這個程序處理的那台車的車牌（暫定）
    route="E01",                  # 那台車屬於哪條路線（暫定）
    demo_buses=False,             # True：另外產生其他路線的示範車（假資料）
    demo_interval=2.0,            # 示範車每幾秒更新一次
)

# 超過這麼多秒沒有新的推論結果，就視為資料過期，API 改回 503
STALE_AFTER_SECONDS = 15.0

@asynccontextmanager
async def lifespan(_app):
    # 服務啟動時：建好資料庫、（選用）啟動示範車、開背景推論執行緒
    # （取代已經被 FastAPI 棄用的 @app.on_event("startup")）
    with db.get_conn(CONFIG["db_path"]) as conn:
        db.init_db(conn)
        for route_id in db.DEFAULT_ROUTES:
            db.add_route(conn, route_id)
        if CONFIG["demo_buses"]:
            import seed_demo

            seed_demo.seed_routes_and_demo_buses(conn)
    if CONFIG["demo_buses"]:
        import seed_demo

        threading.Thread(
            target=seed_demo.simulate_forever,
            args=(CONFIG["db_path"], CONFIG["demo_interval"]),
            daemon=True,
        ).start()
        print("[main] 已啟動示範車（資料庫裡 source = simulated，是假資料，只用來展示多路線）")
    threading.Thread(target=_inference_loop, daemon=True).start()
    yield


app = FastAPI(title="AI 智慧公車 · 座位偵測 API", lifespan=lifespan)

# 開發階段先全開；正式環境請換成白名單，例如：
#   allow_origins=["http://localhost:5173", "https://your-frontend-domain"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_state_lock = threading.Lock()
_latest_result = None
_latest_time = 0.0      # 最近一次成功推論的時間（time.monotonic）
_latest_error = None


def _inference_loop():
    """背景執行緒：持續推論、更新 _latest_result，讓 API 隨時有資料可回。"""
    global _latest_result, _latest_time, _latest_error

    import cv2

    video_path = CONFIG["video"]
    image_path = CONFIG["image"]
    camera_source = CONFIG["camera"]
    refresh = CONFIG["refresh_seconds"]

    try:
        seats_config = load_seats_config(CONFIG["seats_config"])
        # SeatDetector 是單例：模型只會在這裡載入一次
        detector = SeatDetector(
            model_path=CONFIG["model"],
            person_class_name=CONFIG["person_class_name"],
        )
        stream = StreamDetector(
            detector,
            seats_config,
            conf=CONFIG["conf"],
            score_thresh=CONFIG["score_thresh"],
            window=CONFIG["window"],
            min_votes=CONFIG["min_votes"],
            imgsz=CONFIG["imgsz"],
        )
    except Exception as exc:  # noqa: BLE001 - 開機失敗要能被 API 回報出來
        with _state_lock:
            _latest_error = f"初始化失敗：{exc}"
        print(f"[main] 初始化失敗：{exc}")
        return

    # 把這台車登記進資料庫（已存在就更新設定）。資料庫出問題不影響原本的 /api/bus 與 /api/bus-status。
    bus_id = None
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            bus_id = db.ensure_bus(
                conn,
                CONFIG["plate"],
                CONFIG["route"],
                len(seats_config),
                seats_config=CONFIG["seats_config"],
                source="camera" if (video_path is None and image_path is None) else "video",
            )
        print(f"[main] 已登記車輛 {CONFIG['plate']}（路線 {CONFIG['route']}、{len(seats_config)} 席）到 {CONFIG['db_path']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 登記車輛失敗，這次不寫資料庫：{exc}")

    cap = None
    step = 1  # 影片模式：每次推論前，影片往前走幾幀
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with _state_lock:
                _latest_error = f"無法開啟影片：{video_path}"
            print(f"[main] 無法開啟影片：{video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * refresh))
        print(f"[main] 使用影片來源：{video_path}（{fps:.1f} fps，每 {step} 幀推論一次，播完自動循環）")
    elif image_path is not None:
        print(f"[main] 使用靜態圖片來源：{image_path}")
    else:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            with _state_lock:
                _latest_error = (
                    f"無法開啟攝影機（編號 {camera_source}）。"
                    f"可以先用 run_camera_detect.py --camera 0 / 1 / 2 找出可用的編號。"
                )
            print(f"[main] 無法開啟攝影機（編號 {camera_source}）")
            return
        print(f"[main] 使用攝影機來源：{camera_source}")

    last_run = 0.0
    last_logged = None
    last_db_error = None
    rewinds_in_a_row = 0
    try:
        while True:
            tick = time.monotonic()
            try:
                if video_path is not None:
                    # 影片：往前走 step 幀取一張來推論，接著等到下一個 refresh 週期。
                    # 播完就倒回開頭繼續播，並清掉防抖歷史，避免新一輪吃到上一輪結尾的狀態。
                    ok = all(cap.grab() for _ in range(step - 1))
                    if ok:
                        ok, frame = cap.read()
                    if not ok:
                        rewinds_in_a_row += 1
                        if rewinds_in_a_row > 1:
                            raise RuntimeError(f"影片讀不到畫面：{video_path}")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        stream.debouncer.reset()
                        continue
                    rewinds_in_a_row = 0
                elif cap is not None:
                    # 攝影機要「不停地讀」，不然相機內部的緩衝區會堆著舊畫面，
                    # 隔一陣子才讀一幀就會讀到好幾秒前的影像。只在間隔到了才推論。
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("讀取攝影機畫面失敗")
                    now = time.monotonic()
                    if now - last_run < refresh:
                        continue
                    last_run = now
                else:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        raise FileNotFoundError(f"無法讀取圖片：{image_path}")

                result, _ = stream.process_frame(frame, draw=False)

                with _state_lock:
                    _latest_result = result
                    _latest_time = time.monotonic()
                    _latest_error = None

                if bus_id is not None:
                    try:
                        with db.get_conn(CONFIG["db_path"]) as conn:
                            db.update_bus_status(conn, bus_id, result)
                        last_db_error = None
                    except Exception as exc:  # noqa: BLE001 - 同樣的錯誤只印一次，避免洗版
                        if str(exc) != last_db_error:
                            last_db_error = str(exc)
                            print(f"[main] 寫入資料庫失敗：{exc}")

                # 終端機上能看到「後端有在跑、而且結果有變」，不用一直盯著前端猜
                summary = (result["occupied_count"], result["total_seats"])
                if summary != last_logged:
                    last_logged = summary
                    print(f"[main] 已佔用 {summary[0]}/{summary[1]}｜偵測到 {result['person_count']} 人")

                if cap is None or video_path is not None:
                    time.sleep(max(0.0, refresh - (time.monotonic() - tick)))

            except Exception as exc:  # noqa: BLE001 - 迴圈裡任何失敗都不該讓執行緒整個掛掉
                with _state_lock:
                    _latest_error = str(exc)
                print(f"[main] 推論發生錯誤：{exc}")
                time.sleep(refresh)
    finally:
        if cap is not None:
            cap.release()


def _get_latest_or_503() -> dict:
    with _state_lock:
        result = _latest_result
        age = time.monotonic() - _latest_time
        error = _latest_error
    if result is None:
        raise HTTPException(status_code=503, detail=error or "尚未產生第一筆推論結果")
    if age > STALE_AFTER_SECONDS:
        raise HTTPException(
            status_code=503,
            detail=f"已經 {age:.0f} 秒沒有新的推論結果。最近一次錯誤：{error or '無'}",
        )
    return result


@app.get("/")
def root():
    # 瀏覽器直接開根目錄時看到的內容，用來確認服務有在回應。
    return {
        "service": "AI 智慧公車 · 座位偵測 API",
        "endpoints": [
            "/api/routes",
            "/api/routes/{route_id}/buses",
            "/api/buses",
            "/api/buses/{bus_id}",
            "/api/buses/{bus_id}/history",
            "/api/bus-status",
            "/api/bus",
            "/healthz",
            "/docs",
        ],
    }


def _db_error() -> HTTPException:
    return HTTPException(status_code=503, detail="資料庫暫時無法讀取")


@app.get("/api/routes")
def get_routes():
    """路線清單與各路線目前啟用的公車數。"""
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            return db.list_routes(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 讀取路線失敗：{exc}")
        raise _db_error() from exc


@app.get("/api/routes/{route_id}/buses")
def get_route_buses(route_id: str, include_inactive: bool = False):
    """某條路線上的公車與目前座位狀態。路線不存在回 404。"""
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            if not db.route_exists(conn, route_id):
                raise HTTPException(status_code=404, detail=f"沒有這條路線：{route_id}")
            return db.list_bus_status(
                conn, route_id=route_id, include_inactive=include_inactive, stale_seconds=STALE_AFTER_SECONDS
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 讀取路線公車失敗：{exc}")
        raise _db_error() from exc


@app.get("/api/buses")
def get_buses(route_id: Optional[str] = Query(None, max_length=32), include_inactive: bool = False):
    """公車清單與目前座位狀態（可用 route_id 篩選）。資料過期的車不會讓整個清單失敗，用 is_stale 標記。"""
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            return db.list_bus_status(
                conn, route_id=route_id, include_inactive=include_inactive, stale_seconds=STALE_AFTER_SECONDS
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 讀取公車清單失敗：{exc}")
        raise _db_error() from exc


@app.get("/api/buses/{bus_id}")
def get_bus_by_id(bus_id: int):
    """單一公車的完整狀態（含各座位狀態與最後更新時間）。bus_id 不存在回 404。"""
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            status = db.get_bus_status(conn, bus_id=bus_id, stale_seconds=STALE_AFTER_SECONDS)
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 讀取公車失敗：{exc}")
        raise _db_error() from exc
    if status is None:
        raise HTTPException(status_code=404, detail=f"沒有這台公車：{bus_id}")
    return status


@app.get("/api/buses/{bus_id}/history")
def get_bus_history(bus_id: int, minutes: int = Query(60, ge=1, le=1440)):
    """最近 minutes 分鐘的佔用數紀錄（只在佔用數改變或隔一段時間才會有一筆）。"""
    try:
        with db.get_conn(CONFIG["db_path"]) as conn:
            if db.get_bus_status(conn, bus_id=bus_id) is None:
                raise HTTPException(status_code=404, detail=f"沒有這台公車：{bus_id}")
            return db.get_history(conn, bus_id, since_minutes=minutes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 讀取歷史失敗：{exc}")
        raise _db_error() from exc


@app.get("/api/bus-status")
def get_bus_status():
    """總座位數、已佔用數、剩餘數、狀態文字、16 席狀態字典。"""
    return build_bus_status(_get_latest_or_503())


@app.get("/api/bus")
def get_bus():
    """舊格式（只有四個欄位），前端目前打的是這個，保留不動。"""
    return _get_latest_or_503()


@app.get("/healthz")
def healthz():
    """快速確認服務有沒有在監聽，不會等推論迴圈。"""
    return {"status": "ok"}


def _port_in_use(host: str, port: int) -> bool:
    """
    這個埠現在有沒有程式在聽（就算那個程式已經卡死不回應，也會被偵測到）。

    為什麼要檢查：uvicorn 綁定埠時會設 SO_REUSEADDR，而在 Windows 上這個設定會讓
    「兩個程式同時佔用同一個埠」不報錯。如果上次的後端沒關乾淨（例如直接關掉終端機
    視窗、程式還躲在背景），新啟動的後端看起來一切正常，但瀏覽器/前端的連線可能
    被送到舊的那一個，造成「終端機有輸出、curl 有時通、前端卻收不到」的狀況。
    """
    probe_host = "127.0.0.1" if host in ("", "0.0.0.0") else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _parse_args():
    parser = argparse.ArgumentParser(description="座位偵測 API 服務")
    parser.add_argument("--model", default=CONFIG["model"], help="YOLO 權重路徑 (.pt)")
    parser.add_argument(
        "--seats-config", "--calibration", dest="seats_config",
        default=CONFIG["seats_config"], help="固化座位座標檔路徑（--calibration 為舊名稱，仍可使用）",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", default=None, help="影片路徑（播完自動循環；沒有攝影機時用這個）")
    source.add_argument("--image", default=None, help="靜態圖片路徑（測試用）")
    source.add_argument("--camera", type=int, default=None, help="攝影機編號，預設 0")

    parser.add_argument("--conf", type=float, default=CONFIG["conf"], help="YOLO 偵測信心門檻")
    parser.add_argument(
        "--score-thresh", type=float, default=CONFIG["score_thresh"], help="座位佔用判斷的分數門檻"
    )
    parser.add_argument("--imgsz", type=int, default=None, help="推論解析度，遠處的人抓不到時調大（960、1280），預設 640")
    parser.add_argument(
        "--person-class-name", default=CONFIG["person_class_name"], help="模型裡 person 這個 class 的名稱"
    )
    parser.add_argument(
        "--refresh-seconds", type=float, default=CONFIG["refresh_seconds"], help="多久推論一次（秒）"
    )
    parser.add_argument("--window", type=int, default=CONFIG["window"], help="防抖緩衝幀數")
    parser.add_argument(
        "--min-votes", type=int, default=CONFIG["min_votes"], help="緩衝內至少幾幀有人才算 occupied"
    )
    parser.add_argument("--db", default=CONFIG["db_path"], help="SQLite 資料庫檔路徑")
    parser.add_argument("--plate", default=CONFIG["plate"], help="這個程序處理的那台車的車牌")
    parser.add_argument("--route", default=CONFIG["route"], help="那台車的路線，例如 E01")
    parser.add_argument(
        "--demo-buses", action="store_true", help="另外產生其他路線的示範車（假資料，只用來展示多路線）"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if _port_in_use(args.host, args.port):
        raise SystemExit(
            f"[main] 埠 {args.port} 已經有程式在用了，可能是之前沒關乾淨的舊後端（server.py / main.py）。\n"
            f"  1. 查看是誰：netstat -ano | findstr :{args.port}   （最後一欄是 PID）\n"
            f" 2. 結束它：  taskkill /F /PID <PID>\n"
            f"  3. 再重新啟動這支程式。或是改用別的埠：--port 8001（前端 .env 也要跟著改）"
        ) 

    import uvicorn

    CONFIG.update(
        model=args.model,
        seats_config=args.seats_config,
        video=args.video,
        image=args.image,
        camera=args.camera if args.camera is not None else 0,
        conf=args.conf,
        imgsz=args.imgsz,
        score_thresh=args.score_thresh,
        person_class_name=args.person_class_name,
        refresh_seconds=args.refresh_seconds,
        window=args.window,
        min_votes=args.min_votes,
        db_path=args.db,
        plate=args.plate,
        route=args.route,
        demo_buses=args.demo_buses,
    )

    # 直接傳 app 物件而不是 "main:app" 字串，這樣不會因為 reload 機制讓
    # 背景推論執行緒（和單例模型）被建立兩次。
    print(f"[main] API 啟動中：瀏覽器請開 http://127.0.0.1:{args.port}/ （/docs 有互動式文件）")
    print("[main] 注意：0.0.0.0 只是「監聽所有網卡」，在 Windows 上不能直接當網址開。")
    uvicorn.run(app, host=args.host, port=args.port)
