"""
main.py

座位偵測的 FastAPI 服務，這是以server.py改寫的:
    1. 新增 /api/bus-status：總座位數、已佔用數、剩餘數、狀態文字
       （尚有座位 / 座位有限 / 已滿座）、16 席狀態字典
    2. 偵測改走 StreamDetector：固定座標 + 匈牙利配對 + 5 幀中滿 3 幀的時間防抖，
       API 回傳的座位狀態不會因為單幀漏偵測而閃爍
    3. /api/bus 保留，格式不變（前端目前打的就是這個），前端不用改
    4. 如果推論很久都沒有產生新結果（攝影機斷線、推論掛掉），不再一直回傳
       舊資料，改回 503，讓前端能顯示「連線異常」而不是看到凍結的畫面

啟動：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json --camera 0

    # 沒有攝影機時，用公車影片當來源（播完會自動從頭循環，適合前端串接測試／展示）：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json --video assets/videos/bus.mp4

    # 先用靜態圖片測試整條路：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json --image assets/pictures/main/main2.png

    # 或用 uvicorn 啟動（參數就吃下面 CONFIG 的預設值）：
    uvicorn main:app --host 0.0.0.0 --port 8000

注意：stream_test.py 只是在終端機印結果的測試腳本，不會開 API，前端接不到它。
要讓前端有資料，必須讓這支 main.py 一直開著（前端 .env 的 VITE_API_BASE_URL 指向它）。

驗證：
    curl -s http://localhost:8000/healthz
    curl -s http://localhost:8000/api/bus-status | python3 -m json.tool
    # 或直接開瀏覽器：http://localhost:8000/docs（FastAPI 自動產生的互動式文件）

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
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
    refresh_seconds=0.5,
    window=5,
    min_votes=3,
)

# 超過這麼多秒沒有新的推論結果，就視為資料過期，API 改回 503
STALE_AFTER_SECONDS = 15.0

@asynccontextmanager
async def lifespan(_app):
    # 服務啟動時開背景推論執行緒（取代已經被 FastAPI 棄用的 @app.on_event("startup")）
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
    return {
        "service": "AI 智慧公車 · 座位偵測 API",
        "endpoints": ["/api/bus-status", "/api/bus", "/healthz", "/docs"],
    }


@app.get("/api/bus-status")
def get_bus_status():
    return build_bus_status(_get_latest_or_503())


@app.get("/api/bus")
def get_bus():
    return _get_latest_or_503()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

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
    )

    # 直接傳 app 物件而不是 "main:app" 字串，這樣不會因為 reload 機制讓背景推論執行緒（和單例模型）被建立兩次。
    print(f"[main] API 啟動中：瀏覽器請開 http://127.0.0.1:{args.port}/ （/docs 有互動式文件）")
    print("[main] 注意：0.0.0.0 只是「監聽所有網卡」，在 Windows 上不能直接當網址開。")
    uvicorn.run(app, host=args.host, port=args.port)
