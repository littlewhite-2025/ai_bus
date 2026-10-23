"""
main.py，跟 server.py 比，
    1. 新增 /api/bus-status：總座位數、已佔用數、剩餘數、狀態文字
       （尚有座位 / 座位有限 / 已滿座）、16 席狀態字典
    2. 偵測改走 StreamDetector：固定座標 + 匈牙利配對 + 5 幀中滿 3 幀的時間防抖，
       API 回傳的座位狀態不會因為單幀漏偵測而閃爍
    3. /api/bus 保留，格式不變（前端目前打的就是這個），前端不用改
    4. 如果推論很久都沒有產生新結果（攝影機斷線、推論掛掉），不再一直回傳
       舊資料，改回 503，讓前端能顯示「連線異常」而不是看到凍結的畫面

啟動：
    python main.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json --camera 0

    python main.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json --image assets/pictures/main/main2.png

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
    image=None,
    camera=0,
    conf=0.4,
    person_class_name="person",
    score_thresh=0.15,
    refresh_seconds=0.5,
    window=5,
    min_votes=3,
)

# 超過這麼多秒沒有新的推論結果，就視為資料過期，API 改回 503
STALE_AFTER_SECONDS = 15.0

app = FastAPI(title="AI 智慧公車 · 座位偵測 API")

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
        )
    except Exception as exc:  # noqa: BLE001 - 開機失敗要能被 API 回報出來
        with _state_lock:
            _latest_error = f"初始化失敗：{exc}"
        return

    cap = None
    if image_path is None:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            with _state_lock:
                _latest_error = (
                    f"無法開啟攝影機（編號 {camera_source}）。"
                    f"可以先用 run_camera_detect.py --camera 0 / 1 / 2 找出可用的編號。"
                )
            return
        print(f"[main] 使用攝影機來源：{camera_source}")
    else:
        print(f"[main] 使用靜態圖片來源：{image_path}")

    last_run = 0.0
    try:
        while True:
            try:
                if cap is not None:
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

                if cap is None:
                    time.sleep(refresh)

            except Exception as exc:  # noqa: BLE001 - 迴圈裡任何失敗都不該讓執行緒整個掛掉
                with _state_lock:
                    _latest_error = str(exc)
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


@app.on_event("startup")
def _start_background_loop():
    threading.Thread(target=_inference_loop, daemon=True).start()


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
    source.add_argument("--image", default=None, help="靜態圖片路徑（測試用）")
    source.add_argument("--camera", type=int, default=None, help="攝影機編號，預設 0")

    parser.add_argument("--conf", type=float, default=CONFIG["conf"], help="YOLO 偵測信心門檻")
    parser.add_argument(
        "--score-thresh", type=float, default=CONFIG["score_thresh"], help="座位佔用判斷的分數門檻"
    )
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
    import uvicorn

    args = _parse_args()

    CONFIG.update(
        model=args.model,
        seats_config=args.seats_config,
        image=args.image,
        camera=args.camera if args.camera is not None else 0,
        conf=args.conf,
        score_thresh=args.score_thresh,
        person_class_name=args.person_class_name,
        refresh_seconds=args.refresh_seconds,
        window=args.window,
        min_votes=args.min_votes,
    )

    uvicorn.run(app, host=args.host, port=args.port)
