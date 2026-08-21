"""
server.py

把 run_inference.py 的推論邏輯包成一個會「真的監聽」的 FastAPI 服務，讓前端可以
用 fetch 定期拉取最新座位狀態。這支檔案放在跟 run_inference.py / calibrate.py /
seat_calibration.json 同一層目錄下執行。

安裝：
    pip install fastapi "uvicorn[standard]"
    # cv2 / ultralytics 應該已經裝好了（run_inference.py 本來就依賴它們）

啟動：
    python server.py
    # 或者：uvicorn server:app --host 0.0.0.0 --port 8000

前端設定（frontend/.env，記得改完要重啟 `npm run dev`，Vite 只在啟動時讀 .env）：
    VITE_API_BASE_URL=http://localhost:8000

驗證有沒有真的在監聽：
    curl http://localhost:8000/api/bus
    # 應該要回傳 4 個欄位的 JSON；如果 curl 就連不上，代表問題在後端本身
    # （沒跑起來 / port 被佔用 / host 綁錯），還沒輪到前端要 debug。

回傳格式（GET /api/bus）完全對齊 run_inference.py 的 build_result()：
    {
      "occupied_seats": {"A01": "empty", ...},
      "occupied_count": 2,
      "total_seats": 16,
      "person_count": 2
    }
"""

import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from run_inference import load_calibration, build_result

# ---------------------------------------------------------------------------
# 設定 — 改成你實際的路徑 / 畫面來源 / 更新頻率
# ---------------------------------------------------------------------------
CALIBRATION_PATH = "seat_calibration.json"
MODEL_PATH = "runs/detect/seat_detector-3/weights/best.pt"  # 換成你的權重路徑

# 畫面來源二選一：
#   - 靜態圖片路徑（先用這個把「後端有沒有正常監聽」跟「前端串接」兩件事分開測）
#   - 攝影機/RTSP：cv2.VideoCapture 可吃的來源（0、"/dev/video0"、rtsp://... 等）
STATIC_IMAGE_PATH = "assets/pictures/main/main2.png"
CAMERA_SOURCE = None  # 例如 0；設定這個就會改用攝影機迴圈，忽略 STATIC_IMAGE_PATH

REFRESH_SECONDS = 2.0
CONF_THRESHOLD = 0.4
PERSON_CLASS_NAME = "person"

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
_latest_error = None


def _run_one_frame(model, seat_calibration, person_class_id, frame):
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    person_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != person_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            person_boxes.append((x1, y1, x2, y2))
    return build_result(seat_calibration, person_boxes)


def _inference_loop():
    """背景執行緒：定期跑一次推論、更新 _latest_result，讓 /api/bus 隨時有資料可回。"""
    global _latest_result, _latest_error

    import cv2
    from ultralytics import YOLO

    try:
        seat_calibration = load_calibration(CALIBRATION_PATH)
        model = YOLO(MODEL_PATH)
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        person_class_id = name_to_id.get(PERSON_CLASS_NAME.lower())
        if person_class_id is None:
            raise ValueError(f"模型 class 對照表沒有 '{PERSON_CLASS_NAME}': {model.names}")
    except Exception as exc:  # noqa: BLE001 - 開機失敗要能被 /api/bus 回報出來
        with _state_lock:
            _latest_error = f"初始化失敗：{exc}"
        return

    cap = None
    if CAMERA_SOURCE is not None:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            with _state_lock:
                _latest_error = f"無法開啟攝影機來源：{CAMERA_SOURCE}"
            return

    while True:
        try:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("讀取攝影機畫面失敗")
            else:
                frame = cv2.imread(STATIC_IMAGE_PATH)
                if frame is None:
                    raise FileNotFoundError(f"無法讀取圖片：{STATIC_IMAGE_PATH}")

            result = _run_one_frame(model, seat_calibration, person_class_id, frame)

            with _state_lock:
                _latest_result = result
                _latest_error = None

        except Exception as exc:  # noqa: BLE001 - 迴圈裡任何失敗都不該讓執行緒整個掛掉
            with _state_lock:
                _latest_error = str(exc)

        time.sleep(REFRESH_SECONDS)


@app.on_event("startup")
def _start_background_loop():
    threading.Thread(target=_inference_loop, daemon=True).start()


@app.get("/api/bus")
def get_bus():
    with _state_lock:
        if _latest_result is not None:
            return _latest_result
        error = _latest_error
    raise HTTPException(status_code=503, detail=error or "尚未產生第一筆推論結果")


@app.get("/healthz")
def healthz():
    """快速確認服務有沒有在監聽，不會等推論迴圈。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # 顯式指定 host="0.0.0.0"：只呼叫 uvicorn.run(app) 而不給 host/port，
    # 或是忘記真的呼叫 uvicorn.run()（只是 import 了 app 就結束），
    # 是「FastAPI 看起來沒有監聽」最常見的兩個原因。
    uvicorn.run(app, host="0.0.0.0", port=8000)
