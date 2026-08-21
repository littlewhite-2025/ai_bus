"""
server.py

把 run_inference.py 的推論邏輯包成一個會「真的監聽」的 FastAPI 服務，讓前端可以
用 fetch 定期拉取最新座位狀態。這支檔案放在跟 run_inference.py / calibrate.py /
seat_calibration.json 同一層目錄下執行。

安裝：
    pip install fastapi "uvicorn[standard]"
    # cv2 / ultralytics 應該已經裝好了（run_inference.py 本來就依賴它們）

啟動（跟 run_inference.py / run_camera_detect.py 一樣用 argparse，不用改程式碼）：
    # 先用靜態圖片測試整條路（確認「後端有沒有在監聽」跟「前端串接」這兩件事）：
    # python server.py --model runs/detect/seat_detector-3/weights/best.pt --calibration seat_calibration.json --image assets/pictures/main/main2.png

    # 確認攝影機編號可用（跟 run_camera_detect.py 用同一顆鏡頭測過的編號）之後，
    # 改用即時攝影機畫面：
    # python server.py --model runs/detect/seat_detector-3/weights/best.pt --calibration seat_calibration.json --camera 0

--image 跟 --camera 只能擇一；不給的話預設走 --camera 0。

前端設定（frontend/.env，記得改完要重啟 `npm run dev`，Vite 只在啟動時讀 .env）：
    VITE_API_BASE_URL=http://localhost:8000

驗證有沒有真的在監聽、資料合不合理：
    curl -s http://localhost:8000/healthz
    curl -s http://localhost:8000/api/bus | python3 -m json.tool
    # /（根目錄）跟 /favicon.ico 出現 404 是正常的，這兩個路徑本來就沒定義路由，
    # 不代表服務沒在監聽 —— 只要 /healthz、/api/bus 有回應就是正常的。

回傳格式（GET /api/bus）完全對齊 run_inference.py 的 build_result()：
    {
      "occupied_seats": {"A01": "empty", ...},
      "occupied_count": 2,
      "total_seats": 16,
      "person_count": 2
    }

跟 run_inference.py / run_camera_detect.py 的關係：
    - run_inference.py：不用另外跑。這支程式的背景執行緒本來就是把它的邏輯
      （load_calibration + YOLO 偵測 + build_result）包進一個迴圈定期重跑。
    - run_camera_detect.py：不是必要，但建議先跑一次，純粹用來確認攝影機編號
      可以開、模型看得到 person（座位框），跟這支服務本身互不相通。
"""

import argparse
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from run_inference import load_calibration, build_result

# 預設值：直接 `uvicorn server:app` 啟動（沒有經過 argparse）時會用這組。
# 用 `python server.py --...` 啟動的話，下面 main() 會覆寫這個 dict。
CONFIG = dict(
    calibration="seat_calibration.json",
    model="runs/detect/seat_detector-3/weights/best.pt",
    image=None,       # 靜態圖片路徑；跟 camera 二選一
    camera=0,         # 攝影機編號；image 有值時會忽略這個
    conf=0.4,
    person_class_name="person",
    refresh_seconds=2.0,
)

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
    results = model.predict(source=frame, conf=CONFIG["conf"], verbose=False)
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

    image_path = CONFIG["image"]
    camera_source = CONFIG["camera"]

    try:
        seat_calibration = load_calibration(CONFIG["calibration"])
        model = YOLO(CONFIG["model"])
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        person_class_id = name_to_id.get(CONFIG["person_class_name"].lower())
        if person_class_id is None:
            raise ValueError(
                f"模型 class 對照表沒有 '{CONFIG['person_class_name']}': {model.names}"
            )
    except Exception as exc:  # noqa: BLE001 - 開機失敗要能被 /api/bus 回報出來
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
        print(f"[server] 使用攝影機來源：{camera_source}")
    else:
        print(f"[server] 使用靜態圖片來源：{image_path}")

    while True:
        try:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("讀取攝影機畫面失敗")
            else:
                frame = cv2.imread(image_path)
                if frame is None:
                    raise FileNotFoundError(f"無法讀取圖片：{image_path}")

            result = _run_one_frame(model, seat_calibration, person_class_id, frame)

            with _state_lock:
                _latest_result = result
                _latest_error = None

        except Exception as exc:  # noqa: BLE001 - 迴圈裡任何失敗都不該讓執行緒整個掛掉
            with _state_lock:
                _latest_error = str(exc)

        time.sleep(CONFIG["refresh_seconds"])


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


def _parse_args():
    parser = argparse.ArgumentParser(description="座位偵測 API 服務")
    parser.add_argument("--model", default=CONFIG["model"], help="YOLO 權重路徑 (.pt)")
    parser.add_argument("--calibration", default=CONFIG["calibration"], help="座位校準表路徑")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", default=None, help="靜態圖片路徑（測試用）")
    source.add_argument("--camera", type=int, default=None, help="攝影機編號，預設 0")

    parser.add_argument("--conf", type=float, default=CONFIG["conf"], help="YOLO 偵測信心門檻")
    parser.add_argument(
        "--person-class-name", default=CONFIG["person_class_name"], help="模型裡 person 這個 class 的名稱"
    )
    parser.add_argument(
        "--refresh-seconds", type=float, default=CONFIG["refresh_seconds"], help="多久重新推論一次"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()

    CONFIG.update(
        model=args.model,
        calibration=args.calibration,
        image=args.image,
        camera=args.camera if args.camera is not None else 0,
        conf=args.conf,
        person_class_name=args.person_class_name,
        refresh_seconds=args.refresh_seconds,
    )

    # 顯式指定 host：只呼叫 uvicorn.run(app) 而不給 host/port，
    # 或是忘記真的呼叫 uvicorn.run()（只是 import 了 app 就結束），
    # 是「FastAPI 看起來沒有監聽」最常見的兩個原因。
    uvicorn.run(app, host=args.host, port=args.port)
