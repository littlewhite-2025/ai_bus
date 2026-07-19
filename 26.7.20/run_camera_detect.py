"""
只用攝影機做 YOLO 偵測，把 person 跟 seat 的框畫出來，
不做座位校準、不做佔用判斷、不輸出任何檔案。純粹確認「攝影機接上去、模型能不能正常即時偵測到東西」。

流程（每一幀重複）:
    1. 從攝影機讀一幀畫面
    2. YOLO 偵測（person + seat 兩個 class 都畫出來，不篩選）
    3. 畫框 + class名稱 + 信心值
    4. 顯示在視窗，按 q 離開 (離不開的話要在終端機按Ctrl + C)

我是這樣跑的:
    python run_camera_detect.py --model runs/detect/seat_detector-3/weights/best.pt --camera 0

--camera 是攝影機編號，如果 0 打不開，依序試試看 1、2。
"""

import argparse

import cv2
from ultralytics import YOLO

# 不同class用不同顏色畫框，方便肉眼區分
CLASS_COLORS = {
    "person": (255, 200, 0),   # 淺藍
    "seat": (0, 200, 0),       # 綠色
}
DEFAULT_COLOR = (0, 165, 255)  # 其他未預期的class用橘色


def main():
    parser = argparse.ArgumentParser(description="攝影機即時偵測(僅顯示person/seat偵測框)")
    parser.add_argument("--model", required=True, help="YOLO 權重路徑 (.pt)")
    parser.add_argument("--camera", type=int, default=0, help="攝影機編號，內建鏡頭通常是0")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻")
    args = parser.parse_args()

    print(f"載入模型：{args.model}")
    model = YOLO(args.model)
    print(f"模型 class 對照表：{model.names}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟攝影機（編號 {args.camera}），"
            f"請確認攝影機沒有被其他程式占用，或試試 --camera 1 / --camera 2"
        )

    print("開始即時偵測，視窗中按 q 離開")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("讀不到攝影機畫面，結束")
                break

            results = model.predict(source=frame, conf=args.conf, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    color = CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    )

            cv2.imshow("YOLO Detection - Live (person/seat only)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
