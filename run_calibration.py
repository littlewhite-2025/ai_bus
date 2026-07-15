"""
run_calibration.py

一鍵完成座位ID校準的完整流程：
    1. 讀取空場景圖片（沒有人坐的畫面）
    2. 用訓練好的 YOLO 模型偵測 seat
    3. 把偵測到的 seat_boxes 丟給 calibrate_seats() 自動分配座位ID
    4. 呼叫 confirm_and_save()，畫出預覽圖讓你人工確認後才存檔

執行前請先安裝套件：
    pip install ultralytics opencv-python-headless scikit-learn numpy

用法範例：
    python run_calibration.py \\
        --image empty_room.jpg \\
        --model runs/detect/seat_detector/weights/best.pt \\
        --seats-per-side 2

執行完成後會在目前資料夾產生：
    calibration_preview.png   ← 人工確認用的標註預覽圖
    seat_calibration.json      ← 正式的座位校準表（確認存檔後才會有）
"""

import argparse
import sys

import cv2
from ultralytics import YOLO

from calibrate import calibrate_seats, confirm_and_save


def detect_seats(image_path: str, model_path: str, conf: float, seat_class_name: str = "seat"):
    """
    用 YOLO 模型對空場景圖片跑推論，只取出 seat 這個 class 的偵測框。

    回傳:
        image: 讀取到的原始圖片 (numpy array, BGR)，供後續畫預覽圖用
        seat_boxes: [(x1, y1, x2, y2), ...]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片，請確認路徑是否正確：{image_path}")

    model = YOLO(model_path)
    results = model.predict(source=image, conf=conf, verbose=False)

    # class 名稱 -> id 的對照表，YOLO 模型物件本身會帶這個資訊(model.names)
    name_to_id = {name.lower(): idx for idx, name in model.names.items()}
    seat_class_id = name_to_id.get(seat_class_name.lower())

    if seat_class_id is None:
        raise ValueError(
            f"模型的 class 名稱裡沒有找到 '{seat_class_name}'，"
            f"目前模型的 class 對照表為：{model.names}，"
            f"請用 --seat-class-name 指定正確名稱"
        )

    seat_boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != seat_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            seat_boxes.append((x1, y1, x2, y2))

    return image, seat_boxes


def main():
    parser = argparse.ArgumentParser(description="座位ID一鍵校準工具")
    parser.add_argument("--image", required=True, help="空場景圖片路徑")
    parser.add_argument("--model", required=True, help="訓練好的 YOLO 權重路徑 (.pt)")
    parser.add_argument("--seats-per-side", type=int, default=2, help="每排座位數，預設2")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻，預設0.4")
    parser.add_argument("--seat-class-name", default="seat", help="模型裡 seat 這個 class 的名稱，預設'seat'")
    parser.add_argument("--y-tolerance", type=float, default=None, help="分排容忍值，預設自動計算")
    parser.add_argument("--dedup-iou", type=float, default=0.85, help="判定重複偵測的IoU門檻，預設0.85。相鄰座位常被誤刪就調高，同一張椅子重複偵測沒被合併就調低")
    parser.add_argument("--preview-path", default="calibration_preview.png", help="預覽圖輸出路徑")
    parser.add_argument("--output", default="seat_calibration.json", help="校準表輸出路徑")
    args = parser.parse_args()

    print(f"讀取圖片：{args.image}")
    print(f"載入模型：{args.model}")
    image, seat_boxes = detect_seats(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        seat_class_name=args.seat_class_name,
    )

    print(f"共偵測到 {len(seat_boxes)} 個 seat")
    if not seat_boxes:
        print("[錯誤] 沒有偵測到任何座位，請檢查圖片、模型或降低 --conf 門檻後再試一次")
        sys.exit(1)


    calibration = calibrate_seats(
        seat_boxes=seat_boxes,
        seats_per_side=args.seats_per_side,
        y_tolerance=args.y_tolerance,
        dedup_iou_threshold=args.dedup_iou,
    )

    print(f"校準完成，共 {len(calibration)} 個座位：")
    for seat_id, box in calibration.items():
        print(f"  {seat_id}: {box}")

    # 用實際拍到的空場景圖片當背景畫預覽圖，比空白畫布更容易人工判斷對不對
    confirm_and_save(
        calibration=calibration,
        image=image,
        preview_path=args.preview_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
