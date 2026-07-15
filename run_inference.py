"""
這程式是用於讀取一張畫面(或影片單一幀)，偵測畫面中的 person，
拿去跟事先校準好的 seat_calibration.json 做配對，
判斷每個座位是 occupied 還是 empty，輸出你要的 JSON 格式：

{
  "occupied_seats": {
    "A01": "occupied",
    "A02": "empty",
    ...
  },
  "occupied_count": 2,
  "total_seats": 40,
  "person_count": 2
}

流程：
    1. 讀取 seat_calibration.json（run_calibration.py 產生的固定座位表）
    2. 用 YOLO 模型偵測畫面中的 person（seat 這次不用重新偵測，
       座位位置直接拿校準表裡固定的座標）
    3. 每個座位跟所有偵測到的 person 做配對：
       - person bbox 底部中心點落在座位框內 → occupied
       - 或 person bbox 跟座位框 IoU 超過門檻 → occupied
       （兩個條件任一成立就算佔用，比單用IoU更抗坐姿變化，
       比單用中心點更抗bbox定位誤差）
    4. 組裝成最終 JSON，印出並可選擇存檔

用法範例：
    python run_inference.py \\
        --image current_frame.jpg \\
        --model runs/detect/seat_detector/weights/best.pt \\
        --calibration seat_calibration.json
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

from calibrate import _compute_iou, BBox


def load_calibration(path: str) -> Dict[str, BBox]:
    # 讀取 seat_calibration.json，把 list轉回 tuple
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {seat_id: tuple(box) for seat_id, box in raw.items()}


def detect_persons(
    image_path: str, model_path: str, conf: float, person_class_name: str = "person"
) -> List[BBox]:
    
    #用 YOLO 模型對畫面跑推論，只取出 person 這個 class 的偵測框。
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片，請確認路徑是否正確：{image_path}")

    model = YOLO(model_path)
    results = model.predict(source=image, conf=conf, verbose=False)

    name_to_id = {name.lower(): idx for idx, name in model.names.items()}
    person_class_id = name_to_id.get(person_class_name.lower())

    if person_class_id is None:
        raise ValueError(
            f"模型的 class 名稱裡沒有找到 '{person_class_name}'，"
            f"目前模型的 class 對照表為：{model.names}，"
            f"請用 --person-class-name 指定正確名稱"
        )

    person_boxes: List[BBox] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != person_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            person_boxes.append((x1, y1, x2, y2))

    return person_boxes


def is_seat_occupied(seat_box: BBox, person_boxes: List[BBox], iou_thresh: float = 0.15) -> bool:
    """
    判斷一個座位是否被佔用。

    兩個條件任一成立就算佔用：
    1. person bbox 底部中心點（人站/坐的落地位置）落在座位框內
       —— 這個對「坐姿變化」比較抗干擾，不管人怎麼坐，落地點通常都在座位範圍
    2. person bbox 跟座位框的 IoU 超過門檻
       —— 這個補足「人站在座位前方、bbox 底部剛好卡在框外一點點」的情況
    """
    for p in person_boxes:
        px_center = (p[0] + p[2]) / 2
        py_bottom = p[3]
        if seat_box[0] <= px_center <= seat_box[2] and seat_box[1] <= py_bottom <= seat_box[3]:
            return True
        if _compute_iou(seat_box, p) > iou_thresh:
            return True
    return False


def _seat_sort_key(seat_id: str):
    """讓輸出的座位順序照 A01, A02, ..., B01, B02... 排，而不是字典亂序"""
    match = re.match(r"([A-Za-z]+)(\d+)", seat_id)
    if not match:
        return (seat_id, 0)
    prefix, number = match.groups()
    return (prefix, int(number))


def build_result(
    seat_calibration: Dict[str, BBox],
    person_boxes: List[BBox],
    iou_thresh: float = 0.15,
) -> dict:
    # 把座位校準表+ 偵測到的person，組裝成最終輸出的JSON結構
    seat_status: Dict[str, str] = {}
    for seat_id in sorted(seat_calibration.keys(), key=_seat_sort_key):
        box = seat_calibration[seat_id]
        seat_status[seat_id] = "occupied" if is_seat_occupied(box, person_boxes, iou_thresh) else "empty"

    occupied_count = sum(1 for v in seat_status.values() if v == "occupied")

    return {
        "occupied_seats": seat_status,
        "occupied_count": occupied_count,
        "total_seats": len(seat_calibration),
        "person_count": len(person_boxes),
    }


def main():
    parser = argparse.ArgumentParser(description="座位佔用狀態推論工具")
    parser.add_argument("--image", required=True, help="要判斷的畫面圖片路徑")
    parser.add_argument("--model", required=True, help="YOLO 權重路徑 (.pt)")
    parser.add_argument("--calibration", default="seat_calibration.json", help="座位校準表路徑")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻，預設0.4")
    parser.add_argument("--person-class-name", default="person", help="模型裡 person 這個 class 的名稱")
    parser.add_argument("--iou-thresh", type=float, default=0.15, help="座位佔用判斷的IoU門檻，預設0.15")
    parser.add_argument("--output", default=None, help="結果存檔路徑(可選)，不指定就只印在終端機")
    args = parser.parse_args()

    seat_calibration = load_calibration(args.calibration)
    print(f"已載入座位校準表，共 {len(seat_calibration)} 個座位")

    person_boxes = detect_persons(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        person_class_name=args.person_class_name,
    )
    print(f"偵測到 {len(person_boxes)} 個 person")

    result = build_result(seat_calibration, person_boxes, iou_thresh=args.iou_thresh)

    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"已存檔：{args.output}")


if __name__ == "__main__":
    main()
