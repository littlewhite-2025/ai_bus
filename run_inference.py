"""
run_inference.py

用途：讀取一張畫面(或影片單一幀)，偵測畫面中的 person，
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
    3. 每個偵測到的 person，配對到「分數最高的一個座位」（而不是每個座位
       各自獨立判斷），確保 occupied_count 不會超過 person_count：
       - person bbox 底部中心點落在座位框內 → 最高優先權的匹配
       - 否則用 IoU 當分數，取分數最高的座位
    4. 組裝成最終 JSON，印出並可選擇存檔

用法範例：
    python run_inference.py \\
        --image current_frame.jpg \\
        --model runs/detect/seat_detector/weights/best.pt \\
        --calibration seat_calibration.json

我是這樣跑的: python run_inference.py --image assets/pictures/main/main2.png --model runs/detect/seat_detector-3/weights/best.pt --calibration seat_calibration.json
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
    """讀取 seat_calibration.json，把 list 轉回 tuple"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {seat_id: tuple(box) for seat_id, box in raw.items()}


def detect_persons(
    image_path: str, model_path: str, conf: float, person_class_name: str = "person"
) -> List[BBox]:
    """
    用 YOLO 模型對畫面跑推論，只取出 person 這個 class 的偵測框。
    """
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


def match_persons_to_seats(
    seat_calibration: Dict[str, BBox],
    person_boxes: List[BBox],
    iou_thresh: float = 0.15,
    containment_bonus: float = 0.2,
) -> Dict[str, str]:
    """
    把每個偵測到的 person 配對到「分數最高的一個座位」，而不是讓每個座位
    各自獨立判斷「這個人符不符合佔用條件」。

    原本用 is_seat_occupied() 對每個座位各自檢查的做法有個漏洞：
    如果座位框彼此有重疊（例如攝影機透視角度造成相鄰排的框互相重疊），
    同一個人的 bbox 有可能同時滿足兩個座位的佔用條件，
    導致 person_count=1 卻算出 occupied_count=2 這種不合理結果
    （一個人不可能同時坐兩個不相鄰的座位）。

    分數計算：score = IoU + (containment_bonus，如果底部中心點落在座位框內)

    注意：containment（底部中心點落在框內）故意只當「加分項」，不是絕對優先權。
    原因：像公車這種前後排座位的場景，前一排的椅背/頭枕會擋住後排乘客的下半身，
    YOLO 偵測到的 person bbox 常常只到胸口/肩膀就被切斷，這時候 bbox 底部
    根本不是「人真正坐的位置」，而是被前方椅背擋住的視覺邊界，很容易剛好落在
    「前一排」的座位框裡，造成誤判成前一排有人坐、而不是實際坐的那一排。
    改成加分制之後，即使某座位有 containment，只要另一個座位的整體重疊面積(IoU)
    明顯更大，還是會選 IoU 較高、視覺上更合理的那個座位。
    """
    occupied_seat_ids = set()

    for p in person_boxes:
        px_center = (p[0] + p[2]) / 2
        py_bottom = p[3]

        best_seat_id = None
        best_score = 0.0

        for seat_id, seat_box in seat_calibration.items():
            contains = (
                seat_box[0] <= px_center <= seat_box[2]
                and seat_box[1] <= py_bottom <= seat_box[3]
            )
            iou = _compute_iou(seat_box, p)
            score = iou + (containment_bonus if contains else 0.0)

            if score > best_score:
                best_score = score
                best_seat_id = seat_id

        if best_seat_id is not None and best_score > iou_thresh:
            occupied_seat_ids.add(best_seat_id)

    return {
        seat_id: ("occupied" if seat_id in occupied_seat_ids else "empty")
        for seat_id in sorted(seat_calibration.keys(), key=_seat_sort_key)
    }


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
    """把座位校準表 + 偵測到的 person，組裝成最終輸出的 JSON 結構"""
    seat_status = match_persons_to_seats(seat_calibration, person_boxes, iou_thresh)

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
