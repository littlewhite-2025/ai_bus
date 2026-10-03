"""
固定座標 + 匈牙利配對版本的座位比對模組

改用 scipy.optimize.linear_sum_assignment，一次看過所有 person 對所有座位的完整分數表，做出一組總分數最大化、每個座位最多配一個人、每個人最多配一個座位的組合最高的座位。

模型載入的部分做成單例：同一個 process 裡不管呼叫幾次 SeatDetector(...)，YOLO(model_path) 只會真的被執行一次，之後每次都重用同一份已經載入記憶體的模型，避免在伺服器迴圈或重複呼叫的情境下，每一幀都重新讀一次權重檔

用法範例(當成模組用):
    from seat_matcher import SeatDetector, load_seats_config, build_result

    detector = SeatDetector(model_path="runs/detect/seat_detector-3/weights/best.pt")
    person_boxes = detector.detect(image_path="assets/pictures/main/main2.png")

    seats_config = load_seats_config("seat_calibration.json")
    result = build_result(seats_config, person_boxes)

用法範例(當成 CLI 用):
    python seat_matcher.py --image assets/pictures/main/main2.png --model runs/detect/seat_detector-3/weights/best.pt --seats-config seat_calibration.json
"""

from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

DEFAULT_SEATS_CONFIG_PATH = "seat_calibration.json"


# 跟 calibrate.py / run_inference.py 用同一套邏輯，故意不 import 那兩支檔案，讓 seat_matcher.py 可以獨立被其他程式引用，
def _compute_iou(box1: BBox, box2: BBox) -> float:
    # 計算兩個 bbox 的 IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _seat_sort_key(seat_id: str):
    match = re.match(r"([A-Za-z]+)(\d+)", seat_id)
    if not match:
        return (seat_id, 0)
    prefix, number = match.groups()
    return (prefix, int(number))


def load_seats_config(path: str = DEFAULT_SEATS_CONFIG_PATH) -> Dict[str, BBox]:
    # 讀取固化的 16 席座標檔（seat_calibration.json），把 list 轉回 tuple
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {seat_id: tuple(box) for seat_id, box in raw.items()}


# YOLO 模型單例
class SeatDetector:

    """
    YOLO 模型的單例包裝

    整個 process 存活期間，不管呼叫 SeatDetector(...) 幾次，model = YOLO(...)只會真的執行一次；之後每次都重用同一份已經載入記憶體的模型，避免在伺服器迴圈或重複呼叫的情境下，每一幀都重新讀一次權重檔
    第一次呼叫時傳入的 model_path / person_class_name 會被記住；之後如果用不同參數再呼叫 SeatDetector(...)，不會重新載入，而是沿用第一次的設定（並印出警告），避免不小心用錯模型卻沒發現。如果真的需要換模型，呼叫 SeatDetector.reset() 清掉單例之後再重新建立。
    """
    _instance: Optional["SeatDetector"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, model_path: str, person_class_name: str = "person"):
        if self._initialized:
            if model_path != self.model_path:
                print(
                    f"[SeatDetector] 已經載入過模型（{self.model_path}），"
                    f"忽略這次傳入的不同路徑（{model_path}）。"
                    f"如果真的要換模型，請先呼叫 SeatDetector.reset()。"
                )
            return

        from ultralytics import YOLO  # 延遲 import，避免沒用到偵測功能時也要載入 ultralytics

        self.model_path = model_path
        self.person_class_name = person_class_name
        self.model = YOLO(model_path)

        name_to_id = {name.lower(): idx for idx, name in self.model.names.items()}
        self.person_class_id = name_to_id.get(person_class_name.lower())
        if self.person_class_id is None:
            raise ValueError(
                f"模型的 class 名稱裡沒有找到 '{person_class_name}'，"
                f"目前模型的 class 對照表為：{self.model.names}"
            )

        self._initialized = True
        print(f"[SeatDetector] 模型已載入（單例，之後不會重複載入）：{model_path}")

    def detect(self, image_path: str, conf: float = 0.4) -> List[BBox]:
        # 讀一張圖片、跑推論，只取出person這個class的偵測框
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖片，請確認路徑是否正確：{image_path}")

        return self.detect_frame(image, conf=conf)

    def detect_frame(self, frame, conf: float = 0.4) -> List[BBox]:
        
        # 對一個已經讀進記憶體的 frame（例如攝影機讀到的 numpy array）跑推論，跟 detect()共用同一份已經載入的模型，差別只是不用重新從檔案讀圖，給攝影機即時串流的情境用
        results = self.model.predict(source=frame, conf=conf, verbose=False)

        person_boxes: List[BBox] = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != self.person_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append((x1, y1, x2, y2))
        return person_boxes

    @classmethod
    def reset(cls) -> None:
        # 清掉單例，讓下一次 SeatDetector(...) 重新載入模型，主要給測試或換模型時用。
        with cls._lock:
            cls._instance = None


# 匈牙利配對
def _pair_score(seat_box: BBox, person_box: BBox, containment_bonus: float = 0.2) -> float:

    px_center = (person_box[0] + person_box[2]) / 2
    py_bottom = person_box[3]

    contains = (
        seat_box[0] <= px_center <= seat_box[2]
        and seat_box[1] <= py_bottom <= seat_box[3]
    )
    iou = _compute_iou(seat_box, person_box)
    return iou + (containment_bonus if contains else 0.0)


def match_persons_to_seats(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float = 0.15,
    containment_bonus: float = 0.2,
) -> Dict[str, str]:
    """
    用匈牙利演算法做全域最佳配對，取代每個人各自挑分數最高的座位

    建一個 (人數 x 座位數) 的分數矩陣，每一格是 _pair_score()，scipy 的 linear_sum_assignment 預設是"找最小成本"的指派，所以把分數矩陣取負號再丟進去（minimize(-score) 等於 maximize(score)）
    3. 解出配對結果後，過濾掉分數低於 score_thresh 的配對，分數太低代表沒有真的重疊
    """
    seat_ids = sorted(seats_config.keys(), key=_seat_sort_key)

    if not person_boxes or not seat_ids:
        return {seat_id: "empty" for seat_id in seat_ids}

    score_matrix = np.zeros((len(person_boxes), len(seat_ids)), dtype=np.float64)
    for i, person_box in enumerate(person_boxes):
        for j, seat_id in enumerate(seat_ids):
            score_matrix[i, j] = _pair_score(
                seats_config[seat_id], person_box, containment_bonus
            )

    person_idx, seat_idx = linear_sum_assignment(-score_matrix)

    occupied_seat_ids = set()
    for p_i, s_i in zip(person_idx, seat_idx):
        if score_matrix[p_i, s_i] > score_thresh:
            occupied_seat_ids.add(seat_ids[s_i])

    return {
        seat_id: ("occupied" if seat_id in occupied_seat_ids else "empty")
        for seat_id in seat_ids
    }


def build_result(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float = 0.15,
) -> dict:
    # 組裝成最終輸出的 JSON 結構
    seat_status = match_persons_to_seats(seats_config, person_boxes, score_thresh)
    occupied_count = sum(1 for v in seat_status.values() if v == "occupied")

    return {
        "occupied_seats": seat_status,
        "occupied_count": occupied_count,
        "total_seats": len(seats_config),
        "person_count": len(person_boxes),
    }


# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(description="固定座標 + 匈牙利配對版本的座位佔用推論工具")
    parser.add_argument("--image", required=True, help="要判斷的畫面圖片路徑")
    parser.add_argument("--model", required=True, help="YOLO 權重路徑 (.pt)")
    parser.add_argument(
        "--seats-config", default=DEFAULT_SEATS_CONFIG_PATH, help="固化座位座標檔路徑"
    )
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻，預設0.4")
    parser.add_argument(
        "--person-class-name", default="person", help="模型裡 person 這個 class 的名稱"
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.15, help="座位佔用判斷的分數門檻，預設0.15"
    )
    parser.add_argument("--output", default=None, help="結果存檔路徑(可選)，不指定就只印在終端機")
    args = parser.parse_args()

    seats_config = load_seats_config(args.seats_config)
    print(f"已載入固化座位座標檔，共 {len(seats_config)} 個座位")

    detector = SeatDetector(model_path=args.model, person_class_name=args.person_class_name)
    person_boxes = detector.detect(image_path=args.image, conf=args.conf)
    print(f"偵測到 {len(person_boxes)} 個 person")

    result = build_result(seats_config, person_boxes, score_thresh=args.score_thresh)

    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"已存檔：{args.output}")


if __name__ == "__main__":
    main()
