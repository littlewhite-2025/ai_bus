"""
固定座標 + 匈牙利配對版本的座位比對模組。

跟 run_inference.py 原本每個人各自挑分數最高的座位的做法比起來，這改用 scipy.optimize.linear_sum_assignment 做全域最佳配對，
一次看過所有 person 對所有座位的完整分數表，解出一組總分數最大化、每個座位最多配一個人、每個人最多配一個座位的組合，而不是讓每個人各自獨立去搶分數最高的座位。

模型載入的部分做成單例：同一個 process 裡不管呼叫幾次 SeatDetector(...)，YOLO(model_path) 只會真的被執行一次，之後每次都重用同一份已經載入記憶體的模型，避免在伺服器迴圈或重複呼叫的情境下，每一幀都重新讀一次權重檔。

用法範例（當成模組用）：
    from seat_matcher import SeatDetector, load_seats_config, build_result

    detector = SeatDetector(model_path="runs/detect/seat_detector-3/weights/best.pt")
    person_boxes = detector.detect(image_path="assets/pictures/main/main2.png")

    seats_config = load_seats_config("seats_config.json")
    result = build_result(seats_config, person_boxes)

用法範例（當成 CLI 用，跟 run_inference.py 用法幾乎一樣）：
    python seat_matcher.py --image assets/pictures/main/main2.png --model runs/detect/seat_detector-3/weights/best.pt --seats-config seats_config.json
"""

from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

DEFAULT_SEATS_CONFIG_PATH = "seats_config.json"



# 共用工具：跟 calibrate.py / run_inference.py 用同一套邏輯，
# 故意不 import 那兩支檔案，讓 seat_matcher.py 可以獨立被其他程式引用，
# 不用一併拖著校正用的 KMeans、cv2 視窗預覽那些依賴。
def _compute_iou(box1: BBox, box2: BBox) -> float:
    """計算兩個 bbox 的 IoU（交集面積 / 聯集面積）"""
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
    """讀取固化的 16 席座標檔（seats_config.json），把 list 轉回 tuple。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {seat_id: tuple(box) for seat_id, box in raw.items()}


# YOLO 模型單例
class SeatDetector:
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

    def detect(self, image_path: str, conf: float = 0.4, imgsz: Optional[int] = None) -> List[BBox]:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖片，請確認路徑是否正確：{image_path}")

        return self.detect_frame(image, conf=conf, imgsz=imgsz)

    def detect_frame(self, frame, conf: float = 0.4, imgsz: Optional[int] = None) -> List[BBox]:
        return [box for box, _ in self.detect_frame_detailed(frame, conf=conf, imgsz=imgsz)]

    def detect_frame_detailed(
        self, frame, conf: float = 0.4, imgsz: Optional[int] = None
    ) -> List[Tuple[BBox, float]]:
        """跟 detect_frame 一樣，但每個框多回傳一個信心值，除錯用。"""
        kwargs = {"source": frame, "conf": conf, "verbose": False}
        if imgsz:
            kwargs["imgsz"] = imgsz
        results = self.model.predict(**kwargs)

        detections: List[Tuple[BBox, float]] = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != self.person_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(((x1, y1, x2, y2), float(box.conf[0])))
        return detections

    @classmethod
    def reset(cls) -> None:
        """清掉單例，讓下一次 SeatDetector(...) 重新載入模型。主要給測試或換模型時用。"""
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


def _solve_assignment(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float,
    containment_bonus: float,
):
    seat_ids = sorted(seats_config.keys(), key=_seat_sort_key)
    score_matrix = np.zeros((len(person_boxes), len(seat_ids)), dtype=np.float64)
    if not person_boxes or not seat_ids:
        return seat_ids, score_matrix, []

    for i, person_box in enumerate(person_boxes):
        for j, seat_id in enumerate(seat_ids):
            score_matrix[i, j] = _pair_score(
                seats_config[seat_id], person_box, containment_bonus
            )

    person_idx, seat_idx = linear_sum_assignment(-score_matrix)
    pairs = [
        (int(p), int(s))
        for p, s in zip(person_idx, seat_idx)
        if score_matrix[p, s] > score_thresh
    ]
    return seat_ids, score_matrix, pairs


def match_persons_to_seats(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float = 0.15,
    containment_bonus: float = 0.2,
) -> Dict[str, str]:
    seat_ids, _, pairs = _solve_assignment(
        seats_config, person_boxes, score_thresh, containment_bonus
    )
    occupied_seat_ids = {seat_ids[s_i] for _, s_i in pairs}
    return {
        seat_id: ("occupied" if seat_id in occupied_seat_ids else "empty")
        for seat_id in seat_ids
    }


def explain_matching(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float = 0.15,
    containment_bonus: float = 0.2,
) -> List[dict]:
    seat_ids, scores, pairs = _solve_assignment(
        seats_config, person_boxes, score_thresh, containment_bonus
    )
    assigned = {p: s for p, s in pairs}

    rows = []
    for i in range(len(person_boxes)):
        if len(seat_ids) and scores.shape[1]:
            best_j = int(np.argmax(scores[i]))
            best_seat, best_score = seat_ids[best_j], float(scores[i, best_j])
        else:
            best_seat, best_score = None, 0.0

        if i in assigned:
            reason = "ok"
        elif best_score <= score_thresh:
            reason = f"跟所有座位框的重疊都太小（最高 {best_score:.2f} <= 門檻 {score_thresh}）：人在所有座位框外面，或座位框沒畫到這個人出現的位置"
        else:
            reason = f"最適合的座位 {best_seat}（{best_score:.2f}）被別人配走了，其他座位分數都不到門檻"

        rows.append({
            "person": i,
            "assigned": seat_ids[assigned[i]] if i in assigned else None,
            "score": float(scores[i, assigned[i]]) if i in assigned else 0.0,
            "best_seat": best_seat,
            "best_score": best_score,
            "reason": reason,
        })
    return rows


def build_result(
    seats_config: Dict[str, BBox],
    person_boxes: List[BBox],
    score_thresh: float = 0.15,
) -> dict:
    """組裝成最終輸出的 JSON 結構，格式跟 run_inference.py 的 build_result() 完全一致，
    前端（或任何接這個 JSON 的下游）不用因為這次改動而跟著改。"""
    seat_status = match_persons_to_seats(seats_config, person_boxes, score_thresh)
    occupied_count = sum(1 for v in seat_status.values() if v == "occupied")

    return {
        "occupied_seats": seat_status,
        "occupied_count": occupied_count,
        "total_seats": len(seats_config),
        "person_count": len(person_boxes),
    }


# CLI（跟 run_inference.py 用法幾乎一樣，方便直接拿來對照測試）
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