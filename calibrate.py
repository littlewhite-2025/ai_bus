"""
這是為了把偵測到的所有Seat Bbox 自動分配成固定座位ID (A01, A02, B01, B02...)

規則：
1. 先依 x 中心點，分成左側(A排)、右側(B排)
2. 每側再依 y 中心點，由前到後分排
3. 每排內依 x 中心點，由左到右編號
 
這份腳本只需要在「空場景」（沒有人坐）對著固定機位跑一次，
把結果存成 seat_calibration.json，之後推論時都拿這份表去比對，
不要每一幀重新排序，否則座位ID會抖動。
"""

from typing import List, Tuple, Dict
import json
 
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
BoxWithCenter = Tuple[BBox, float, float]  # (box, cx, cy)
 
 
def _get_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _split_left_right(
    boxes_with_centers: List[BoxWithCenter], x_mid: float
) -> Tuple[List[BoxWithCenter], List[BoxWithCenter]]:
    # 依x中心點中位數，分成左側與右側兩堆
    left = [item for item in boxes_with_centers if item[1] < x_mid]
    right = [item for item in boxes_with_centers if item[1] >= x_mid]
    return left, right


def _group_into_rows(
    side_boxes: List[BoxWithCenter], y_tolerance: float = None
) -> List[List[BoxWithCenter]]:
    """
    把同一側的座位，依 y 中心點分排（由前到後）。
 
    做法：先依 cy 排序，若兩個座位的 cy 差距在容忍值內，視為同一排；
    超過容忍值就視為換到下一排。容忍值預設用座位框高度平均值的一半，
    可依實際場景手動覆寫（例如座位框很扁或很高時）。
    """
    if not side_boxes:
        return []

    sorted_boxes = sorted(side_boxes, key=lambda item: item[2])  # 依 cy 由小到大(前到後)

    if y_tolerance is None:
        heights = [item[0][3] - item[0][1] for item in sorted_boxes]
        y_tolerance = (sum(heights) / len(heights)) * 0.5

    rows: List[List[BoxWithCenter]] = []
    current_row = [sorted_boxes[0]]

    for item in sorted_boxes[1:]:
        prev_cy = current_row[-1][2]
        if abs(item[2] - prev_cy) <= y_tolerance:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    return rows

 
def calibrate_seats(
    seat_boxes: List[BBox],
    seats_per_side: int = 2,
    y_tolerance: float = None,
) -> Dict[str, BBox]:
    """
    主函式：把一批座位 bbox 校準成固定座位ID。
 
    參數:
        seat_boxes: 偵測到的所有座位框 [(x1,y1,x2,y2), ...]
        seats_per_side: 每排座位數，用來檢查分排結果是否符合預期（僅做警告用）
        y_tolerance: 同一排的 y 座標容忍值，預設自動計算，特殊場景可手動指定
 
    回傳:
        { "A01": (x1,y1,x2,y2), "A02": ..., "B01": ..., ... }
    """
    if not seat_boxes:
        raise ValueError("seat_boxes 不可為空，請確認偵測結果或標註是否正確")
 
    boxes_with_centers = [(box, *_get_center(box)) for box in seat_boxes]

    cx_values = sorted(item[1] for item in boxes_with_centers)
    x_mid = cx_values[len(cx_values) // 2]
    left, right = _split_left_right(boxes_with_centers, x_mid)

    left_rows = _group_into_rows(left, y_tolerance)
    right_rows = _group_into_rows(right, y_tolerance)
 
    # 檢查每排座位數是否符合預期，不符合就提醒（不中斷程式，方便你先看整體結果）
    for side_name, rows in [("A", left_rows), ("B", right_rows)]:
        for row_idx, row in enumerate(rows, start=1):
            if len(row) != seats_per_side:
                print(
                    f"[警告] {side_name} 側第 {row_idx} 排偵測到 {len(row)} 個座位，"
                    f"預期為 {seats_per_side} 個，請檢查偵測結果或 y_tolerance 設定"
                )

    result: Dict[str, BBox] = {}
 
    for prefix, rows in [("A", left_rows), ("B", right_rows)]:
        seat_number = 1
        for row in rows:
            row_sorted = sorted(row, key=lambda item: item[1])  # 排內依 cx 由左到右
            for box, cx, cy in row_sorted:
                result[f"{prefix}{seat_number:02d}"] = box
                seat_number += 1

    return result
 
 
def save_calibration(calibration: Dict[str, BBox], output_path: str = "seat_calibration.json") -> None:
    # 把校準結果存成 json，供推論階段讀取比對用
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    print(f"已儲存座位校準表：{output_path}（共 {len(calibration)} 個座位）")


if __name__ == "__main__":
    # 範例：4 個座位，左側2個(一排)、右側2個(一排)
    example_boxes = [
        (10, 500, 240, 1040),    # 左側
        (300, 520, 560, 1040),   # 左側
        (700, 500, 930, 1040),   # 右側
        (990, 520, 1250, 1040),  # 右側
    ]

    calibration = calibrate_seats(example_boxes, seats_per_side=2)
    for seat_id, box in calibration.items():
        print(seat_id, box)

    save_calibration(calibration)
