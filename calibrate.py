"""
這是為了把偵測到的所有Seat Bbox 自動分配成固定座位ID (A01, A02, B01, B02...)

規則：
1. 用 KMeans(n_clusters=2) 對座位 x 中心點分群，分成左側(A排)、右側(B排)
   （比中位數硬切更能抵抗座位分佈不均、少數離群座位造成的偏移）
2. 每側再依 y 中心點，由前到後分排（容忍值用 median(高度) 計算，
   比 mean 更能抵抗遠近視角造成的框高差異）
3. 每排內依 x 中心點，由左到右編號

這份腳本只需要在「空場景」（沒有人坐）對著固定機位跑一次。
存檔前會先畫出標註好座位ID的預覽圖，人工確認無誤後按 y 才會真的存檔，
避免排錯座位ID卻沒發現，導致整批座位ID錯亂。
"""

from typing import List, Tuple, Dict, Optional
import json
import os
import statistics
import sys

import numpy as np
import cv2
from sklearn.cluster import KMeans

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
BoxWithCenter = Tuple[BBox, float, float]  # (box, cx, cy)


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


def _deduplicate_boxes(seat_boxes: List[BBox], iou_threshold: float = 0.6) -> List[BBox]:
    """
    去除重複偵測到的座位框。

    YOLO 對同一張椅子有時會輸出兩個高度重疊的框（NMS沒完全合併），
    如果不先去重，calibrate_seats() 會把同一張椅子當成兩個座位、
    分配兩個不同ID，導致校準結果出現「A02跟A03標到同一個位置」這種錯誤。

    做法：兩兩比較所有框，IoU 超過門檻視為同一張椅子，只保留其中一個。
    """
    kept: List[BBox] = []
    for box in seat_boxes:
        is_duplicate = any(_compute_iou(box, kept_box) >= iou_threshold for kept_box in kept)
        if not is_duplicate:
            kept.append(box)

    removed_count = len(seat_boxes) - len(kept)
    if removed_count > 0:
        print(f"[提醒] 偵測到 {removed_count} 個重複/高度重疊的座位框，已自動合併去除")

    return kept


def _get_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _split_left_right_kmeans(
    boxes_with_centers: List[BoxWithCenter],
) -> Tuple[List[BoxWithCenter], List[BoxWithCenter]]:
    """
    用 KMeans(n_clusters=2) 對座位 x 中心點分群，分成左側 / 右側兩堆。

    比原本用「x 中心點中位數」硬切更穩：如果座位左右分佈不均
    （例如左側 5 個、右側 3 個），中位數切法容易把邊界座位切到錯的一側，
    KMeans 是依實際分佈群聚，比較不會被這種不對稱影響。
    """
    if len(boxes_with_centers) < 2:
        raise ValueError("座位數量太少，無法用 KMeans 分成左右兩側（至少需要2個座位）")

    cx_array = np.array([[item[1]] for item in boxes_with_centers])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(cx_array)
    centers = kmeans.cluster_centers_.flatten()

    # cx 平均值較小的那一群視為左側(A)，較大的視為右側(B)
    left_label = int(np.argmin(centers))
    right_label = int(np.argmax(centers))

    left = [item for item, lbl in zip(boxes_with_centers, labels) if lbl == left_label]
    right = [item for item, lbl in zip(boxes_with_centers, labels) if lbl == right_label]

    return left, right


def _group_into_rows(
    side_boxes: List[BoxWithCenter], y_tolerance: float = None
) -> List[List[BoxWithCenter]]:
    """
    把同一側的座位，依 y 中心點分排（由前到後）。

    做法：先依 cy 排序，若兩個座位的 cy 差距在容忍值內，視為同一排；
    超過容忍值就視為換到下一排。容忍值預設用座位框高度的「中位數」的一半，
    比原本的平均值更抗雜訊——如果因為遠近視角，某幾個座位框特別大或特別小，
    中位數比較不會被這些極端值拉偏，分排會更穩定。
    可依實際場景手動覆寫。
    """
    if not side_boxes:
        return []

    sorted_boxes = sorted(side_boxes, key=lambda item: item[2])  # 依 cy 由小到大(前到後)

    if y_tolerance is None:
        heights = [item[0][3] - item[0][1] for item in sorted_boxes]
        y_tolerance = statistics.median(heights) * 0.5

    rows: List[List[BoxWithCenter]] = []
    current_row = [sorted_boxes[0]]

    for item in sorted_boxes[1:]:
        # 用「目前這排所有座位 cy 的平均值」當基準，而不是只跟上一個座位比較。
        # 只跟上一個比較容易出現「鏈式漂移」：即使每次差距都在容忍值內，
        # 一路累積下來還是可能把本來不同排的座位串成同一排
        # （例如 cy = 100, 108, 116, 124...，每次只差8，但100跟124其實該是不同排）。
        # 跟整排平均值比較，離群值一旦超過容忍值就會被切開，不會被慢慢拖著走。
        row_reference = statistics.mean(cy for _, _, cy in current_row)
        if abs(item[2] - row_reference) <= y_tolerance:
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
        y_tolerance: 同一排的 y 座標容忍值，預設自動計算(median高度*0.5)，特殊場景可手動指定

    回傳:
        { "A01": (x1,y1,x2,y2), "A02": ..., "B01": ..., ... }
    """
    if not seat_boxes:
        raise ValueError("seat_boxes 不可為空，請確認偵測結果或標註是否正確")

    seat_boxes = _deduplicate_boxes(seat_boxes)

    if len(seat_boxes) < 2:
        raise ValueError(
            f"去重後只剩 {len(seat_boxes)} 個座位，數量太少無法分左右兩側，"
            f"請檢查 YOLO 偵測結果是否正常（可能是 --conf 門檻太高導致漏檢）"
        )

    boxes_with_centers = [(box, *_get_center(box)) for box in seat_boxes]

    left, right = _split_left_right_kmeans(boxes_with_centers)

    if not left or not right:
        raise ValueError(
            f"座位分群結果有一側是空的（左側 {len(left)} 個、右側 {len(right)} 個），"
            f"可能原因：\n"
            f"  1. 偵測到的座位數量太少（目前共 {len(seat_boxes)} 個），"
            f"導致 KMeans 無法正確分出左右兩群\n"
            f"  2. YOLO 只偵測到畫面其中一側的座位，另一側完全漏檢\n"
            f"建議：先確認 YOLO 偵測是否正常（降低 --conf 門檻重跑一次），"
            f"或用 visualize_calibration() 搭配原始偵測框畫圖檢查漏檢狀況"
        )

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


def visualize_calibration(
    calibration: Dict[str, BBox],
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    把座位校準結果畫成一張圖（每個座位框 + 座位ID文字），方便人工確認。

    參數:
        calibration: calibrate_seats() 的輸出結果
        image: 可選，背景圖（例如空場景的實際畫面截圖），
               若不提供則自動建立一張依座位範圍決定大小的白色畫布

    回傳:
        標註好的圖片 (numpy array，BGR格式，可直接用 cv2.imwrite 存檔)
    """
    boxes = list(calibration.values())

    if image is not None:
        canvas = image.copy()
    else:
        max_x = max(b[2] for b in boxes) + 50
        max_y = max(b[3] for b in boxes) + 50
        canvas = np.full((int(max_y), int(max_x), 3), 255, dtype=np.uint8)

    for seat_id, (x1, y1, x2, y2) in calibration.items():
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = (0, 165, 255) if seat_id.startswith("A") else (255, 100, 0)  # BGR，A橘/B藍
        cv2.rectangle(canvas, pt1, pt2, color, 2)
        cv2.putText(
            canvas, seat_id, (pt1[0] + 5, pt1[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )

    return canvas


def save_calibration(calibration: Dict[str, BBox], output_path: str = "seat_calibration.json") -> None:
    # 把校準結果存成 json，供推論階段讀取比對用
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    print(f"已儲存座位校準表：{output_path}（共 {len(calibration)} 個座位）")


def _has_display() -> bool:
    """
    粗略判斷目前環境是否有GUI顯示可用。
    cv2.imshow 在沒有顯示環境（如伺服器/容器）下呼叫會直接讓程式crash
    （Qt的致命錯誤，Python的try/except攔不住），所以要先判斷再決定要不要開視窗。
    """
    if sys.platform.startswith(("win32", "darwin")):
        return True  # Windows/macOS 一般都有GUI，不強求嚴謹判斷
    return bool(os.environ.get("DISPLAY"))  # Linux 看有沒有設定 DISPLAY（X11）


def confirm_and_save(
    calibration: Dict[str, BBox],
    image: Optional[np.ndarray] = None,
    preview_path: str = "calibration_preview.png",
    output_path: str = "seat_calibration.json",
) -> bool:
    """
    畫出校準結果讓使用者人工確認，確認無誤(輸入 y)才真的存檔。

    這一步是為了避免 KMeans/分排邏輯萬一在某些邊界情況分錯（例如座位很密集、
    或某排座位數不齊），卻在沒人發現的情況下直接存檔，導致整批座位ID錯亂，
    後續 person-seat 配對全部跟著錯。

    回傳:
        True 表示已存檔，False 表示使用者取消
    """
    canvas = visualize_calibration(calibration, image=image)
    cv2.imwrite(preview_path, canvas)
    print(f"已產生校準預覽圖：{preview_path}，請打開檢查每個座位ID標的位置是否正確")

    # 有GUI環境才跳出視窗方便直接看；沒有GUI（例如伺服器/容器）就直接跳過，
    # 使用者改用上面存的預覽圖檔案確認即可，不影響後續流程。
    # 注意：cv2.imshow 在無顯示環境下呼叫會直接讓程式 crash（Qt致命錯誤），
    # 所以一定要先判斷 _has_display()，不能只靠 try/except。
    if _has_display():
        try:
            cv2.imshow("Seat Calibration Preview", canvas)
            print("視窗顯示中，按任意鍵關閉視窗")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("開啟顯示視窗失敗，請直接打開上方的預覽圖檔案確認")
    else:
        print("目前環境沒有偵測到顯示裝置，請直接打開上方的預覽圖檔案確認")

    answer = input("座位ID是否正確？確認請輸入 y，取消請輸入 n：").strip().lower()
    if answer == "y":
        save_calibration(calibration, output_path)
        return True
    else:
        print("已取消儲存，請調整標註/偵測結果或 y_tolerance 後重新執行 calibrate_seats()")
        return False


if __name__ == "__main__":
    # 範例：6 個座位，左側(A) 一排3個、右側(B) 一排3個
    example_boxes = [
        (10, 500, 240, 1040),     # A側
        (300, 520, 560, 1040),    # A側
        (580, 510, 640, 1040),    # A側
        (700, 500, 930, 1040),    # B側
        (990, 520, 1250, 1040),   # B側
        (1270, 505, 1500, 1040),  # B側
    ]

    calibration = calibrate_seats(example_boxes, seats_per_side=3)
    for seat_id, box in calibration.items():
        print(seat_id, box)

    confirm_and_save(calibration)
