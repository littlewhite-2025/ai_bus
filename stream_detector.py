"""
即時串流版的座位偵測：在 seat_matcher.py（固定座標 + 匈牙利配對）的基礎上，
加上「時間滑動防抖」跟「畫面左上角 OSD」。

為什麼需要防抖：
    YOLO 是一幀一幀獨立判斷的，人只要稍微晃動、被前排椅背遮到一下，單幀的
    偵測框就可能掉一幀，座位狀態會在 occupied / empty 之間來回跳，畫面上的
    座位框就變成紅綠燈一直閃。這裡替每個座位各自準備一條 deque(maxlen=5)，
    記錄最近 5 幀「這個座位有沒有人」，5 幀中至少 3 幀有人才判定為 occupied。
    這是對稱的多數決：從有人變成沒人，也要連續累積到「最近 5 幀裡有人的
    不到 3 幀」才會翻回 empty，所以單幀的誤判、漏判都不會讓狀態翻轉。

    代價是反應會慢幾幀：真的有人坐下，要等到第 3 個有人的畫面才會亮紅燈。
    以 10 FPS 來說大約 0.3 秒，對公車座位這種場景完全可以接受。

OSD（左上角）：
    Seats: X/16   X 預設是「已佔用座位數」，加 --osd-count empty 可以改成顯示空位數
    Crowd: ...    擁擠度（Empty / Normal / Crowded），門檻跟前端網頁一致：
                  佔用率 < 40% 空曠、< 75% 普通、其餘擁擠
    FPS: ...      即時 FPS（含推論時間，用指數移動平均讓數字不要跳太快）
    OSD 用英文標籤，因為 OpenCV 內建字型畫不出中文。

用法（當成 CLI）：
    python stream_detector.py --model runs/detect/seat_detector-3/weights/best.pt \\
        --seats-config seats_config.json --camera 0

    # 也可以吃錄好的影片，方便重複測試防抖效果：
    python stream_detector.py --model runs/detect/seat_detector-3/weights/best.pt \\
        --seats-config seats_config.json --video assets/videos/test.mp4

    視窗中按 q 離開。沒有螢幕的環境（SSH 等）加 --no-display。

用法（當成模組，例如給 server.py 用）：
    stream = StreamDetector(SeatDetector(model_path=...), load_seats_config("seats_config.json"))
    result, annotated = stream.process_frame(frame)
    # result 的格式跟 seat_matcher.build_result() 完全一樣，只是座位狀態是防抖後的
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2

from seat_matcher import (
    BBox,
    DEFAULT_SEATS_CONFIG_PATH,
    SeatDetector,
    load_seats_config,
    match_persons_to_seats,
)

# 顏色是 BGR（OpenCV 的順序）
COLOR_EMPTY = (0, 200, 0)       # 綠
COLOR_OCCUPIED = (0, 0, 255)    # 紅
COLOR_PERSON = (255, 200, 0)    # 淺藍
COLOR_WARN = (0, 200, 255)      # 黃
COLOR_TEXT = (255, 255, 255)


# 時間滑動防抖
class SeatDebouncer:
    """
    每個座位一條 deque(maxlen=window)，存最近 window 幀「這個座位有沒有人」。
    最近 window 幀中至少 min_votes 幀有人，才判定為 occupied。

    預設 window=5、min_votes=3，也就是「5 幀中滿 3 幀」。
    剛開始畫面還不足 5 幀的時候，同樣要求至少 3 幀有人，所以開機後最前面
    幾幀座位一律先當 empty，等累積到足夠的票才會亮起來。
    """

    def __init__(self, window: int = 5, min_votes: int = 3):
        if window < 1:
            raise ValueError("window 必須 >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes 必須介於 1 和 window 之間")
        self.window = window
        self.min_votes = min_votes
        self._history: Dict[str, Deque[bool]] = {}

    def update(self, raw_status: Dict[str, str]) -> Dict[str, str]:
        #  吃進這一幀的原始座位狀態，回傳防抖後的座位狀態（key 的順序不變）。
        stable: Dict[str, str] = {}
        for seat_id, state in raw_status.items():
            buf = self._history.setdefault(seat_id, deque(maxlen=self.window))
            buf.append(state == "occupied")
            stable[seat_id] = "occupied" if sum(buf) >= self.min_votes else "empty"
        return stable

    def reset(self) -> None:
        # 清掉所有歷史，例如換影片來源、攝影機重新連線之後用。
        self._history.clear()



# OSD 用的小工具
def crowd_level(occupied: int, total: int) -> Tuple[str, Tuple[int, int, int]]:
    """擁擠度，門檻跟前端 utils/format.js 的 crowdLevel() 一致。"""
    if not total:
        return "N/A", COLOR_TEXT
    ratio = occupied / total
    if ratio < 0.4:
        return "Empty", COLOR_EMPTY
    if ratio < 0.75:
        return "Normal", COLOR_WARN
    return "Crowded", COLOR_OCCUPIED


class FpsMeter:
    # 即時 FPS：每次呼叫 tick() 算一次「距離上一幀過了多久」，再用指數移動平均（EMA）平滑，不然每幀的推論時間有高有低，數字會跳得很難讀。
    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self._last: Optional[float] = None
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (
                    self.smoothing * inst + (1 - self.smoothing) * self.fps
                )
        self._last = now
        return self.fps


def _draw_osd(
    frame,
    occupied: int,
    total: int,
    fps: float,
    count_mode: str = "occupied",
) -> None:
    # 在畫面左上角畫 OSD（直接改 frame）。字大小會跟著畫面寬度縮放。
    shown = occupied if count_mode == "occupied" else total - occupied
    level, level_color = crowd_level(occupied, total)

    lines = [
        (f"Seats: {shown}/{total}", COLOR_TEXT),
        (f"Crowd: {level}", level_color),
        (f"FPS: {fps:.1f}", COLOR_TEXT),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, frame.shape[1] / 1280)
    thickness = max(1, round(scale * 2))
    pad = int(10 * scale)

    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text, _ in lines]
    line_h = max(h for _, h in sizes) + int(8 * scale)
    panel_w = max(w for w, _ in sizes) + pad * 2
    panel_h = line_h * len(lines) + pad

    # 半透明黑底，讓文字在亮背景上也看得清楚
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w + pad, panel_h + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = pad + line_h - int(8 * scale)
    for text, color in lines:
        cv2.putText(frame, text, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h


def _draw_seats(
    frame,
    seats_config: Dict[str, BBox],
    status: Dict[str, str],
    person_boxes: Optional[List[BBox]] = None,
) -> None:
    """用防抖後的狀態把座位框塗色：綠 = 空位、紅 = 有人。"""
    overlay = frame.copy()
    for seat_id, box in seats_config.items():
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_OCCUPIED if status.get(seat_id) == "occupied" else COLOR_EMPTY
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    scale = max(0.4, frame.shape[1] / 1800)
    for seat_id, box in seats_config.items():
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_OCCUPIED if status.get(seat_id) == "occupied" else COLOR_EMPTY
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, seat_id, (x1 + 4, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT, 1, cv2.LINE_AA,
        )

    for px1, py1, px2, py2 in person_boxes or []:
        cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), COLOR_PERSON, 1)


class StreamDetector:
    """
    把「偵測 → 匈牙利配對 → 時間防抖 → OSD」串成一個 process_frame()。

    detector 只要有 detect_frame(frame, conf) 就能用（SeatDetector 單例當然可以），
    方便之後做測試時換成假的偵測器。
    """

    def __init__(
        self,
        detector,
        seats_config: Dict[str, BBox],
        conf: float = 0.4,
        score_thresh: float = 0.15,
        window: int = 5,
        min_votes: int = 3,
        count_mode: str = "occupied",
        imgsz: Optional[int] = None,
    ):
        if count_mode not in ("occupied", "empty"):
            raise ValueError("count_mode 只能是 'occupied' 或 'empty'")
        self.detector = detector
        self.seats_config = seats_config
        self.conf = conf
        self.score_thresh = score_thresh
        self.count_mode = count_mode
        self.imgsz = imgsz
        self.debouncer = SeatDebouncer(window=window, min_votes=min_votes)
        self.fps_meter = FpsMeter()
        # 最近一幀「防抖前」的原始座位狀態，給需要對照防抖效果的工具用（例如 stream_test.py）
        self.last_raw_status: Dict[str, str] = {}

    def process_frame(self, frame, draw: bool = True) -> Tuple[dict, Optional[object]]:
        """
        處理一幀，回傳 (result, annotated_frame)。
        result 格式跟 seat_matcher.build_result() 一樣，但座位狀態是防抖後的；
        person_count 是這一幀實際偵測到的人數（不防抖）。
        draw=False 的話不畫圖，annotated_frame 回傳 None（給純後端用）。
        """
        fps = self.fps_meter.tick()

        # imgsz 只有有設定才傳，這樣只實作 detect_frame(frame, conf) 的簡單偵測器也能用
        kwargs = {"conf": self.conf}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        person_boxes = self.detector.detect_frame(frame, **kwargs)
        raw_status = match_persons_to_seats(
            self.seats_config, person_boxes, score_thresh=self.score_thresh
        )
        self.last_raw_status = raw_status
        stable_status = self.debouncer.update(raw_status)

        occupied_count = sum(1 for v in stable_status.values() if v == "occupied")
        total = len(self.seats_config)
        result = {
            "occupied_seats": stable_status,
            "occupied_count": occupied_count,
            "total_seats": total,
            "person_count": len(person_boxes),
        }

        if not draw:
            return result, None

        annotated = frame.copy()
        _draw_seats(annotated, self.seats_config, stable_status, person_boxes)
        _draw_osd(annotated, occupied_count, total, fps, self.count_mode)
        return result, annotated


def run_stream(
    stream: StreamDetector,
    source,
    display: bool = True,
    window_name: str = "Seat Detection - Live",
    max_frames: Optional[int] = None,
) -> Optional[dict]:
    """
    開啟來源（攝影機編號 int 或影片路徑 str）、逐幀處理，回傳最後一幀的 result。
    display=True 時顯示視窗，按 q 離開；影片播完或讀不到畫面也會結束。
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟來源：{source}。攝影機請確認沒被其他程式占用，"
            f"或先用 run_camera_detect.py 找出可用的編號。"
        )

    last_result = None
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("讀不到畫面（影片結束或攝影機中斷），結束")
                break

            last_result, annotated = stream.process_frame(frame, draw=display)
            frames += 1

            if display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and frames >= max_frames:
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return last_result


def main():
    parser = argparse.ArgumentParser(description="即時座位偵測（時間防抖 + OSD）")
    parser.add_argument("--model", required=True, help="YOLO 權重路徑 (.pt)")
    parser.add_argument(
        "--seats-config", default=DEFAULT_SEATS_CONFIG_PATH, help="固化座位座標檔路徑"
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=None, help="攝影機編號，預設 0")
    source.add_argument("--video", default=None, help="影片檔路徑（測試防抖用）")

    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻")
    parser.add_argument("--score-thresh", type=float, default=0.15, help="座位佔用判斷的分數門檻")
    parser.add_argument("--imgsz", type=int, default=None, help="推論解析度，遠處的人抓不到時調大（例如 960、1280），預設 640")
    parser.add_argument("--person-class-name", default="person", help="模型裡 person 的 class 名稱")
    parser.add_argument("--window", type=int, default=5, help="防抖緩衝幀數，預設 5")
    parser.add_argument("--min-votes", type=int, default=3, help="緩衝內至少幾幀有人才算 occupied，預設 3")
    parser.add_argument(
        "--osd-count", choices=("occupied", "empty"), default="occupied",
        help="OSD 的 Seats: X/16 顯示已佔用數還是空位數，預設 occupied",
    )
    parser.add_argument("--no-display", action="store_true", help="不開視窗（沒有螢幕的環境用），結束時印出最後結果")
    args = parser.parse_args()

    seats_config = load_seats_config(args.seats_config)
    print(f"已載入固化座位座標檔，共 {len(seats_config)} 個座位")

    detector = SeatDetector(model_path=args.model, person_class_name=args.person_class_name)
    stream = StreamDetector(
        detector,
        seats_config,
        conf=args.conf,
        score_thresh=args.score_thresh,
        window=args.window,
        min_votes=args.min_votes,
        count_mode=args.osd_count,
        imgsz=args.imgsz,
    )

    source = args.video if args.video is not None else (args.camera if args.camera is not None else 0)
    print(f"開始即時偵測，來源：{source}（視窗中按 q 離開，離不開的話在終端機按 Ctrl+C）")

    last = run_stream(stream, source, display=not args.no_display)
    if last is not None and args.no_display:
        print(json.dumps(last, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()