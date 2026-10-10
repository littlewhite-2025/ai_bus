"""
即時串流版的座位偵測：在 seat_matcher.py（固定座標 + 匈牙利配對）的基礎上，
加上「時間滑動防抖」跟「畫面左上角 OSD」。


OSD（左上角）：
    Seats: X/16   X 預設是「已佔用座位數」，加 --osd-count empty 可以改成顯示空位數
    Crowd: ...    擁擠度（Empty / Normal / Crowded），門檻跟前端網頁一致：佔用率 < 40% 空曠、< 75% 普通、其餘擁擠
    FPS: ...      即時 FPS（含推論時間，用指數移動平均讓數字不要跳太快）
    OSD 用英文標籤，因為 OpenCV 內建字型畫不出中文

用法：
    python stream_detector.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seat_calibration.json --camera 0

    # 也可以用錄好的影片：
    python stream_detector.py --model runs/detect/seat_detector-3/weights/best.pt --seats-config seat_calibration.json --video assets/videos/test.mp4

    視窗中按 q 離開。沒有螢幕的環境（SSH 等）加 --no-display
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
    def __init__(self, window: int = 5, min_votes: int = 3):
        if window < 1:
            raise ValueError("window 必須 >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes 必須介於 1 和 window 之間")
        self.window = window
        self.min_votes = min_votes
        self._history: Dict[str, Deque[bool]] = {}

    def update(self, raw_status: Dict[str, str]) -> Dict[str, str]:
        # 吃進這一幀的原始座位狀態，回傳防抖後的座位狀態（key 的順序不變）
        stable: Dict[str, str] = {}
        for seat_id, state in raw_status.items():
            buf = self._history.setdefault(seat_id, deque(maxlen=self.window))
            buf.append(state == "occupied")
            stable[seat_id] = "occupied" if sum(buf) >= self.min_votes else "empty"
        return stable

    def reset(self) -> None:
        # 清掉所有歷史，例如換影片來源、攝影機重新連線之後用
        self._history.clear()



# OSD 用的小工具
def crowd_level(occupied: int, total: int) -> Tuple[str, Tuple[int, int, int]]:
    # 擁擠度，門檻跟前端 utils/format.js 的 crowdLevel() 一致
    if not total:
        return "N/A", COLOR_TEXT
    ratio = occupied / total
    if ratio < 0.4:
        return "Empty", COLOR_EMPTY
    if ratio < 0.75:
        return "Normal", COLOR_WARN
    return "Crowded", COLOR_OCCUPIED


class FpsMeter:
    # 即時 FPS
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
    # 在畫面左上角畫 OSD（直接改 frame），字大小會跟著畫面寬度縮放
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
    # 用防抖後的狀態把座位框塗色：綠 = 空位、紅 = 有人
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
    def __init__(
        self,
        detector,
        seats_config: Dict[str, BBox],
        conf: float = 0.4,
        score_thresh: float = 0.15,
        window: int = 5,
        min_votes: int = 3,
        count_mode: str = "occupied",
    ):
        if count_mode not in ("occupied", "empty"):
            raise ValueError("count_mode 只能是 'occupied' 或 'empty'")
        self.detector = detector
        self.seats_config = seats_config
        self.conf = conf
        self.score_thresh = score_thresh
        self.count_mode = count_mode
        self.debouncer = SeatDebouncer(window=window, min_votes=min_votes)
        self.fps_meter = FpsMeter()

    def process_frame(self, frame, draw: bool = True) -> Tuple[dict, Optional[object]]:
        # 處理一幀，回傳 (result, annotated_frame)
        
        fps = self.fps_meter.tick()

        person_boxes = self.detector.detect_frame(frame, conf=self.conf)
        raw_status = match_persons_to_seats(
            self.seats_config, person_boxes, score_thresh=self.score_thresh
        )
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
    
    # 開啟來源（攝影機編號 int 或影片路徑 str）、逐幀處理，回傳最後一幀的 result。display=True 時顯示視窗，按 q 離開；影片播完或讀不到畫面也會結束。
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
    )

    source = args.video if args.video is not None else (args.camera if args.camera is not None else 0)
    print(f"開始即時偵測，來源：{source}（視窗中按 q 離開，離不開的話在終端機按 Ctrl+C）")

    last = run_stream(stream, source, display=not args.no_display)
    if last is not None and args.no_display:
        print(json.dumps(last, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
