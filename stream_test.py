"""
stream_test.py

用公車 MP4 影片跑連續推論

用法：
    python stream_test.py --video assets/videos/bus.mp4 \\
        --model runs/detect/seat_detector-3/weights/best.pt \\
        --seats-config seats_config.json

    --interval 0.5   每隔幾秒（影片時間）推論一次，預設 0.5，跟 main.py 的
                     refresh_seconds 一致；0 表示每一幀都推論
    --max-frames 120 最多推論幾次（不給就跑完整支影片）
    --delay 0.2      每次推論後多等幾秒，輸出太快來不及截圖時用

換了影片之後，先確認座位座標對不對得上（這兩個模式不需要載入模型，幾秒就好）：
    # 1. 把影片第 0 秒（或 --at 指定的秒數）畫上座位框，存成圖片，肉眼確認
    python stream_test.py --video assets/videos/bus.mp4 --preview check.png

    # 2. 如果對不上（鏡頭角度不同），把影片裡的一幀存下來拿去重新校正座位
    #    （最好挑車上沒人的那一刻，--at 指定秒數）
    python stream_test.py --video assets/videos/bus.mp4 --extract-frame empty.png --at 3
    python run_calibration.py --image empty.png --model <權重> --seats-per-side 2 ...
    #    校正完把產生的 seat_calibration.json 複製成 seats_config.json

    # 3. 如果只是解析度不同（鏡頭角度一樣），用 --seats-ref-size 告訴它
    #    seats_config.json 當初是在多大的畫面上校正的，座標會自動等比例縮放：
    python stream_test.py --video bus.mp4 --seats-ref-size 1280 720 --preview check.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2

from bus_status import build_bus_status
from seat_matcher import BBox, DEFAULT_SEATS_CONFIG_PATH, SeatDetector, load_seats_config
from stream_detector import StreamDetector, _draw_seats

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"


# 小工具
def _enable_color(force_off: bool) -> bool:
    if force_off or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")  # 這行會讓 Windows 終端機開啟 ANSI 跳脫碼支援
    return True


def _paint(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{RESET}" if use_color else text


def _fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def seat_string(status: Dict[str, str]) -> str:
    parts = []
    for prefix in sorted({sid[0] for sid in status}):
        cells = "".join(
            "#" if status[sid] == "occupied" else "."
            for sid in sorted(k for k in status if k[0] == prefix)
        )
        parts.append(f"{prefix}[{cells}]")
    return " ".join(parts)


def _status_color(text: str) -> str:
    return {"尚有座位": GREEN, "座位有限": YELLOW, "已滿座": RED}.get(text, "")


def scale_seats(seats: Dict[str, BBox], ref_size: Tuple[int, int], frame_size: Tuple[int, int]) -> Dict[str, BBox]:
    sx = frame_size[0] / ref_size[0]
    sy = frame_size[1] / ref_size[1]
    return {
        sid: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for sid, b in seats.items()
    }


def seats_fit_report(seats: Dict[str, BBox], frame_size: Tuple[int, int]) -> Optional[str]:
    w, h = frame_size
    outside = [
        sid for sid, (x1, y1, x2, y2) in seats.items()
        if x2 > w * 1.02 or y2 > h * 1.02 or x1 < -w * 0.02 or y1 < -h * 0.02
    ]
    if outside:
        return (
            f"警告：有 {len(outside)} 個座位框超出影片畫面（影片 {w}x{h}）：{', '.join(sorted(outside))}。"
            f"座標很可能不是在這個解析度/角度下校正的，建議先用 --preview 檢查。"
        )
    return None


def _read_frame_at(video: str, at_seconds: float):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video}")
    if at_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"讀不到影片在 {at_seconds} 秒的畫面，請確認影片長度。")
    return frame


# 連續推論
def run_test(
    stream: StreamDetector,
    video: str,
    interval: float = 0.5,
    max_frames: Optional[int] = None,
    delay: float = 0.0,
    use_color: bool = False,
    out=print,
) -> dict:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(interval * fps))

    out(f"影片：{video}  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"{fps:.1f} fps, {total_frames} 幀)")
    out(f"每 {step} 幀推論一次（約 {step / fps:.2f} 秒）｜防抖：{stream.debouncer.window} 幀中滿 "
        f"{stream.debouncer.min_votes} 幀｜座位 {len(stream.seats_config)} 席")
    out("-" * 100)

    runs = 0
    raw_flips = stable_flips = 0
    prev_raw: Optional[Dict[str, str]] = None
    prev_stable: Optional[Dict[str, str]] = None
    occupied_series = []
    start = time.perf_counter()
    last_result = None
    idx = 0

    try:
        while True:
            if not cap.grab():
                break
            if idx % step != 0:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break

            result, _ = stream.process_frame(frame, draw=False)
            last_result = result
            runs += 1

            raw = stream.last_raw_status
            stable = result["occupied_seats"]
            if prev_raw is not None:
                raw_flips += sum(raw[k] != prev_raw[k] for k in raw)
                stable_flips += sum(stable[k] != prev_stable[k] for k in stable)
            prev_raw, prev_stable = raw, stable
            occupied_series.append(result["occupied_count"])

            status = build_bus_status(result)
            cur_fps = stream.fps_meter.fps
            line = (
                f"[{runs:4d}] {_fmt_time(idx / fps)} | persons={result['person_count']:2d} "
                f"| occupied {status['occupied_count']:2d}/{status['total_seats']} "
                f"| remaining {status['remaining_seats']:2d} | {seat_string(stable)} | "
                f"{_paint(status['status_text'], _status_color(status['status_text']), use_color)} "
                f"| FPS {cur_fps:4.1f}"
            )
            out(line)

            idx += 1
            if max_frames is not None and runs >= max_frames:
                break
            if delay > 0:
                time.sleep(delay)
    finally:
        cap.release()

    elapsed = time.perf_counter() - start
    summary = {
        "inferences": runs,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(runs / elapsed, 2) if elapsed > 0 else 0.0,
        "raw_flips": raw_flips,
        "stable_flips": stable_flips,
        "occupied_min": min(occupied_series) if occupied_series else None,
        "occupied_max": max(occupied_series) if occupied_series else None,
        "last_result": last_result,
    }

    out("-" * 100)
    out(f"完成：共推論 {runs} 次，耗時 {elapsed:.1f} 秒，平均 {summary['avg_fps']} 次/秒")
    if occupied_series:
        out(f"已佔用座位數範圍：{summary['occupied_min']} ~ {summary['occupied_max']} 席")
    out(f"座位狀態翻轉次數：防抖前 {raw_flips} 次 -> 防抖後 {stable_flips} 次")
    if last_result is not None:
        final = build_bus_status(last_result)
        out(f"最後狀態：{final['status_text']}（剩餘 {final['remaining_seats']}/{final['total_seats']} 席）")
    return summary


def main():
    parser = argparse.ArgumentParser(description="公車影片連續推論測試（終端機輸出）")
    parser.add_argument("--video", required=True, help="公車影片路徑 (.mp4)")
    parser.add_argument("--model", default=None, help="YOLO 權重路徑 (.pt)；--preview / --extract-frame 不需要")
    parser.add_argument("--seats-config", default=DEFAULT_SEATS_CONFIG_PATH, help="固化座位座標檔路徑")
    parser.add_argument(
        "--seats-ref-size", type=int, nargs=2, metavar=("W", "H"), default=None,
        help="seats_config.json 當初校正時的畫面大小，會依影片解析度等比例縮放座標",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO 偵測信心門檻")
    parser.add_argument("--score-thresh", type=float, default=0.15, help="座位佔用判斷的分數門檻")
    parser.add_argument("--person-class-name", default="person", help="模型裡 person 的 class 名稱")
    parser.add_argument("--window", type=int, default=5, help="防抖緩衝幀數，預設 5")
    parser.add_argument("--min-votes", type=int, default=3, help="緩衝內至少幾幀有人才算 occupied，預設 3")
    parser.add_argument("--interval", type=float, default=0.5, help="每隔幾秒（影片時間）推論一次，0 = 每幀")
    parser.add_argument("--max-frames", type=int, default=None, help="最多推論幾次")
    parser.add_argument("--delay", type=float, default=0.0, help="每次推論後多等幾秒")
    parser.add_argument("--no-color", action="store_true", help="輸出不要上色")
    parser.add_argument("--at", type=float, default=0.0, help="--preview / --extract-frame 取影片的第幾秒")
    parser.add_argument("--preview", metavar="PNG", default=None, help="把座位框畫在影片的一幀上存檔後離開（不載入模型）")
    parser.add_argument("--extract-frame", metavar="PNG", default=None, help="把影片的一幀原樣存檔後離開，拿去重新校正座位")
    args = parser.parse_args()

    seats = load_seats_config(args.seats_config)

    # 先看影片大小，檢查／縮放座位座標
    probe = cv2.VideoCapture(args.video)
    if not probe.isOpened():
        raise SystemExit(f"無法開啟影片：{args.video}")
    frame_size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    probe.release()

    if args.seats_ref_size:
        seats = scale_seats(seats, tuple(args.seats_ref_size), frame_size)
        print(f"座位座標已從 {args.seats_ref_size[0]}x{args.seats_ref_size[1]} 縮放到影片解析度 {frame_size[0]}x{frame_size[1]}")

    warning = seats_fit_report(seats, frame_size)
    if warning:
        print(warning)

    if args.extract_frame:
        cv2.imwrite(args.extract_frame, _read_frame_at(args.video, args.at))
        print(f"已存檔：{args.extract_frame}（影片 {args.at} 秒的原始畫面）")
        return

    if args.preview:
        frame = _read_frame_at(args.video, args.at)
        _draw_seats(frame, seats, {sid: "empty" for sid in seats})
        cv2.imwrite(args.preview, frame)
        print(f"已存檔：{args.preview}。請確認綠色座位框有蓋在實際的座位上；對不上的話代表要重新校正或用 --seats-ref-size。")
        return

    if not args.model:
        raise SystemExit("連續推論需要 --model 指定 YOLO 權重路徑。")

    detector = SeatDetector(model_path=args.model, person_class_name=args.person_class_name)
    stream = StreamDetector(
        detector, seats,
        conf=args.conf, score_thresh=args.score_thresh,
        window=args.window, min_votes=args.min_votes,
    )
    run_test(
        stream, args.video,
        interval=args.interval, max_frames=args.max_frames, delay=args.delay,
        use_color=_enable_color(args.no_color),
    )


if __name__ == "__main__":
    main()
