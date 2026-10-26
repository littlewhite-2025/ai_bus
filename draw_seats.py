"""
draw_seats.py

手動畫座位框

用法：
    # 用一張圖片畫
    python draw_seats.py --image assets/pictures/11-43-14-884.png --output seats_config.json

    # 直接用影片的某一幀畫（--at 是第幾秒）
    python draw_seats.py --video bus.mp4 --at 3 --output seats_config.json

    # 接續編輯：載入現有的座標檔（自動校正跑出來的 seat_calibration.json 也可以），
    # 刪掉不對的、補畫缺的
    python draw_seats.py --image xxx.png --load seat_calibration.json --output seats_config.json

    # 不開視窗，只把座標檔畫在圖上存成 PNG，確認用
    python draw_seats.py --image xxx.png --load seats_config.json --preview-out check.png

操作：
    左鍵拖曳     畫一個座位框，自動編號（A01、A02...，目前選的那一側的下一個空號）
    右鍵         刪除滑鼠所在位置的框（重疊時刪面積最小的那個）
    a / b        切換左側（A）/ 右側（B）
    u            復原上一步（畫或刪）
    s            存檔（舊檔會先備份成 .bak）
    h            顯示/隱藏操作說明
    q 或 ESC     離開（有沒存的變更會先提醒，再按一次才會離開）

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

WINDOW = "Draw seats"
MIN_BOX_SIZE = 8  # 小於這個大小（像素）的拖曳視為誤觸，不新增
COLORS = {"A": (0, 165, 255), "B": (255, 100, 0)}  # BGR：A 橘、B 藍，跟自動校正的預覽同色系
DEFAULT_COLOR = (0, 200, 0)


def _color(seat_id: str) -> Tuple[int, int, int]:
    return COLORS.get(seat_id[:1].upper(), DEFAULT_COLOR)


def _split_id(seat_id: str) -> Tuple[str, int]:
    m = re.match(r"([A-Za-z]+)(\d+)$", seat_id)
    return (m.group(1), int(m.group(2))) if m else (seat_id, 0)


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# 編輯邏輯
class SeatEditor:
    def __init__(self, image_size: Tuple[int, int], prefix: str = "A"):
        self.width, self.height = image_size
        self.prefix = prefix.upper()
        self.seats: Dict[str, BBox] = {}
        self._history: List[tuple] = []
        self.dirty = False

    def next_id(self, prefix: Optional[str] = None) -> str:
        prefix = (prefix or self.prefix).upper()
        used = {n for sid in self.seats for p, n in [_split_id(sid)] if p == prefix}
        n = 1
        while n in used:
            n += 1
        return f"{prefix}{n:02d}"

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix.upper()

    def add_box(self, x1: float, y1: float, x2: float, y2: float) -> Optional[str]:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0.0, x1), min(float(self.width), x2)
        y1, y2 = max(0.0, y1), min(float(self.height), y2)
        if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            return None
        sid = self.next_id()
        self.seats[sid] = (x1, y1, x2, y2)
        self._history.append(("add", sid))
        self.dirty = True
        return sid

    def delete_at(self, x: float, y: float) -> Optional[str]:
        hits = [
            ((b[2] - b[0]) * (b[3] - b[1]), sid)
            for sid, b in self.seats.items()
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]
        ]
        if not hits:
            return None
        _, sid = min(hits)
        box = self.seats.pop(sid)
        self._history.append(("delete", sid, box))
        self.dirty = True
        return sid

    def undo(self) -> Optional[str]:
        if not self._history:
            return None
        action = self._history.pop()
        if action[0] == "add":
            self.seats.pop(action[1], None)
            msg = f"undo add {action[1]}"
        else:
            self.seats[action[1]] = action[2]
            msg = f"undo delete {action[1]}"
        self.dirty = True
        return msg

    def sorted_seats(self) -> Dict[str, BBox]:
        return dict(sorted(self.seats.items(), key=lambda kv: _split_id(kv[0])))

    def overlap_warnings(self, threshold: float = 0.5) -> List[str]:
        ids = list(self.sorted_seats())
        out = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                v = _iou(self.seats[a], self.seats[b])
                if v > threshold:
                    out.append(f"{a} 和 {b} 重疊 {v:.0%}，是不是重複畫了？")
        return out

    def save(self, path: str) -> None:
        if os.path.exists(path):
            shutil.copyfile(path, path + ".bak")
        data = {sid: [round(v, 1) for v in box] for sid, box in self.sorted_seats().items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.dirty = False

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        seats = {}
        for sid, box in raw.items():
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                raise ValueError(f"{path} 裡 {sid} 的格式不對，應該是 [x1, y1, x2, y2]")
            seats[sid] = tuple(float(v) for v in box)
        self.seats = dict(sorted(seats.items(), key=lambda kv: _split_id(kv[0])))
        self._history.clear()
        self.dirty = False


# 繪圖
def _put_text(img, text, org, scale, color, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(
    image,
    editor: SeatEditor,
    scale: float = 1.0,
    drag: Optional[Tuple[float, float, float, float]] = None,
    show_help: bool = True,
    message: str = "",
):
    disp = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image.copy()

    def to_disp(b):
        return tuple(int(round(v * scale)) for v in b)

    overlay = disp.copy()
    for sid, box in editor.seats.items():
        x1, y1, x2, y2 = to_disp(box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _color(sid), -1)
    cv2.addWeighted(overlay, 0.18, disp, 0.82, 0, disp)

    for sid, box in editor.sorted_seats().items():
        x1, y1, x2, y2 = to_disp(box)
        cv2.rectangle(disp, (x1, y1), (x2, y2), _color(sid), 2)
        _put_text(disp, sid, (x1 + 4, y1 + 22), 0.7, _color(sid))

    if drag is not None:
        x1, y1, x2, y2 = to_disp(drag)
        preview_id = editor.next_id()
        cv2.rectangle(disp, (x1, y1), (x2, y2), _color(preview_id), 2, cv2.LINE_AA)
        _put_text(disp, preview_id + "?", (min(x1, x2) + 4, min(y1, y2) + 22), 0.7, _color(preview_id))

    counts: Dict[str, int] = {}
    for sid in editor.seats:
        counts[_split_id(sid)[0]] = counts.get(_split_id(sid)[0], 0) + 1
    count_text = " ".join(f"{p}:{n}" for p, n in sorted(counts.items())) or "-"
    unsaved = "  *unsaved*" if editor.dirty else ""
    _put_text(disp, f"Next: {editor.next_id()}   Boxes: {len(editor.seats)} [{count_text}]{unsaved}",
              (10, 28), 0.7, (255, 255, 255))
    if message:
        _put_text(disp, message, (10, 58), 0.6, (0, 255, 255), 1)
    if show_help:
        _put_text(disp, "drag: draw | right-click: delete | a/b: side | u: undo | s: save | h: help | q: quit",
                  (10, disp.shape[0] - 14), 0.55, (255, 255, 255), 1)
    return disp


# 視窗互動
class SeatDrawer:
    def __init__(self, image, editor: SeatEditor, output: str, display_width: int = 1400):
        self.image = image
        self.editor = editor
        self.output = output
        h, w = image.shape[:2]
        self.scale = min(1.0, display_width / w) if display_width else 1.0
        self.show_help = True
        self.message = ""
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_cur: Optional[Tuple[float, float]] = None
        self._quit_armed = False

    def _to_image(self, x: int, y: int) -> Tuple[float, float]:
        h, w = self.image.shape[:2]
        return (min(max(x / self.scale, 0.0), float(w)), min(max(y / self.scale, 0.0), float(h)))

    def on_mouse(self, event, x, y, flags=0, param=None) -> None:
        ix, iy = self._to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_cur = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_cur = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            x0, y0 = self._drag_start
            sid = self.editor.add_box(x0, y0, ix, iy)
            self.message = f"added {sid}" if sid else "too small, ignored"
            self._drag_start = self._drag_cur = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            sid = self.editor.delete_at(ix, iy)
            self.message = f"deleted {sid}" if sid else "nothing here to delete"

    def handle_key(self, key: int) -> bool:
        if key in (ord("a"), ord("A")):
            self.editor.set_prefix("A")
        elif key in (ord("b"), ord("B")):
            self.editor.set_prefix("B")
        elif key in (ord("u"), ord("U")):
            self.message = self.editor.undo() or "nothing to undo"
        elif key in (ord("s"), ord("S")):
            self.editor.save(self.output)
            warnings = self.editor.overlap_warnings()
            for w in warnings:
                print("提醒：" + w)
            self.message = f"saved -> {self.output}" + (f" ({len(warnings)} overlap warning, see terminal)" if warnings else "")
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        elif key in (ord("q"), ord("Q"), 27):
            if self.editor.dirty and not self._quit_armed:
                self._quit_armed = True
                self.message = "unsaved changes! press s to save, or q again to quit"
                return True
            return False
        if key not in (ord("q"), ord("Q"), 27):
            self._quit_armed = False
        return True

    def frame(self):
        drag = None
        if self._drag_start is not None and self._drag_cur is not None:
            drag = (*self._drag_start, *self._drag_cur)
        return render(self.image, self.editor, self.scale, drag, self.show_help, self.message)

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        seen_visible = False
        try:
            while True:
                cv2.imshow(WINDOW, self.frame())
                k = cv2.waitKey(20)
                if k != -1 and not self.handle_key(k & 0xFF):
                    break
                visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
                if visible >= 1:
                    seen_visible = True
                elif seen_visible:
                    break
        finally:
            cv2.destroyAllWindows()


# CLI
def _load_image(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"讀不到圖片：{args.image}（路徑有中文的話，Windows 上 OpenCV 可能讀不到，請改用英文路徑）")
        return img
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{args.video}")
    if args.at > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"讀不到影片在 {args.at} 秒的畫面，請確認影片長度。")
    return frame


def main():
    parser = argparse.ArgumentParser(description="手動畫座位框，輸出 seats_config.json")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="用這張圖片畫")
    src.add_argument("--video", help="用影片的某一幀畫")
    parser.add_argument("--at", type=float, default=0.0, help="--video 取第幾秒的畫面，預設 0")
    parser.add_argument("--output", default="seats_config.json", help="輸出座標檔，預設 seats_config.json")
    parser.add_argument("--load", default=None, help="載入現有座標檔接續編輯")
    parser.add_argument("--prefix", default="A", help="一開始畫哪一側，預設 A（左）")
    parser.add_argument("--display-width", type=int, default=1400, help="視窗顯示寬度（原圖太大時縮小顯示），0 = 不縮放")
    parser.add_argument("--preview-out", default=None, help="不開視窗，把座位框畫在圖上存成 PNG 後離開")
    args = parser.parse_args()

    image = _load_image(args)
    h, w = image.shape[:2]
    editor = SeatEditor((w, h), prefix=args.prefix)
    if args.load:
        editor.load(args.load)
        print(f"已載入 {len(editor.seats)} 個座位：{args.load}")
        outside = [s for s, b in editor.seats.items() if b[2] > w * 1.02 or b[3] > h * 1.02]
        if outside:
            print(f"警告：{', '.join(outside)} 超出這張圖（{w}x{h}），座標可能不是在這個畫面大小下畫的。")

    if args.preview_out:
        cv2.imwrite(args.preview_out, render(image, editor, show_help=False))
        print(f"已存檔：{args.preview_out}")
        return

    print(f"畫面大小 {w}x{h}。左鍵拖曳畫框、右鍵刪除、a/b 切換左右側、s 存檔、h 看說明。")
    drawer = SeatDrawer(image, editor, args.output, args.display_width)
    drawer.run()

    print(f"共 {len(editor.seats)} 個座位" + ("（有未儲存的變更！）" if editor.dirty else ""))
    for warning in editor.overlap_warnings():
        print("提醒：" + warning)


if __name__ == "__main__":
    main()
