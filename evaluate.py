"""
一鍵輸出全國賽報表

用法:
    python evaluate.py                              # 使用下面的預設權重
    python evaluate.py --weights runs/detect/train/weights/best.pt
    python evaluate.py --weights best.pt --data seat_dataset/data.yaml

呼叫 model.val() 在驗證集上評估，終端機印出Precision、Recall、mAP50、mAP50-95(含每個類別)，產生 confusion_matrix.png，並複製到雲端同步資料夾
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# 設定
DATA = "seat_dataset/data.yaml"
DEFAULT_WEIGHTS = "runs/detect/seat_detector-3/weights/best.pt"   # 訓練出來的權重路徑

# 雲端同步資料夾，預設存在本機
CLOUD_DIR = ""

LOCAL_OUT = "eval_runs"   # 本機輸出資料夾


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型權重 .pt 路徑")
    ap.add_argument("--data", default=DATA, help="data.yaml 路徑")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"找不到權重檔: {args.weights}\n請用 --weights 指定，或修改檔案裡的 DEFAULT_WEIGHTS")
    if not Path(args.data).exists():
        sys.exit(f"找不到資料集設定檔: {args.data}")

    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        plots=True,            # 自動產生混淆矩陣等圖
        project=LOCAL_OUT,
        name="report",
        exist_ok=True,         # 每次覆蓋同一個資料夾
    )

    box = results.box
    summary = {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }

    # 終端機報表
    print("\n" + "=" * 50)
    print(" 全國賽模型評估報表")
    print("=" * 50)
    print(f" 權重: {args.weights}")
    print(f" 資料: {args.data}")
    print("-" * 50)
    for k, v in summary.items():
        print(f" {k:<10}: {v:.4f}")

    per_class = {}
    print("-" * 50)
    print(f" {'類別':<16}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for i, c in enumerate(box.ap_class_index):
        name = results.names[int(c)]
        p, r, ap50, ap5095 = box.class_result(i)
        per_class[name] = dict(P=float(p), R=float(r), mAP50=float(ap50), mAP50_95=float(ap5095))
        print(f" {name:<16}{p:>8.3f}{r:>8.3f}{ap50:>9.3f}{ap5095:>10.3f}")
    print("=" * 50)

    # 存檔
    save_dir = Path(results.save_dir)
    report = {"weights": args.weights, "data": args.data, "summary": summary, "per_class": per_class}
    (save_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    files = ["confusion_matrix.png", "confusion_matrix_normalized.png", "metrics.json"]
    existing = [save_dir / f for f in files if (save_dir / f).exists()]
    if not (save_dir / "confusion_matrix.png").exists():
        print("注意: 沒有產生 confusion_matrix.png，請確認驗證集有資料")

    print(f"\n本機輸出資料夾: {save_dir.resolve()}")

    # 放雲端
    if CLOUD_DIR:
        dest = Path(CLOUD_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for f in existing:
                shutil.copy2(f, dest / f.name)
            print(f"已複製到雲端同步資料夾: {dest}")
        except Exception as e:
            print(f"複製到雲端資料夾失敗: {e}")
    else:
        print("尚未設定 CLOUD_DIR，所以沒有複製到雲端 (請編輯檔案上方的設定)")


if __name__ == "__main__":
    main()
