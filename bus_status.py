# 把推論結果（seat_matcher.build_result / StreamDetector.process_frame 的輸出）轉成 /api/bus-status 要回傳的格式，並判斷狀態文字。


from typing import Dict

STATUS_AVAILABLE = "尚有座位"
STATUS_LIMITED = "座位有限"
STATUS_FULL = "已滿座"

# 佔用率達到這個比例就算座位有限
LIMITED_RATIO = 0.75


def status_text(occupied: int, total: int) -> str:
    remaining = total - occupied
    if total <= 0 or remaining <= 0:
        return STATUS_FULL
    if occupied / total >= LIMITED_RATIO:
        return STATUS_LIMITED
    return STATUS_AVAILABLE


def build_bus_status(result: dict) -> dict:
    total = result["total_seats"]
    occupied = result["occupied_count"]
    return {
        "total_seats": total,
        "occupied_count": occupied,
        "remaining_seats": max(total - occupied, 0),
        "status_text": status_text(occupied, total),
        "occupied_seats": dict(result["occupied_seats"]),
    }
