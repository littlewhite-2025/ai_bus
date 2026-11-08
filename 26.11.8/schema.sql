-- schema.sql
--
-- 多台公車 / 多條路線 / 座位狀態 的 SQLite 資料結構。
-- 這份是「唯一的結構定義」，db.py 的 init_db() 直接讀它；全部用 IF NOT EXISTS，
-- 重複執行不會壞掉，也不會清掉既有資料。
--
-- 時間一律存 UTC，格式 'YYYY-MM-DDTHH:MM:SSZ'（ISO 8601），前端要顯示台灣時間時再轉換。
--
-- 表的分工：
--   routes          路線（E01 / E02 / E25）
--   buses           車輛（車牌、屬於哪條路線、有監測幾個座位、座位座標檔、資料來源）
--   bus_status      每台車「目前」的狀態，一台車只有一列，每次推論用 upsert 覆蓋
--   bus_status_log  歷史紀錄，只在佔用數改變或隔一段時間才寫一筆，避免資料庫無限長大
--   v_bus_status    給 API 讀的檢視表，把「剩餘座位數」算出來（不另外存，避免跟已佔用數對不起來）

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS routes (
    route_id   TEXT PRIMARY KEY,                       -- 'E01' / 'E02' / 'E25'
    name       TEXT NOT NULL DEFAULT '',               -- 路線名稱，之後有資料再補
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS buses (
    bus_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    plate        TEXT NOT NULL UNIQUE,                 -- 車牌，例如 KKA-1234
    route_id     TEXT NOT NULL REFERENCES routes(route_id) ON UPDATE CASCADE,
    total_seats  INTEGER NOT NULL CHECK (total_seats > 0),
                                                       -- 有納入監測的座位數 = 該車 seats_config 的座位數
    seats_config TEXT,                                 -- 該車的座位座標檔路徑（每台車的座位排列可能不同）
    source       TEXT NOT NULL DEFAULT 'camera'
                 CHECK (source IN ('camera', 'video', 'simulated')),
                                                       -- 資料來源；simulated = 示範用假資料，不是真實推論
    is_active    INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_buses_route ON buses(route_id);

CREATE TABLE IF NOT EXISTS bus_status (
    bus_id         INTEGER PRIMARY KEY REFERENCES buses(bus_id) ON DELETE CASCADE,
    occupied_count INTEGER NOT NULL CHECK (occupied_count >= 0),
    person_count   INTEGER NOT NULL DEFAULT 0 CHECK (person_count >= 0),
    occupied_seats TEXT NOT NULL,                      -- JSON：{"A01": "empty", "A02": "occupied", ...}
    updated_at     TEXT NOT NULL                       -- 最後更新時間（UTC）
);

-- 已佔用數不能超過該車的座位數（跨表的條件 CHECK 做不到，用 trigger 在資料庫這一層擋住）
CREATE TRIGGER IF NOT EXISTS trg_status_capacity_insert
BEFORE INSERT ON bus_status
WHEN NEW.occupied_count > (SELECT total_seats FROM buses WHERE bus_id = NEW.bus_id)
BEGIN
    SELECT RAISE(ABORT, 'occupied_count exceeds total_seats');
END;

CREATE TRIGGER IF NOT EXISTS trg_status_capacity_update
BEFORE UPDATE ON bus_status
WHEN NEW.occupied_count > (SELECT total_seats FROM buses WHERE bus_id = NEW.bus_id)
BEGIN
    SELECT RAISE(ABORT, 'occupied_count exceeds total_seats');
END;

CREATE TABLE IF NOT EXISTS bus_status_log (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bus_id         INTEGER NOT NULL REFERENCES buses(bus_id) ON DELETE CASCADE,
    occupied_count INTEGER NOT NULL,
    recorded_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_log_bus_time ON bus_status_log(bus_id, recorded_at);

CREATE VIEW IF NOT EXISTS v_bus_status AS
SELECT
    b.bus_id,
    b.plate,
    b.route_id,
    r.name                          AS route_name,
    b.source,
    b.is_active,
    b.total_seats,
    s.occupied_count,
    b.total_seats - s.occupied_count AS remaining_seats,   -- 剩餘座位數：算出來的，不存
    s.person_count,
    s.occupied_seats,
    s.updated_at
FROM buses b
JOIN routes r ON r.route_id = b.route_id
LEFT JOIN bus_status s ON s.bus_id = b.bus_id;           -- 還沒收到過資料的車也會列出來（狀態欄位是 NULL）
