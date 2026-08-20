# AI 智慧公車 · 前端 (React + Vite)

第一版（純 HTML/CSS/JS）的重構版本，架構相同、視覺調整為更平面（拿掉裝飾性光暈與過多對比色），並改用 React + Vite 以利之後做視覺效果、效能與資安強化。

## 執行

這個沙盒環境無法連外網安裝套件，請在本機執行：

```bash
npm install
npm run dev      # 開發伺服器，預設 http://localhost:5173
npm run build    # 產出正式版到 dist/
```

## 目前狀態：初始化 / 預設值

`src/constants/defaults.js` 定義了：

- `ROUTE_DIRECTORY`：目前先寫死 4 條路線（紅28 / 橘1 / 藍25 / 綠12）當作車輛清單的預設資料，之後若有車隊管理 API 可以取代這個常數。
- `createDefaultSeatSnapshot()`：座位初始化為全部 `empty`、`occupied_count: 0`、`person_count: 0`，格式對齊 `run_inference.py` 輸出的 JSON。
- `connectionStatus` 初始為 `"idle"`（尚未連接後端），不是隨機模擬資料——這點跟第一版的 mock fleet 不同，現在沒接 API 就會誠實顯示「尚未連接後端」而不是假造即時數據。

## 之後串接後端

只需要改 `src/api/busApi.js`（以及設定 `.env` 裡的 `VITE_API_BASE_URL`）：

```
GET /api/fleet          -> [{ id, route, routeColor, plate, seats, updatedAt, connectionStatus }, ...]
GET /api/buses/:id      -> { id, route, routeColor, plate, seats, updatedAt, connectionStatus }

seats = {
  occupied_seats: { "A01": "empty", ... },
  occupied_count: number,
  total_seats: number,
  person_count: number
}
```

`connectionStatus` 建議由後端回傳 `"connected"` / `"error"`，前端會自動切換文字與顏色（`src/utils/format.js` 的 `CONNECTION_STATUS`）。其他元件（`BusCard`、`StatRow`、`SeatPanel`、`Sidebar`…）完全不用改，因為它們只讀這個統一格式。

`src/api/client.js` 是共用的 `fetch` 包裝：GET-only、`credentials: "omit"`、8 秒逾時中止、沒設定 `VITE_API_BASE_URL` 時會丟出明確的 `ApiNotConfiguredError` 而不是靜默失敗。

## 這次調整的地方

- **重構為 React + Vite**：拆成 `components/`、`hooks/`、`api/`、`constants/`、`utils/`，畫面用 JSX 表達（避免舊版 `innerHTML` 字串拼接，順帶降低 XSS 風險）。
- **車輛 / 座位初始化**：不再用假資料模擬多台車即時跳動，預設狀態就是「已知路線 + 全空座位 + 尚未連線」。
- **拿掉「車頭 Front」標籤**：座位圖不再顯示這行文字。
- **收斂光暈與對比色**：拿掉背景放射狀光暈、座位圖的掃描光暈動畫、卡片的漸層光線、脈動陰影；顏色只保留在真正需要辨識狀態的地方（路線色、擁擠程度、連線狀態燈號）。
