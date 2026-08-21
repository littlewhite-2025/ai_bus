// Nav items shown both inline in the desktop top bar and inside the
// hamburger dropdown (mobile relies on the dropdown only — see TopBar.jsx).
export const NAV_LINKS = [
  { key: "about", label: "ABOUT" },
  { key: "news", label: "NEWS" },
];

// TODO: 換成你的 repo 連結
export const GITHUB_URL = "https://github.com/your-org/your-repo";

// 頁尾佔位資訊 — 之後直接改這幾個值即可，元件不用動
export const FOOTER = {
  copyright: `© ${new Date().getFullYear()} AI 智慧公車`,
  email: "contact@example.com", // TODO: 換成正式聯絡信箱
  note: "此頁尾為佔位內容，之後會補上正式的聯絡方式與版權說明。",
};
