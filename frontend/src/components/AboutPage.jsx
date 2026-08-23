export default function AboutPage() {
  return (
    <section className="page" aria-labelledby="about-heading">
      <h1 id="about-heading" className="page__title">
        About
      </h1>
      <hr className="page__divider" />

      <p className="page__body">
        國道快捷公車在尖峰時段常常一座難求。候車的人只能等到公車進站，才知道還有沒有位子，有時甚至得連續等好幾班。現有的公車資訊多半只提供到站時間與車輛位置，卻缺少乘客最在意的「車內實際擁擠程度」與「剩餘座位數」。
      </p>

      <p className="page__body">
        我們因此開發了這套「AI 國道快捷公車剩餘座位偵測與擁擠度資訊顯示系統」，希望讓乘客在等車前就能提前掌握車況。
      </p>

      <h2 className="page__subtitle">我們做了什麼</h2>

      <p className="page__body">
        我們打造了一套「AI 國道快捷公車剩餘座位偵測與擁擠度資訊顯示系統」。透過在車廂內架設攝影機，結合 YOLOv8 影像辨識技術，即時分析座位使用狀況，並計算出剩餘座位數與擁擠程度，最後把這些資訊同步更新到網頁平台上。
      </p>

      <p className="page__body">
        乘客只要打開網頁，就能在出門前或等車時，提前掌握下一班車的車內狀況，不用再盲目等待。
      </p>

      <h2 className="page__subtitle">系統如何運作</h2>

      <p className="page__body">系統核心流程簡單清楚：</p>

      <p className="page__body">
        公車內攝影機 → AI 影像辨識（YOLOv8）→ 人物與座位狀態分析 → 剩餘座位數計算 → 擁擠程度判斷 → 資訊平台更新 → 乘客即時查看
      </p>

      <p className="page__body">
        我們把複雜的 AI 運算，轉化成乘客一看就懂的「空閒／普通／擁擠」狀態，以及實際剩餘座位數。技術上採用輕量化的 YOLOv8 與 OpenCV，搭配 Python 後端與響應式網頁前端，讓系統能在有限的運算環境中穩定運作，同時確保手機與電腦都能順暢瀏覽。
      </p>

      <h2 className="page__subtitle">誰會用到這個系統</h2>

      <p className="page__body">
        <strong>乘客：</strong>
        提前知道車況，減少盲目等待，提升搭車意願。
      </p>

      <p className="page__body">
        <strong>客運業者：</strong>
        累積真實擁擠度數據，作為調整班距或車型的參考依據。
      </p>

      <p className="page__body">
        <strong>交通管理單位：</strong>
        提供一個可參考的智慧公車應用案例。
      </p>

      <h2 className="page__subtitle">技術實現</h2>

      <p className="page__body">
        系統核心是透過車內攝影機即時擷取畫面，使用 YOLOv8 進行人員偵識與座位狀態分析，再計算剩餘座位數與擁擠程度等，並將結果即時呈現在網頁平台上。
      </p>

      <p className="page__body">
        前端採用 React + Vite 架構開發，著重於響應式設計、視覺呈現與使用體驗的優化。後端則自行開發並完成前後端整合，同時處理資料傳輸、安全性考量，以及內網環境下的部署與穿透需求，確保系統在實際應用場景中能穩定運作。
      </p>

      <p className="page__body">
        我們刻意把複雜的 AI 運算與系統架構，轉換成乘客容易理解的資訊，讓技術真正服務於日常乘車體驗。
      </p>

      <h2 className="page__subtitle">網頁設計理念</h2>

      <p className="page__body">
        這個專案不只是功能的實現，也包含了對使用體驗、系統穩定性與資安風險的實際考量。
      </p>

      <p className="page__body">
        因此在開發過程中特別注意程式結構、介面美感與潛在的安全問題。
      </p>

      <p className="page__body">
        未來仍會持續優化模型在不同光線與遮擋情況下的表現，並探索與既有公車動態資訊系統的整合可能性。
      </p>

      <p className="page__body">
        但因為是使用影像辨識，結果可能不一定是百分之百正確，但我們將持續優化，給用戶帶來更好的使用體驗。
      </p>

      <h2 className="page__subtitle">我們的想法</h2>

      <p className="page__body">
        我們希望把 AI 真正落實在日常生活中，並且選擇相對低成本的硬體與開源技術，做出一個實用的原型系統，驗證「車內座位偵測」的可行性。未來也希望這套架構能延伸到市區公車或其他公路客運，為智慧交通多貢獻一點力量。
      </p>

      <p className="page__body">
        目前系統已完成從影像擷取、辨識計算到網頁呈現的完整流程。雖然在夜間低光源或乘客密集遮擋以及角度不同時，辨識率還有進步空間，不過我們會持續收集實地影像資料來優化模型，並嘗試與現有公車動態資訊系統介接，讓它更接近實際應用。
      </p>

      <hr className="page__divider" />
      <p className="page__body">
        本專案由團隊共同完成，涵蓋 AI 模型應用、前後端開發、系統整合與介面設計、簡報設計、文件撰寫等等。
      </p>
    </section>
  );
}