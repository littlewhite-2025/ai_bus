import { ArrowUpIcon } from "../utils/icons.jsx";
import { useScrollTop } from "../hooks/useScrollTop.js";

export default function ScrollTopButton() {
  const { isVisible, scrollToTop } = useScrollTop();

  return (
    <button
      type="button"
      className={`scroll-top icon-btn icon-btn--raised${isVisible ? " is-visible" : ""}`}
      onClick={scrollToTop}
      aria-label="回到頁面頂端"
      tabIndex={isVisible ? 0 : -1}
    >
      <ArrowUpIcon width={18} height={18} />
    </button>
  );
}
