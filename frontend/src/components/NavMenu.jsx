import { NAV_LINKS } from "../constants/site";

export default function NavMenu({ isOpen, currentPage, onNavigate }) {
  return (
    <nav id="nav-menu" className={`nav-menu${isOpen ? " is-open" : ""}`} aria-label="選單" aria-hidden={!isOpen}>
      {/* SVG filter def — zero visual footprint on its own, referenced by
          .nav-menu__goo via filter: url(#nav-goo). Rendered once here since
          NavMenu only ever mounts a single instance. */}
      <svg width="0" height="0" aria-hidden="true" focusable="false" style={{ position: "absolute" }}>
        <filter id="nav-goo">
          <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur" />
          <feColorMatrix
            in="blur"
            mode="matrix"
            values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -10"
            result="goo"
          />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
      </svg>

      <div className="nav-menu__goo">
        <span className="nav-menu__blob nav-menu__blob--seed" />
        <span className="nav-menu__blob nav-menu__blob--fill" />
      </div>

      <ul className="nav-menu__list">
        {NAV_LINKS.map((link) => (
          <li key={link.key}>
            <button
              type="button"
              className={`nav-menu__link${currentPage === link.key ? " is-active" : ""}`}
              onClick={() => onNavigate(link.key)}
            >
              {link.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
