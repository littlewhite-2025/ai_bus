import { NAV_LINKS } from "../constants/site";

export default function NavMenu({ isOpen, currentPage, onNavigate }) {
  return (
    <nav id="nav-menu" className={`nav-menu${isOpen ? " is-open" : ""}`} aria-label="選單" aria-hidden={!isOpen}>
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
