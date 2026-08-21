import { FOOTER, GITHUB_URL } from "../constants/site";
import { GithubIcon } from "../utils/icons.jsx";

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__row">
        <span className="site-footer__copyright">{FOOTER.copyright}</span>
        <span className="site-footer__links">
          <a className="site-footer__link" href={`mailto:${FOOTER.email}`}>
            {FOOTER.email}
          </a>
          <a
            className="site-footer__link"
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            <GithubIcon width={13} height={13} />
            GitHub
          </a>
        </span>
      </div>
      <p className="site-footer__note">{FOOTER.note}</p>
    </footer>
  );
}
