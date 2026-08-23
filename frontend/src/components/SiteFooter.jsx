import { FOOTER } from "../constants/site";

export default function SiteFooter() {
  const year = new Date().getFullYear();

  return (
    <footer className="site-footer">
      <div className="site-footer__row">
        <a className="site-footer__contact" href={`mailto:${FOOTER.email}`}>
          CONTACT: {FOOTER.email}
        </a>

        <p className="site-footer__copyright">
          Copyright © {year} {FOOTER.projectName}, All rights reserved.
          <br />
          Mainly developed by {FOOTER.developer}.
        </p>

        <span className="site-footer__icon">
          <img src="/icons/brand-icon.png" alt={FOOTER.projectName} />
        </span>
      </div>
    </footer>
  );
}
