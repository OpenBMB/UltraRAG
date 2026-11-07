import { useEffect } from "react";

const THEME_STYLE_HREFS = [
  "/theme/vendor/css/fonts.css",
  "/theme/vendor/css/bootstrap.min.css",
  "/theme/style.css",
];
const MARKDOWN_STYLE_HREFS = [
  "/theme/vendor/css/katex.min.css",
  "/theme/vendor/css/highlight.js/github.min.css",
];
const styles = new Map<string, { link: HTMLLinkElement; owners: number; created: boolean }>();

function useStyles(hrefs: string[], enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    for (const href of hrefs) {
      const current = styles.get(href);
      if (current) { current.owners += 1; continue; }
      const marker = `theme-style:${href}`;
      const existing = document.head.querySelector<HTMLLinkElement>(`link[data-theme-style="${marker}"]`);
      const link = existing ?? document.createElement("link");
      if (!existing) {
        link.rel = "stylesheet";
        link.href = href;
        link.dataset.themeStyle = marker;
        document.head.appendChild(link);
      }
      styles.set(href, { link, owners: 1, created: !existing });
    }
    return () => {
      for (const href of hrefs) {
        const current = styles.get(href);
        if (!current || --current.owners > 0) continue;
        if (current.created) current.link.remove();
        styles.delete(href);
      }
    };
  }, [hrefs, enabled]);
}

export function useThemeStyles(enabled = true) {
  useStyles(THEME_STYLE_HREFS, enabled);
}

export function useMarkdownStyles() {
  useStyles(MARKDOWN_STYLE_HREFS, true);
}
