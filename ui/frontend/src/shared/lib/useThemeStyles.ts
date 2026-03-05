import { useEffect } from "react";

const THEME_STYLE_HREFS = [
  "/theme/vendor/css/fonts.css",
  "/theme/vendor/css/bootstrap.min.css",
  "/theme/vendor/css/katex.min.css",
  "/theme/vendor/css/highlight.js/github.min.css",
  "/theme/style.css",
];

function styleMarker(href: string): string {
  return `theme-style:${href}`;
}

export function useThemeStyles(enabled = true) {
  useEffect(() => {
    if (!enabled) return;

    const created: HTMLLinkElement[] = [];
    for (const href of THEME_STYLE_HREFS) {
      const marker = styleMarker(href);
      const existing = document.head.querySelector<HTMLLinkElement>(
        `link[data-theme-style="${marker}"]`,
      );
      if (existing) continue;
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      link.dataset.themeStyle = marker;
      document.head.appendChild(link);
      created.push(link);
    }

    return () => {
      for (const link of created) {
        link.remove();
      }
    };
  }, [enabled]);
}
