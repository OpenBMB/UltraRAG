/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useUiStore } from "@/features/ui/useUiStore";

type LocaleCode = "zh" | "en";
type LocaleDict = Record<string, string>;

declare global {
  interface Window {
    I18N_LOCALES?: Record<string, LocaleDict>;
  }
}

type I18nContextValue = {
  locale: LocaleCode;
  setLocale: (locale: LocaleCode) => void;
  t: (key: string, fallback?: string) => string;
  ready: boolean;
};

const FALLBACK_LOCALES: Record<LocaleCode, LocaleDict> = {
  zh: {
    settings: "设置",
    builder: "构建器",
    language: "语言",
    auth_account: "账号",
    select_pipeline: "选择 Pipeline",
  },
  en: {
    settings: "Settings",
    builder: "Builder",
    language: "Language",
    auth_account: "Account",
    select_pipeline: "Select Pipeline",
  },
};

const I18nContext = createContext<I18nContextValue | null>(null);

const scriptLoadCache = new Map<string, Promise<void>>();

function loadScript(src: string): Promise<void> {
  const cached = scriptLoadCache.get(src);
  if (cached) return cached;

  const promise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[data-i18n-src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.i18nSrc = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load i18n script: ${src}`));
    document.head.appendChild(script);
  });
  scriptLoadCache.set(src, promise);
  return promise;
}

function getLocaleFromStorage(): LocaleCode {
  const value = localStorage.getItem("ultrarag_ui_lang");
  return value === "en" ? "en" : "zh";
}

export function UltraI18nProvider({ children }: { children: ReactNode }) {
  const locale = useUiStore((s) => s.locale);
  const setLocaleInStore = useUiStore((s) => s.setLocale);
  const [ready, setReady] = useState(false);
  const [dynamicLocales, setDynamicLocales] = useState<Record<string, LocaleDict>>({});

  useEffect(() => {
    setLocaleInStore(getLocaleFromStorage());
  }, [setLocaleInStore]);

  useEffect(() => {
    let disposed = false;
    const load = async () => {
      try {
        await Promise.all([
          loadScript("/theme/i18n/en.js"),
          loadScript("/theme/i18n/zh.js"),
        ]);
      } catch {
        // Keep fallback dictionary available even if theme assets are absent.
      }
      if (disposed) return;
      setDynamicLocales(window.I18N_LOCALES ?? {});
      setReady(true);
    };
    void load();
    return () => {
      disposed = true;
    };
  }, []);

  const setLocale = useCallback(
    (nextLocale: LocaleCode) => {
      setLocaleInStore(nextLocale);
      localStorage.setItem("ultrarag_ui_lang", nextLocale);
    },
    [setLocaleInStore],
  );

  const t = useCallback(
    (key: string, fallback?: string) => {
      const mergedLocale = {
        ...FALLBACK_LOCALES[locale],
        ...(dynamicLocales[locale] ?? {}),
      };
      return mergedLocale[key] ?? fallback ?? key;
    },
    [dynamicLocales, locale],
  );

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t,
      ready,
    }),
    [locale, ready, setLocale, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within UltraI18nProvider");
  }
  return context;
}
