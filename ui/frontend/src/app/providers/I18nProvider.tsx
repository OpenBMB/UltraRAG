import type { ReactNode } from "react";
import { UltraI18nProvider } from "@/shared/i18n/provider";

type I18nProviderProps = {
  children: ReactNode;
};

export function I18nProvider({ children }: I18nProviderProps) {
  return <UltraI18nProvider>{children}</UltraI18nProvider>;
}
