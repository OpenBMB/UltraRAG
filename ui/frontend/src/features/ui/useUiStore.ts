import { create } from "zustand";

type UiState = {
  locale: "zh" | "en";
  setLocale: (locale: "zh" | "en") => void;
};

export const useUiStore = create<UiState>((set) => ({
  locale: "zh",
  setLocale: (locale) => set({ locale }),
}));
