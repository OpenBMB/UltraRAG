import { BrowserRouter } from "react-router-dom";
import { QueryProvider } from "@/app/providers/QueryProvider";
import { I18nProvider } from "@/app/providers/I18nProvider";
import { AppRouter } from "@/app/router";

export function App() {
  return (
    <QueryProvider>
      <I18nProvider>
        <BrowserRouter>
          <AppRouter />
        </BrowserRouter>
      </I18nProvider>
    </QueryProvider>
  );
}
