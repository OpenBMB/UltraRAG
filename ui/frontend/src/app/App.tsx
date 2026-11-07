import { BrowserRouter } from "react-router-dom";
import { QueryProvider } from "@/app/providers/QueryProvider";
import { I18nProvider } from "@/app/providers/I18nProvider";
import { AppRouter } from "@/app/router";
import { AppErrorBoundary } from "@/app/AppErrorBoundary";

export function App() {
  return (
    <QueryProvider>
      <I18nProvider>
        <BrowserRouter>
          <AppErrorBoundary><AppRouter /></AppErrorBoundary>
        </BrowserRouter>
      </I18nProvider>
    </QueryProvider>
  );
}
