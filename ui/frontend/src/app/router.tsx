import { lazy, Suspense } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { NotFoundPage } from "@/pages/NotFoundPage";

const ChatPage = lazy(() => import("@/pages/ChatPage").then((page) => ({ default: page.ChatPage })));
const BuilderPage = lazy(() => import("@/pages/BuilderPage").then((page) => ({ default: page.BuilderPage })));

export function AppRouter() {
  const { search } = useLocation();

  return (
    <Routes>
      <Route path="/" element={<Navigate to={`/chat${search}`} replace />} />
      <Route path="/chat" element={
        <Suspense fallback={<main className="route-loading" role="status">Loading UltraRAG…</main>}>
          <ChatPage />
        </Suspense>
      } />
      <Route path="/settings" element={
        <Suspense fallback={<main className="route-loading" role="status">Loading UltraRAG…</main>}>
          <BuilderPage />
        </Suspense>
      } />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
