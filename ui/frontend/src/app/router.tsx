import { Navigate, Route, Routes } from "react-router-dom";
import { ChatPage } from "@/pages/ChatPage";
import { BuilderPage } from "@/pages/BuilderPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  const search = window.location.search;

  return (
    <Routes>
      <Route path="/" element={<Navigate to={`/chat${search}`} replace />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/settings" element={<BuilderPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
