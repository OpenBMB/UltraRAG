import { useQuery } from "@tanstack/react-query";
import { fetchChatSessions } from "@/shared/api/chat";
import type { ChatSession } from "@/shared/api/types";

export const CHAT_SESSIONS_QUERY_KEY = ["chat", "sessions"] as const;

export function useChatSessions(enabled = true) {
  return useQuery<ChatSession[]>({
    queryKey: CHAT_SESSIONS_QUERY_KEY,
    queryFn: () => fetchChatSessions(300),
    staleTime: 10_000,
    enabled,
  });
}
