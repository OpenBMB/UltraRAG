import { useQuery } from "@tanstack/react-query";
import { fetchCurrentUser } from "@/shared/api/auth";

export const AUTH_ME_QUERY_KEY = ["auth", "me"] as const;

export function useAuthMe() {
  return useQuery({
    queryKey: AUTH_ME_QUERY_KEY,
    queryFn: fetchCurrentUser,
    staleTime: 60_000,
  });
}
