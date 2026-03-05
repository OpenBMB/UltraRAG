import { useQuery } from "@tanstack/react-query";
import { fetchPipelines } from "@/shared/api/pipelines";

export const PIPELINES_QUERY_KEY = ["pipelines", "list"] as const;

export function usePipelines() {
  return useQuery({
    queryKey: PIPELINES_QUERY_KEY,
    queryFn: fetchPipelines,
    staleTime: 15_000,
  });
}
