export type ApiErrorPayload = {
  error?: string;
  details?: string;
  [key: string]: unknown;
};

export type UserInfo = {
  logged_in: boolean;
  user_id: string;
  username: string | null;
  nickname: string | null;
  model_settings: Record<string, unknown> | null;
  is_admin: boolean;
};

export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  meta?: Record<string, unknown>;
  timestamp?: string;
};

export type SourceDoc = {
  id: number;
  displayId?: number;
  title: string;
  content: string;
};

export type ChatSession = {
  id: string;
  title: string;
  pipeline?: string | null;
  timestamp?: number;
  messages?: ChatMessage[];
  updated_at?: string;
  created_at?: string;
  [key: string]: unknown;
};

export type PipelineItem = {
  name: string;
  is_ready?: boolean;
  config?: Record<string, unknown>;
  [key: string]: unknown;
};

export type PipelineChatFinalData = {
  status?: string;
  answer?: string;
  dataset_path?: string | null;
  memory_path?: string | null;
  is_first_turn?: boolean;
  is_multiturn?: boolean;
  [key: string]: unknown;
};

export type PipelineChatEvent =
  | {
      type: "token";
      content?: string;
      is_final?: boolean;
      [key: string]: unknown;
    }
  | {
      type: "final";
      data?: PipelineChatFinalData;
      [key: string]: unknown;
    }
  | {
      type: "error";
      message?: string;
      [key: string]: unknown;
    }
  | {
      type: "sources";
      data?: SourceDoc[];
      [key: string]: unknown;
    }
  | {
      type: "step_start" | "step_end";
      [key: string]: unknown;
    };

export type PipelineChatHistoryResponse = {
  session_id: string;
  history: Array<{ role: string; content: string }>;
  is_first_turn?: boolean;
  message_count?: number;
  [key: string]: unknown;
};
