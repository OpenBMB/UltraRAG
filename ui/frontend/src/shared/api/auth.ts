import { httpRequest } from "@/shared/api/http";
import type { UserInfo } from "@/shared/api/types";

type LoginPayload = {
  username: string;
  password: string;
};

type RegisterPayload = {
  username: string;
  password: string;
};

type ChangePasswordPayload = {
  current_password: string;
  new_password: string;
};

type ModelRoleSettings = {
  api_key?: string;
  base_url?: string;
  model_name?: string;
};

type ModelSettingsPayload = {
  retriever?: ModelRoleSettings;
  generation?: ModelRoleSettings;
};

export async function fetchCurrentUser(): Promise<UserInfo> {
  return httpRequest<UserInfo>("/api/auth/me");
}

export async function login(payload: LoginPayload): Promise<UserInfo> {
  return httpRequest<UserInfo>("/api/auth/login", {
    method: "POST",
    body: payload,
  });
}

export async function register(payload: RegisterPayload): Promise<UserInfo> {
  return httpRequest<UserInfo>("/api/auth/register", {
    method: "POST",
    body: payload,
  });
}

export async function logout(): Promise<{ status: string }> {
  return httpRequest<{ status: string }>("/api/auth/logout", {
    method: "POST",
  });
}

export async function updateNickname(nickname: string): Promise<{ status: string; nickname: string }> {
  return httpRequest<{ status: string; nickname: string }>("/api/auth/nickname", {
    method: "POST",
    body: { nickname },
  });
}

export async function changePassword(payload: ChangePasswordPayload): Promise<{ status: string }> {
  return httpRequest<{ status: string }>("/api/auth/change-password", {
    method: "POST",
    body: payload,
  });
}

export async function updateModelSettings(
  payload: ModelSettingsPayload,
): Promise<{ status: string; model_settings?: Record<string, unknown> }> {
  return httpRequest<{ status: string; model_settings?: Record<string, unknown> }>(
    "/api/auth/model-settings",
    {
      method: "POST",
      body: payload,
    },
  );
}
