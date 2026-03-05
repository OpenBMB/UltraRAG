import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Button } from "@/shared/ui/button";

type AuthDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onLogin: (payload: { username: string; password: string }) => Promise<void>;
  onRegister: (payload: { username: string; password: string }) => Promise<void>;
  isSubmitting: boolean;
  errorMessage?: string;
};

type AuthMode = "login" | "register";

export function AuthDialog({
  open,
  onOpenChange,
  onLogin,
  onRegister,
  isSubmitting,
  errorMessage,
}: AuthDialogProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const submitLabel = useMemo(() => {
    if (isSubmitting) return mode === "login" ? "登录中..." : "注册中...";
    return mode === "login" ? "登录" : "注册";
  }, [isSubmitting, mode]);

  const title = mode === "login" ? "账号登录" : "注册账号";
  const description =
    mode === "login"
      ? "登录后可保存会话历史与个人设置。"
      : "注册后将自动登录，并启用会话持久化。";

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!username.trim() || !password.trim()) return;
    if (mode === "login") {
      await onLogin({ username: username.trim(), password });
    } else {
      await onRegister({ username: username.trim(), password });
    }
    setPassword("");
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="auth-dialog">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">
            <span>用户名</span>
            <input
              className="auth-input"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="例如 alice_01"
              disabled={isSubmitting}
            />
          </label>
          <label className="auth-label">
            <span>密码</span>
            <input
              className="auth-input"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="至少 6 位"
              disabled={isSubmitting}
            />
          </label>

          {errorMessage ? <p className="auth-error">{errorMessage}</p> : null}

          <div className="auth-actions">
            <Button type="submit" disabled={isSubmitting || !username.trim() || !password.trim()}>
              {submitLabel}
            </Button>
            <Button
              type="button"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => setMode((prev) => (prev === "login" ? "register" : "login"))}
            >
              {mode === "login" ? "没有账号？去注册" : "已有账号？去登录"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
