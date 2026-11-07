import { Component, type ReactNode } from "react";

/** Offer recovery if a render or an outdated lazy-loaded chunk fails. */
export class AppErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    const chinese = document.documentElement.lang.startsWith("zh");
    return (
      <main className="missing-page" role="alert">
        <h1>{chinese ? "页面暂时无法加载" : "The page could not be loaded"}</h1>
        <p>{chinese ? "请检查网络连接后重新加载。" : "Check your connection and reload to continue."}</p>
        <button className="ur-btn ur-btn--default ur-btn--md" onClick={() => window.location.reload()}>
          {chinese ? "重新加载" : "Reload"}
        </button>
      </main>
    );
  }
}
