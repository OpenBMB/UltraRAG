import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "@/app/App";
import "@xyflow/react/dist/style.css";
import "highlight.js/styles/github.css";
import "katex/dist/katex.min.css";
import "@/app/styles/tokens.css";
import "@/app/styles/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
