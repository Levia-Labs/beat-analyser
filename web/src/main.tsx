import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import App from "./App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/page.css";
import "./styles/chips.css";
import "./styles/overlays.css";
import "./styles/library.css";
import "./styles/rows.css";
import "./styles/viewer.css";
import "./styles/about.css";

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </StrictMode>,
  );
}
