import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom has no canvas and logs "Not implemented" on getContext; null is the real "no canvas" answer.
HTMLCanvasElement.prototype.getContext = () => null;

afterEach(cleanup);
