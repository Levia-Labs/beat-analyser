import js from "@eslint/js";
import react from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import globals from "globals";
import tseslint from "typescript-eslint";

// The rules of the shared crepidoma config, inlined: that file sits outside web/, so Node
// resolves its imports from a folder with no node_modules.
export default [
  { ignores: ["dist/**", "coverage/**", "node_modules/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: { globals: { ...globals.browser } },
  },
  { files: ["vite.config.ts", "eslint.config.js"], languageOptions: { globals: globals.node } },
  jsxA11y.flatConfigs.recommended,
  react.configs.flat["recommended-latest"],
];
