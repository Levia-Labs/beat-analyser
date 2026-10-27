/// <reference types="vite/client" />

interface ImportMetaEnv {
  // The commit the image was built from (Dockerfile ARG); unset outside a docker build.
  readonly VITE_COMMIT?: string;
}
