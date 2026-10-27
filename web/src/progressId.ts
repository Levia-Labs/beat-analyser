// getRandomValues exists in insecure contexts (plain http on a LAN IP); randomUUID does not.
export function progressId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
