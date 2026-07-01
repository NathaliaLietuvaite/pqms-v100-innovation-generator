// Shared CORS helper with origin allowlist (defense-in-depth).
// Only reflects Origin when it matches the allowlist; otherwise falls back
// to the primary production origin so browsers block cross-origin misuse.

const STATIC_ALLOWED_ORIGINS = [
  "https://pqms-v100-innovation-generator.lovable.app",
  "http://localhost:8080",
  "http://localhost:5173",
];

// Allow the Lovable preview/sandbox subdomains and any custom FRONTEND_URL.
const ALLOWED_ORIGIN_PATTERNS: RegExp[] = [
  /^https:\/\/([a-z0-9-]+\.)*lovable\.app$/i,
  /^https:\/\/([a-z0-9-]+\.)*lovableproject\.com$/i,
  /^https:\/\/([a-z0-9-]+\.)*lovable\.dev$/i,
];

function isOriginAllowed(origin: string | null): origin is string {
  if (!origin) return false;
  const extra = Deno.env.get("FRONTEND_URL");
  if (extra && origin === extra) return true;
  if (STATIC_ALLOWED_ORIGINS.includes(origin)) return true;
  return ALLOWED_ORIGIN_PATTERNS.some((re) => re.test(origin));
}

export function getCorsHeaders(req: Request): Record<string, string> {
  const origin = req.headers.get("Origin");
  const allowOrigin = isOriginAllowed(origin) ? origin : STATIC_ALLOWED_ORIGINS[0];
  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Headers":
      "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Vary": "Origin",
  };
}
