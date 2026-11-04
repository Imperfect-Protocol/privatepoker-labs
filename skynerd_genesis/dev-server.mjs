/**
 * Zero-dependency dev server for the Skynerd Genesis web app.
 * Serves the app (web/), the compiled simulator (build-web/), and the frozen
 * reference documents (reference/); spawns `tsc --watch` so edits to src/
 * rebuild while it runs.
 *
 * Run with: npm run dev   (or: node dev-server.mjs)
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8000);
const roots = {
  "/": here,
  "/build-web/": path.join(here, "build-web"),
  "/reference/": path.join(here, "reference"),
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".map": "application/json",
};

function serve(urlPath, res) {
  let resolved = null;
  const entries = Object.entries(roots);
  for (const [prefix, root] of [...entries.slice(1), entries[0]]) {
    const matches = prefix === "/"
      ? true // "/" is the fallback root: any path not caught by the other prefixes
      : urlPath.startsWith(prefix);
    if (matches) {
      const rel = prefix === "/"
        ? (urlPath === "/" ? "index.html" : urlPath.slice(1))
        : urlPath.slice(prefix.length);
      const candidate = path.normalize(path.join(root, rel));
      if (candidate.startsWith(root) && fs.existsSync(candidate)
          && fs.statSync(candidate).isFile()) {
        resolved = candidate;
      }
      break;
    }
  }
  if (!resolved) {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("404 not found");
    return;
  }
  const ext = path.extname(resolved).toLowerCase();
  res.writeHead(200, {
    "content-type": MIME[ext] ?? "application/octet-stream",
    "cache-control": "no-store",
  });
  fs.createReadStream(resolved).pipe(res);
}

// tsc --watch keeps build-web fresh during development.
const watcher = spawn("tsc", ["-p", "tsconfig.web.json", "--watch", "--preserveWatchOutput"], {
  stdio: "inherit",
});
watcher.on("error", () => {
  console.error("(tsc --watch failed to start; run `npx tsc -p tsconfig.web.json` manually)");
});

const server = http.createServer((req, res) => {
  serve(decodeURIComponent(req.url.split("?")[0]), res);
});
server.listen(PORT, () => {
  console.log(`\n  Skynerd Genesis dev server`);
  console.log(`  → http://localhost:${PORT}\n`);
});

process.on("SIGINT", () => {
  watcher.kill();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 300).unref();
});
