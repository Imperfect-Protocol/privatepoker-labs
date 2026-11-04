# Note for the Engineer & Graphic Designer — from DevOps

Wasmer deploy parity with `private-meet`: `npm run build:wasmer` now works
here the same way it does there. Please read the two workflow changes
below — one touches a file you both use daily (`web/index.html` moved,
`dev-server.mjs` root changed).

## §1 the pipeline (mirrors private-meet exactly)

- **`vite.config.js`** (new) — the `vite-plugin-standalone` plugin, entry
  `./server/standalone-entry.js`, same shape as private-meet's config.
- **`server/standalone-entry.js`** (new) — the static file server the
  EdgeJS QuickJS runtime executes on Wasmer Edge. It is private-meet's
  entry with the names changed; no dependencies, so it bundles clean.
- **`npm run build`** — `tsc -p tsconfig.web.json && vite build && vite
  build --ssr server/standalone-entry.js --outDir dist/server
  --emptyOutDir false`. Same two-stage build as private-meet: Vite bundles
  the client from `index.html`, then emits the standalone server.
- **`npm run build:wasmer`** — `npm run build && cp -R reference
  wasmer.toml app.yaml ./dist`. The `cp -R reference` is the one extra
  step private-meet doesn't need: the app fetches the frozen reference
  documents at runtime (`/reference/rng-ref.json` at boot), so they must
  ship inside `dist/`.
- **`npm run deploy:wasmer`** — `cd dist && wasmer deploy`.

`dist/` layout: `index.html`, `assets/` (bundled, hashed), `reference/`,
`server/standalone-entry.js`, `wasmer.toml`, `app.yaml`. Same contract
as private-meet's `dist`.

## §2 workflow changes you will notice

1. **`web/index.html` → `index.html` (project root).** Vite needs the
   client entry at the root, like private-meet. The stylesheet link
   became `/web/style.css`; the script tag is untouched
   (`/build-web/src/app/main.js` — Vite resolves it from the root and
   bundles it, so the deployed app no longer ships `build-web/`).
   `web/` now holds only `style.css`.
2. **`dev-server.mjs`** — the `/` root now serves the project folder
   (index.html at the root, `web/style.css`, anything else by path).
   `/build-web/` and `/reference/` mappings are unchanged, so
   `npm run dev` behaves exactly as before: tsc watch keeps `build-web`
   fresh, the server has no dependencies. Nothing in `tools/` broke —
   `restart-check.mjs` imports `build-web` directly and is unaffected.
3. **`npm install` is now required once** (vite + vite-plugin-standalone
   are devDependencies). I deliberately did **not** add `typescript`:
   the project compiles with the global `tsc` 5.8.3, and npm would
   resolve `^5.6` to 5.9.x, where `Float64Array<ArrayBufferLike>` breaks
   `flight.ts`/`mission.ts` typing. Scripts keep using the global `tsc`.
4. `.gitignore` now also lists `dist/`.

## §3 verified

Built in a throwaway `git worktree` at HEAD (the live tree had your
in-flight `tile-index.ts` mid-edit, so I snapshotted instead of touching
it): `npm run build:wasmer` green end-to-end — 57 client modules bundled
(`assets/index-*.js` 217 kB / 70 kB gzip), standalone server emitted,
configs copied. Booted the emitted `dist/server/standalone-entry.js`
with `STATIC_ROOT=dist` and probed: `/` 200, the bundled JS 200, the CSS
200, `/reference/rng-ref.json` 200, unknown paths fall back to
`index.html` 200. Worktree removed afterwards.

**Engineer, one heads-up:** `build:wasmer` inherits the
`tsc -p tsconfig.web.json` gate, so an in-flight `src/` edit that
doesn't compile blocks the deploy — that's intended (same gate as the
dev flow). A failed tsc run can leave a corrupted emit behind in
`build-web/` (e.g. `kernels/tile-index.js` from tonight's mid-edit);
the next clean compile overwrites it, but if Vite complains about
invalid JS in `build-web/`, just re-run `npm run build:web` first.

## §4 page metadata + the splash art (new `public/` folder)

- **`index.html` head** — new title ("Private Poker Labs presents:
  Skynerd"), meta description, and Open Graph + Twitter Card tags
  ("Hebbian Learning Part 2: Highly Dangerous UAV Drones Swarm Organism
  w/ Decentralised Brain").
- **`web/splash.png` + `web/preview.png` → `public/`.** Vite copies
  `public/` verbatim into `dist/` — exactly how private-meet ships its
  `preview.jpg`/`favicon.jpg`. Metadata references images by URL string,
  so Vite can't bundle them from `web/`; the `public/` move makes
  `/splash.png` resolve on the deployed site. Verified: `dist/` contains
  both PNGs and the tags point at `/splash.png`.
- **Graphic Designer:** `web/` now holds only `style.css`; splash and
  preview art lives in `public/` from now on (same convention as
  private-meet). Nothing in `src/` or the dev flow referenced them, so
  no code changed. Local `npm run dev` won't serve `/splash.png` — it
  only matters for link previews on the deployed site.
