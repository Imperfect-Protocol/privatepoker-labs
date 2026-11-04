import { defineConfig } from 'vite';
import { standalone } from 'vite-plugin-standalone';

// Mirrors private-meet's pipeline:
//   vite build                                                    → client bundle in dist/
//   vite build --ssr server/standalone-entry.js --outDir dist/server
//                                                                 → standalone EdgeJS QuickJS server
// The client entry (index.html) loads the tsc web build (build-web/),
// which Vite resolves from the project root and bundles normally.
export default defineConfig({
  plugins: [
    standalone({
      entry: {
        index: './server/standalone-entry.js',
      },
    }),
  ],
});
