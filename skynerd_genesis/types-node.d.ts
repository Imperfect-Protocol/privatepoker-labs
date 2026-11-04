/**
 * Minimal ambient declarations for the Node APIs the test runner uses.
 * Zero-dependency choice: when the webapp gains a package.json install step
 * (npm i -D @types/node), delete this file.
 */
declare module "node:fs" {
  export function readFileSync(path: string): Buffer;
  export function readFileSync(path: string, encoding: string): string;
}
declare module "node:url" {
  export function fileURLToPath(url: string): string;
}
declare module "node:path" {
  export function dirname(path: string): string;
  export function join(...segments: string[]): string;
}
declare module "node:crypto" {
  export function createHash(algorithm: string): {
    update(data: string): { digest(): Buffer };
  };
}
declare interface Buffer extends Uint8Array {
  equals(other: Uint8Array): boolean;
}
declare var Buffer: {
  from(data: Uint8Array): Buffer;
};
declare var process: {
  exit(code?: number): never;
};
