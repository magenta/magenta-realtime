import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import {fileURLToPath} from 'url';
import {viteSingleFile} from 'vite-plugin-singlefile';
import {execSync} from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const abs = (relativePath: string) =>
  path.resolve(__dirname, '..', relativePath);

const commitHash = (() => {
  try {
    return execSync('git rev-parse --short HEAD').toString().trim();
  } catch (e) {
    return 'unknown';
  }
})();

export default defineConfig(() => ({
  define: {
    __COMMIT_HASH__: JSON.stringify(commitHash),
  },
  base: './',
  server: {
    port: 62421,
    fs: {
      allow: ['..'],
    },
  },
  logLevel: 'info',
  plugins: [react(), viteSingleFile()],
  resolve: {
    alias: {
      '@': abs('src'),
    },
  },
  css: {
    postcss: abs('config'),
  },
  publicDir: abs('public'),
  build: {
    outDir: abs('dist'),
    emptyOutDir: true,
  },
}));
