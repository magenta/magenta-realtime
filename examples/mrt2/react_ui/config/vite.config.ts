/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import {fileURLToPath} from 'url';
import svgr from 'vite-plugin-svgr';
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

export default defineConfig(({command}) => ({
  define: {
    __COMMIT_HASH__: JSON.stringify(commitHash),
  },
  base: './',
  server: {
    port: 62420,
    fs: {
      allow: ['..'],
    },
  },
  logLevel: 'info',
  plugins: [react(), svgr({svgrOptions: {jsxRuntime: 'automatic'}}), viteSingleFile()],
  resolve: {
    alias: {
      '@': abs('src'),
      ...(command !== 'build' && { 'fonts': abs('../resources/fonts/') }),
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
