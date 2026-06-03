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

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import { ThemeProvider } from '@mui/material/styles';
import { magentaTheme } from '@magenta-rt/common';
import './index.css';

window.onerror = function(msg, url, lineNo, columnNo, error) {
  const el = document.createElement('div');
  el.style.cssText = 'color:red; background:black; position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999; font-family:monospace; padding:20px; word-wrap:break-word;';
  el.innerText = "JS CRASH: " + msg + "\n" + url + ":" + lineNo + ":" + columnNo + "\n\n" + (error && error.stack ? error.stack : "");
  document.body.appendChild(el);
  return false;
};

import App from './App.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={magentaTheme}>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
