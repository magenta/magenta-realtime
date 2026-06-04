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

/** @type {import('tailwindcss').Config} */
import plugin from 'tailwindcss/plugin';

export default {
  darkMode: ['class'],
  content: ['index.html', 'src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      width: {
        plugin: '1600px',
        controlbar: '1100px',
      },
      height: {
        plugin: '800px',
        content: '660px',
        paddedcontent: '620px',
        subcontent: '540px',
      },
      fontFamily: {
        sans: [
          'Google Sans Text',
          'system-ui',
          'sans-serif',
        ],
        display: [
          'Google Sans',
          'system-ui',
          'sans-serif',
        ],
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        progress: 'hsl(var(--progress))',
        updater: 'hsl(var(--updater))',
        chart: {
          1: 'hsl(var(--chart-1))',
          2: 'hsl(var(--chart-2))',
          3: 'hsl(var(--chart-3))',
          4: 'hsl(var(--chart-4))',
          5: 'hsl(var(--chart-5))',
        },
        sidebar: {
          DEFAULT: 'hsl(var(--sidebar-background))',
          foreground: 'hsl(var(--sidebar-foreground))',
          primary: 'hsl(var(--sidebar-primary))',
          'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
          accent: 'hsl(var(--sidebar-accent))',
          'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
          border: 'hsl(var(--sidebar-border))',
          ring: 'hsl(var(--sidebar-ring))',
        },
        prompt: {
          0: 'hsl(var(--prompt-0))',
          1: 'hsl(var(--prompt-1))',
          2: 'hsl(var(--prompt-2))',
          3: 'hsl(var(--prompt-3))',
          4: 'hsl(var(--prompt-4))',
          5: 'hsl(var(--prompt-5))',
          6: 'hsl(var(--prompt-6))',
          7: 'hsl(var(--prompt-7))',
          8: 'hsl(var(--prompt-8))',
          9: 'hsl(var(--prompt-9))',
        },
        vinyl: {
          0: 'hsl(var(--vinyl-0))',
          1: 'hsl(var(--vinyl-1))',
          2: 'hsl(var(--vinyl-2))',
          3: 'hsl(var(--vinyl-3))',
          4: 'hsl(var(--vinyl-4))',
          5: 'hsl(var(--vinyl-5))',
          6: 'hsl(var(--vinyl-6))',
          7: 'hsl(var(--vinyl-7))',
          8: 'hsl(var(--vinyl-8))',
          9: 'hsl(var(--vinyl-9))',
          10: 'hsl(var(--vinyl-10))',
          11: 'hsl(var(--vinyl-11))',
          12: 'hsl(var(--vinyl-12))',
          13: 'hsl(var(--vinyl-13))',
        },
      },
    },
  },
  plugins: [
    require('tailwindcss-animate'),

    plugin(function ({addUtilities}) {
      addUtilities({
        '.arc': {
          '--b': '5px',
          '--a': '270deg',
          width: '80px',
          aspectRatio: '1',
          padding: 'var(--b)',
          borderRadius: '9999px',
          transform: 'rotate(-135deg)',
          '--_g':
            '/var(--b) var(--b) no-repeat radial-gradient(50% 50%, #000 97%, #0000)',
          mask: `
            top var(--_g),
            calc(50% + 50%*sin(var(--a))) calc(50% - 50%*cos(var(--a))) var(--_g),
            linear-gradient(#0000 0 0) content-box intersect,
            conic-gradient(#000 var(--a), #0000 0)
          `,
        },
        '.arc-progress': {
          '--b': '5px',
          '--a': 'var(--a)',
          width: '80px',
          aspectRatio: '1',
          padding: 'var(--b)',
          borderRadius: '9999px',
          transform: 'rotate(-135deg)',
          '--_g':
            '/var(--b) var(--b) no-repeat radial-gradient(50% 50%, #000 97%, #0000)',
          mask: `
            top var(--_g),
            calc(50% + 50%*sin(var(--a))) calc(50% - 50%*cos(var(--a))) var(--_g),
            linear-gradient(#0000 0 0) content-box intersect,
            conic-gradient(#000 var(--a), #0000 0)
          `,
        },
      });
    }),
  ],
};
