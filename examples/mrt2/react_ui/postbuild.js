import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const indexPath = path.join(__dirname, 'dist', 'index.html');
let html = fs.readFileSync(indexPath, 'utf8');

// Remove crossorigin attribute and type="module" which break file:// protocols
html = html.replace(/<script type="module" crossorigin>/g, '<script>');
html = html.replace(/<script type="module" crossorigin src=/g, '<script src=');
html = html.replace(/<script type="module">/g, '<script>');
html = html.replace(/ crossorigin/g, '');

// The singlefile plugin puts the <script> in the <head>. We must move it to the end of <body>
// since inline scripts can't use 'defer' and it needs document.body to exist.
const scriptMatch = html.match(/(<script>[\s\S]*?<\/script>)/);
if (scriptMatch) {
    html = html.replace(scriptMatch[0], '');
    html = html.replace('</body>', () => scriptMatch[0] + '\n  </body>');
}

fs.writeFileSync(indexPath, html);
console.log('Post-build: Removed CORS/module attributes from index.html');

const testHtml = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
</head>
<body style="background: green;">
    <h1 style="color: white; font-size: 32px;" id="output">Hello Audio Unit! Basic HTML works.</h1>
    <script>
        window.webkit.messageHandlers.auHost.postMessage({type: 'log', value: 'TEST_SCRIPT_LOG: Inline script executed.'});
        document.getElementById('output').innerText += " And JavaScript works too!";
    </script>
</body>
</html>`;
fs.writeFileSync(path.join(__dirname, 'dist', 'test.html'), testHtml);
console.log('Post-build: Wrote test.html');
