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

// Move script to end of body for inline execution
const scriptMatch = html.match(/(<script>[\s\S]*?<\/script>)/);
if (scriptMatch) {
    html = html.replace(scriptMatch[0], '');
    html = html.replace('</body>', () => scriptMatch[0] + '\n  </body>');
}

fs.writeFileSync(indexPath, html);
console.log('Post-build: Processed index.html for Collider');
