// Serve the production artifact under the real GitHub Pages subpath.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';

const root = resolve('dist');
const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.pdf': 'application/pdf',
  '.bib': 'text/plain; charset=utf-8',
};

createServer(async (request, response) => {
  try {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (path === '/DrMAS') {
      response.writeHead(301, { Location: '/DrMAS/' }).end();
      return;
    }
    if (!path.startsWith('/DrMAS/')) {
      response.writeHead(404).end('Not found');
      return;
    }
    const file = resolve(root, path.slice('/DrMAS/'.length) || 'index.html');
    if (!file.startsWith(root + sep)) {
      response.writeHead(403).end('Forbidden');
      return;
    }
    const contents = await readFile(file);
    response.writeHead(200, { 'Content-Type': types[extname(file)] || 'application/octet-stream' });
    response.end(request.method === 'HEAD' ? undefined : contents);
  } catch {
    response.writeHead(404).end('Not found');
  }
}).listen(4173, '127.0.0.1', () => console.log('Production test site: http://127.0.0.1:4173/DrMAS/'));
