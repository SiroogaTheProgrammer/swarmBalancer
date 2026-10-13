// Minimal, safe Markdown -> DOM for the built-in manual: headings, paragraphs, lists, code, tables,
// **bold**, `code` and [links](...). Text always goes through text nodes - documents cannot inject HTML.

import { h } from './util.js';

function inline(text) {
  const out = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[([^\]]+)\]\(([^)]+)\))/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(h('code', {}, m[1].slice(1, -1)));
    else if (m[2]) out.push(h('b', {}, m[2].slice(2, -2)));
    else {
      const href = m[5];
      out.push(/^https?:\/\//.test(href) ? h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, m[4]) : h('u', {}, m[4]));
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function markdown(src) {
  const root = h('div', { class: 'md' });
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const body = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      root.appendChild(h('pre', {}, h('code', {}, body.join('\n'))));
      continue;
    }
    const hd = /^(#{1,4})\s+(.*)$/.exec(line);
    if (hd) { root.appendChild(h(`h${Math.min(3, hd[1].length)}`, {}, inline(hd[2]))); i++; continue; }
    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      const table = h('table');
      body.forEach((r, k) => table.appendChild(h('tr', {}, cells(r).map((c) => h(k === 0 ? 'th' : 'td', {}, inline(c))))));
      root.appendChild(table);
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const list = h(ordered ? 'ol' : 'ul');
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        let item = lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, '');
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i])) item += ' ' + lines[i++].trim();
        list.appendChild(h('li', {}, inline(item)));
      }
      root.appendChild(list);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*\||\s*([-*]|\d+\.)\s+)/.test(lines[i])) para.push(lines[i++].trim());
    root.appendChild(h('p', {}, inline(para.join(' '))));
  }
  return root;
}
