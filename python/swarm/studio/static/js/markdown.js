// Markdown -> DOM for the FIELD MANUAL. Builds nodes (never innerHTML of document text): headings with
// anchors, paragraphs, lists, fenced code, tables, block quotes, <details>/<summary>, inline code, emphasis
// and links. Links to other docs/*.md open inside the console.

import { h } from './util.js';

export const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[`*_]/g, '').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-');

const INLINE = /(`+)([\s\S]*?)\1|\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)|\*\*([^*]+?)\*\*|__([^_]+?)__|\*([^*\s][^*]*?)\*|(?<![\w])_([^_]+?)_(?![\w])|<(https?:\/\/[^>\s]+)>/g;

export function inline(text, ctx) {
  const frag = document.createDocumentFragment();
  const re = new RegExp(INLINE.source, 'g'); // own state: inline() recurses into emphasis and link text
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1]) frag.appendChild(h('code', {}, m[2]));
    else if (m[3]) frag.appendChild(link(m[3], m[4], ctx));
    else if (m[5] || m[6]) frag.appendChild(h('strong', {}, inline(m[5] || m[6], ctx)));
    else if (m[7] || m[8]) frag.appendChild(h('em', {}, inline(m[7] || m[8], ctx)));
    else if (m[9]) frag.appendChild(link(m[9], m[9], ctx));
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

function link(label, url, ctx) {
  const content = inline(label, ctx);
  if (/^https?:\/\//.test(url)) return h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', class: 'ext' }, content);
  if (url.startsWith('#')) return h('a', { href: '#', onclick: (e) => { e.preventDefault(); ctx.anchor(url.slice(1)); } }, content);
  const [path, anchor] = url.split('#');
  const doc = ctx.resolve(path);
  if (doc) return h('a', { href: '#', onclick: (e) => { e.preventDefault(); ctx.open(doc, anchor); } }, content);
  return h('span', { title: url }, h('code', {}, content));
}

function table(lines, ctx) {
  const cells = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
  const head = cells(lines[0]);
  const t = h('table', {}, h('thead', {}, h('tr', {}, head.map((c) => h('th', {}, inline(c, ctx))))));
  const body = h('tbody');
  for (const l of lines.slice(2)) body.appendChild(h('tr', {}, cells(l).map((c) => h('td', {}, inline(c, ctx)))));
  t.appendChild(body);
  return t;
}

export function render(md, ctx) {
  const root = document.createDocumentFragment();
  const stack = [root];
  const top = () => stack[stack.length - 1];
  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  let para = [];
  const flush = () => { if (para.length) { top().appendChild(h('p', {}, inline(para.join(' '), ctx))); para = []; } };
  const isSep = (l) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(l || '');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let m;
    if ((m = /^\s*(```|~~~)(.*)$/.exec(line))) {
      flush();
      const fence = m[1], code = [];
      for (i++; i < lines.length && !lines[i].trim().startsWith(fence); i++) code.push(lines[i]);
      top().appendChild(h('pre', {}, h('code', {}, code.join('\n'))));
      continue;
    }
    if (/^\s*<details>\s*(<summary>(.*)<\/summary>)?\s*$/i.test(line)) {
      flush();
      const d = h('details');
      const s = /<summary>(.*)<\/summary>/i.exec(line);
      if (s) d.appendChild(h('summary', {}, inline(s[1], ctx)));
      top().appendChild(d);
      stack.push(d);
      continue;
    }
    if ((m = /^\s*<summary>(.*)<\/summary>\s*$/i.exec(line))) { flush(); top().appendChild(h('summary', {}, inline(m[1], ctx))); continue; }
    if (/^\s*<\/details>\s*$/i.test(line)) { flush(); if (stack.length > 1) stack.pop(); continue; }
    if ((m = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line))) {
      flush();
      const level = Math.min(4, m[1].length);
      top().appendChild(h(`h${level}`, { id: slug(m[2]) }, inline(m[2], ctx)));
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); top().appendChild(h('hr')); continue; }
    if (line.includes('|') && isSep(lines[i + 1])) {
      flush();
      const rows = [line, lines[i + 1]];
      for (i += 2; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) rows.push(lines[i]);
      i--;
      top().appendChild(table(rows, ctx));
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const q = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) q.push(lines[i].replace(/^\s*>\s?/, ''));
      i--;
      const bq = h('blockquote');
      bq.appendChild(render(q.join('\n'), ctx));
      top().appendChild(bq);
      continue;
    }
    if ((m = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line))) {
      flush();
      const ordered = /\d/.test(m[2]);
      const list = h(ordered ? 'ol' : 'ul');
      const base = m[1].length;
      let item = null, nested = null;
      for (; i < lines.length; i++) {
        const l = lines[i];
        const im = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(l);
        if (im && im[1].length <= base + 1) { item = h('li', {}, inline(im[3], ctx)); list.appendChild(item); nested = null; continue; }
        if (im && im[1].length > base + 1 && item) {
          if (!nested) { nested = h(/\d/.test(im[2]) ? 'ol' : 'ul'); item.appendChild(nested); }
          nested.appendChild(h('li', {}, inline(im[3], ctx)));
          continue;
        }
        if (l.trim() && /^\s{2,}\S/.test(l) && item) { item.appendChild(document.createTextNode(' ')); item.appendChild(inline(l.trim(), ctx)); continue; }
        break;
      }
      i--;
      top().appendChild(list);
      continue;
    }
    if (!line.trim()) { flush(); continue; }
    para.push(line.trim());
  }
  flush();
  return root;
}
