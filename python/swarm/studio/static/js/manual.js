// FIELD MANUAL: the repository's documentation (docs/*.md, README.md) rendered in the console.

import { api } from './api.js';
import { render, slug } from './markdown.js';
import { $, clear, h, setChildren, toast } from './util.js';

export class Manual {
  constructor(app) {
    this.app = app;
    this.docs = [];
    this.current = null;
  }

  async init() {
    try { this.docs = (await api('docs')).docs; } catch (e) { toast(e.message, 'err'); return; }
    const ul = clear($('#doclist'));
    for (const d of this.docs) {
      ul.appendChild(h('li', { dataset: { id: d.id }, onclick: () => this.open(d.id) }, d.title, h('small', {}, d.path)));
    }
  }

  resolve(path) {
    const name = path.split('/').pop().toLowerCase();
    const d = this.docs.find((x) => x.path.split('/').pop().toLowerCase() === name);
    return d ? d.id : null;
  }

  async open(id, anchor) {
    if (!this.docs.length) await this.init();
    let doc;
    try { doc = await api(`docs/${id}`); } catch (e) { toast(e.message, 'err'); return; }
    this.current = id;
    [...$('#doclist').children].forEach((li) => li.classList.toggle('sel', li.dataset.id === id));
    const ctx = {
      resolve: (p) => this.resolve(p),
      open: (d, a) => this.open(d, a),
      anchor: (a) => this.scrollTo(a),
    };
    const box = $('#doc');
    setChildren(box, render(doc.markdown, ctx));
    const toc = clear($('#toc'));
    box.querySelectorAll('h2, h3').forEach((el) => {
      toc.appendChild(h('a', { href: '#', class: el.tagName.toLowerCase(), onclick: (e) => { e.preventDefault(); el.scrollIntoView({ behavior: 'smooth' }); } }, el.textContent));
    });
    box.scrollTop = 0;
    if (anchor) setTimeout(() => this.scrollTo(anchor), 30);
    this.app.show('manual');
  }

  scrollTo(anchor) {
    const el = document.getElementById(slug(anchor)) || document.getElementById(anchor);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  }
}
