// The studio API: every call carries this session's token (it arrived in the launch URL's #fragment).

import { sleep } from './util.js';

let TOKEN = '';

export function setToken(t) { TOKEN = t || ''; }
export function hasToken() { return !!TOKEN; }

export async function api(path, { method = 'GET', body, raw = false } = {}) {
  const opts = { method, headers: { 'X-Studio-Token': TOKEN }, cache: 'no-store' };
  if (body !== undefined) {
    if (raw) opts.body = body;
    else { opts.body = JSON.stringify(body); opts.headers['Content-Type'] = 'application/json'; }
  }
  let r;
  try { r = await fetch('/api/' + path, opts); } catch (e) { throw new Error('studio server unreachable - is `python -m swarm.studio` still running?'); }
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const err = new Error((data && data.error) || `${r.status} ${r.statusText}`);
    err.status = r.status;
    throw err;
  }
  return data;
}

/** Polls a job until it finishes; onUpdate(job) sees every state. Rejects with the job's error. */
export async function waitJob(id, onUpdate, interval = 350) {
  for (;;) {
    const job = await api(`jobs/${id}`);
    if (onUpdate) onUpdate(job);
    if (job.state === 'done') return job;
    if (job.state === 'error' || job.state === 'cancelled') {
      const err = new Error(job.error || job.state);
      err.job = job;
      throw err;
    }
    await sleep(interval);
  }
}
