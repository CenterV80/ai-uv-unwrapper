import { Engine } from './core/engine.js';

const engine = new Engine();
self.onmessage = async ev => {
  const { id, method, args } = ev.data;
  const progress = (stage, frac, loss) => self.postMessage({ id, progress: { stage, frac, loss } });
  try {
    const result = await engine[method](args, progress);
    self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
self.postMessage({ ready: true });
