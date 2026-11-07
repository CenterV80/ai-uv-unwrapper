// UV methods compared by the benchmark. Each returns per-corner UVs in [0,1] for mesh.faces,
// plus (optionally) the seam probabilities it used.
import { createRequire } from 'node:module';
import { computeEdgeFeatures } from '../src/core/features.js';
import { SeamPredictor } from '../src/core/model.js';
import { predictSeams } from '../src/core/predict.js';
import { unwrap } from '../src/core/unwrap.js';
import weights from '../src/weights.js';

const predictor = SeamPredictor.fromJSON(weights);
const DIHEDRAL_SEAM_DEG = 30; // baseline: cut where the surface folds by >= 30°

function runPipeline(mesh, ctx, prob, params) {
  const feats = ctx.feats || (ctx.feats = computeEdgeFeatures(mesh));
  const r = unwrap(mesh, prob, feats.dihedral, params);
  return { cornerUV: r.cornerUV, prob, extra: { autoCuts: r.stats.autoCuts, overlapSplits: r.stats.overlapSplits } };
}

export const METHODS = {
  /** learned seam predictor + our chart pipeline */
  ours(mesh, ctx, opts) {
    const feats = ctx.feats || (ctx.feats = computeEdgeFeatures(mesh));
    const { prob } = predictSeams(mesh, predictor, feats);
    return runPipeline(mesh, ctx, prob, { threshold: weights.meta.defaultThreshold, ...opts.params });
  },
  /** same pipeline, seams from a fixed dihedral-angle threshold (no learning) */
  dihedral(mesh, ctx, opts) {
    const feats = ctx.feats || (ctx.feats = computeEdgeFeatures(mesh));
    const t = DIHEDRAL_SEAM_DEG / 180;
    const prob = Float64Array.from(feats.dihedral, a => Math.min(1, a / (2 * t)));
    return runPipeline(mesh, ctx, prob, { ...opts.params, threshold: 0.5 });
  },
  /** same pipeline fed with the artist's seams: isolates seam quality from the parameterizer */
  artistSeams(mesh, ctx, opts) {
    if (!ctx.labels) return { skip: 'no reference UVs' };
    const prob = Float64Array.from(ctx.labels, l => (l === 1 ? 1 : 0));
    return runPipeline(mesh, ctx, prob, { ...opts.params, threshold: 0.5, minFragment: 0 });
  },
  /** the artist's own UV layout, measured as-is (reference, may tile or overlap on purpose) */
  artistUV(mesh, ctx) {
    if (!mesh.cornerUV) return { skip: 'no reference UVs' };
    return { cornerUV: Float32Array.from(mesh.cornerUV) };
  },
  /** xatlas (MIT, WebAssembly build "xatlasjs") with default chart/pack options */
  async xatlas(mesh) {
    if (mesh.nV > 65535) return { skip: 'xatlasjs supports at most 65535 vertices' };
    const xa = await getXatlas();
    xa.createAtlas();
    try {
      // xatlas treats very small faces as degenerate, so normalise the scale (UVs are scale-free)
      xa.addMesh(Uint16Array.from(mesh.faces), Float32Array.from(mesh.pos, x => (x * 10) / mesh.diag));
      const atlas = xa.generateAtlas({}, { padding: 2, resolution: 1024 });
      const out = atlas.meshes[0];
      if (out.index.length !== mesh.faces.length) return { error: `xatlas changed the face count (${out.index.length / 3} vs ${mesh.nF})` };
      const W = atlas.width, H = atlas.height, S = Math.max(W, H) || 1;
      const uv = out.vertex.coords1, cornerUV = new Float32Array(6 * mesh.nF);
      for (let i = 0; i < out.index.length; i++) {
        cornerUV[2 * i] = (uv[2 * out.index[i]] * W) / S;
        cornerUV[2 * i + 1] = (uv[2 * out.index[i] + 1] * H) / S;
      }
      return { cornerUV };
    } finally { xa.destroyAtlas(); }
  },
};

let xatlasPromise = null;
function getXatlas() {
  if (!xatlasPromise) xatlasPromise = (async () => {
    const require = createRequire(import.meta.url);
    const dir = require.resolve('xatlasjs/dist/node/xatlas.js').replace(/xatlas\.js$/, '');
    const createXAtlasModule = require(dir + 'xatlas.js');
    const { Api } = await import(dir + 'api.mjs');
    const XA = Api(createXAtlasModule);
    return new Promise(res => { const o = new XA(() => res(o), p => dir + p); });
  })();
  return xatlasPromise;
}
