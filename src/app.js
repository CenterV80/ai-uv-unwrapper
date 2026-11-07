import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { makeSample } from './samples.js';
import { toOBJ, zipFiles } from './exporters.js';

const $ = id => document.getElementById(id);
const WEIGHTS_KEY = 'seamline-weights-v1';

// ---------------------------------------------------------------- engine (worker with fallback)
class EngineClient {
  constructor() { this.seq = 0; this.pending = new Map(); this.local = null; }
  async init() {
    try {
      this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      await new Promise((res, rej) => {
        const t = setTimeout(() => rej(new Error('worker timeout')), 8000);
        this.worker.onmessage = ev => {
          if (ev.data.ready) { clearTimeout(t); res(); return; }
          const p = this.pending.get(ev.data.id); if (!p) return;
          if (ev.data.progress) { p.onProgress && p.onProgress(ev.data.progress); return; }
          this.pending.delete(ev.data.id);
          ev.data.error ? p.reject(new Error(ev.data.error)) : p.resolve(ev.data.result);
        };
        this.worker.onerror = e => { clearTimeout(t); rej(e); };
      });
    } catch (e) {
      console.warn('Worker unavailable, running on main thread', e);
      this.worker = null;
      const { Engine } = await import('./core/engine.js');
      this.local = new Engine();
    }
  }
  call(method, args, onProgress) {
    if (this.local) {
      return new Promise((res, rej) => setTimeout(async () => {
        try { res(await this.local[method](args, (stage, frac, loss) => onProgress && onProgress({ stage, frac, loss }))); } catch (e) { rej(e); }
      }, 30));
    }
    const id = ++this.seq;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject, onProgress }); this.worker.postMessage({ id, method, args }); });
  }
}
const engine = new EngineClient();

// ---------------------------------------------------------------- state
const S = {
  mesh: null, prob: null, result: null, userSeam: null, origUV: null,
  mode: 'checker', editing: false, name: '', trainedCount: 0, busy: false,
};
const params = () => ({
  threshold: +$('thr').value, maxDistortion: +$('dist').value, padding: +$('pad').value, resolution: +$('res').value,
});

// ---------------------------------------------------------------- theme colours from CSS tokens
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function themeColors() { return { view: css('--view'), ink: css('--ink'), muted: css('--muted'), line: css('--line'), seam: css('--seam'), accent: css('--accent'), panel: css('--panel') }; }

// ---------------------------------------------------------------- 3D view
const c3d = $('c3d');
const renderer = new THREE.WebGLRenderer({ canvas: c3d, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
camera.position.set(2.2, 1.6, 2.6);
const controls = new OrbitControls(camera, c3d);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8090a0, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(3, 5, 4); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-4, -1, -3); scene.add(fill);

const checkerTex = makeCheckerTexture();
const meshMat = new THREE.MeshStandardMaterial({ map: checkerTex, roughness: 0.75, metalness: 0, vertexColors: false, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, side: THREE.DoubleSide });
let displayMesh = null, seamLines = null, hoverLine = null;

function makeCheckerTexture() {
  const N = 1024, cv = document.createElement('canvas'); cv.width = cv.height = N;
  const g = cv.getContext('2d');
  const cells = 16, s = N / cells;
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const hue = (x / cells) * 300;
    const light = (x + y) % 2 ? 78 : 58;
    g.fillStyle = `hsl(${hue} 55% ${light}%)`;
    g.fillRect(x * s, y * s, s, s);
  }
  g.fillStyle = 'rgba(20,24,32,.75)'; g.font = `${Math.round(s * 0.32)}px monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    if ((x + y) % 2) g.fillText(String.fromCharCode(65 + x) + (cells - y), x * s + s / 2, y * s + s / 2);
  }
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function resize() {
  for (const [cv, fn] of [[c3d, null], [$('cuv'), null]]) { void fn; const r = cv.parentElement.getBoundingClientRect(); cv.dataset.w = r.width; cv.dataset.h = r.height; }
  const r = c3d.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height); camera.updateProjectionMatrix();
  drawUV();
}
new ResizeObserver(resize).observe(c3d.parentElement);
new ResizeObserver(resize).observe($('cuv').parentElement);
renderer.setAnimationLoop(() => { controls.update(); renderer.setClearColor(themeColors().view || '#dfe4ea'); renderer.render(scene, camera); });

/** Corner normals with a crease angle so hard-surface models keep their edges. */
function cornerNormals(m, creaseDeg = 40) {
  const { nV, nF, faces, pos } = m;
  const fn = new Float32Array(3 * nF), fa = new Float32Array(nF);
  for (let f = 0; f < nF; f++) {
    const a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, l = Math.hypot(nx, ny, nz) || 1;
    fn[3 * f] = nx / l; fn[3 * f + 1] = ny / l; fn[3 * f + 2] = nz / l; fa[f] = l;
  }
  const start = new Int32Array(nV + 1);
  for (let i = 0; i < 3 * nF; i++) start[faces[i] + 1]++;
  for (let v = 0; v < nV; v++) start[v + 1] += start[v];
  const list = new Int32Array(3 * nF), fillp = start.slice(0, nV);
  for (let f = 0; f < nF; f++) for (let k = 0; k < 3; k++) list[fillp[faces[3 * f + k]]++] = f;
  const cosT = Math.cos(creaseDeg * Math.PI / 180);
  const out = new Float32Array(9 * nF);
  for (let f = 0; f < nF; f++) for (let k = 0; k < 3; k++) {
    const v = faces[3 * f + k];
    let x = 0, y = 0, z = 0;
    for (let i = start[v]; i < start[v + 1]; i++) {
      const g = list[i];
      const d = fn[3 * f] * fn[3 * g] + fn[3 * f + 1] * fn[3 * g + 1] + fn[3 * f + 2] * fn[3 * g + 2];
      if (d >= cosT) { x += fn[3 * g] * fa[g]; y += fn[3 * g + 1] * fa[g]; z += fn[3 * g + 2] * fa[g]; }
    }
    const l = Math.hypot(x, y, z) || 1;
    out[9 * f + 3 * k] = x / l; out[9 * f + 3 * k + 1] = y / l; out[9 * f + 3 * k + 2] = z / l;
  }
  return out;
}

function buildDisplayMesh() {
  const m = S.mesh;
  if (displayMesh) { scene.remove(displayMesh); displayMesh.geometry.dispose(); }
  const g = new THREE.BufferGeometry();
  const P = new Float32Array(9 * m.nF);
  for (let f = 0; f < m.nF; f++) for (let k = 0; k < 3; k++) {
    const v = m.faces[3 * f + k];
    P[9 * f + 3 * k] = m.pos[3 * v]; P[9 * f + 3 * k + 1] = m.pos[3 * v + 1]; P[9 * f + 3 * k + 2] = m.pos[3 * v + 2];
  }
  g.setAttribute('position', new THREE.BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(cornerNormals(m), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(6 * m.nF), 2));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(9 * m.nF).fill(1), 3));
  g.computeBoundingSphere();
  displayMesh = new THREE.Mesh(g, meshMat);
  scene.add(displayMesh);
  // frame camera
  const bs = g.boundingSphere;
  controls.target.copy(bs.center);
  const vfov = camera.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = bs.radius / Math.sin(Math.min(vfov, hfov) / 2) * 1.08;
  const dir = new THREE.Vector3(0.9, 0.55, 1.1).normalize();
  camera.position.copy(bs.center).addScaledVector(dir, dist);
  camera.near = bs.radius / 200; camera.far = bs.radius * 50; camera.updateProjectionMatrix();
}

function chartColor(i) {
  const h = (i * 0.61803398875) % 1;
  return new THREE.Color().setHSL(h, 0.55, 0.62);
}
function probColor(p) {
  const stops = [[0, [0.23, 0.44, 0.85]], [0.35, [0.65, 0.70, 0.77]], [0.65, [0.95, 0.64, 0.23]], [1, [0.88, 0.14, 0.37]]];
  for (let i = 1; i < stops.length; i++) if (p <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (p - a) / (b - a);
    return [ca[0] + t * (cb[0] - ca[0]), ca[1] + t * (cb[1] - ca[1]), ca[2] + t * (cb[2] - ca[2])];
  }
  return stops[3][1];
}

/** Current seam state per edge, as shown to the user. */
function effectiveSeams() {
  const m = S.mesh; if (!m) return null;
  const out = new Uint8Array(m.nE);
  const thr = +$('thr').value;
  for (let e = 0; e < m.nE; e++) {
    let s;
    if (S.userSeam[e]) s = S.userSeam[e] > 0;
    else if (S.result) s = !!S.result.seam[e];
    else s = S.prob ? S.prob[e] >= thr : false;
    out[e] = s && !m.boundary[e] ? 1 : 0;
  }
  return out;
}

function refresh3D() {
  if (!displayMesh) return;
  const m = S.mesh, g = displayMesh.geometry;
  const uvA = g.attributes.uv, colA = g.attributes.color;
  const mode = S.mode;
  const uvSrc = mode === 'orig' ? S.origUV : (S.result ? S.result.cornerUV : null);
  if (uvSrc) uvA.array.set(uvSrc); else uvA.array.fill(0);
  uvA.needsUpdate = true;
  meshMat.map = (mode === 'checker' || mode === 'orig') && uvSrc ? checkerTex : null;
  meshMat.vertexColors = mode === 'charts' || mode === 'prob' || !uvSrc;
  const C = colA.array;
  if (mode === 'charts' && S.result) {
    for (let f = 0; f < m.nF; f++) { const c = chartColor(S.result.faceChart[f]); for (let k = 0; k < 3; k++) C.set([c.r, c.g, c.b], 9 * f + 3 * k); }
  } else C.fill(mode === 'prob' ? 0.82 : 0.9);
  colA.needsUpdate = true;
  meshMat.needsUpdate = true;

  // lines
  if (seamLines) { scene.remove(seamLines); seamLines.geometry.dispose(); seamLines = null; }
  const pts = [], cols = [];
  const pushEdge = (e, c) => {
    const a = m.edgeV[2 * e], b = m.edgeV[2 * e + 1];
    pts.push(m.pos[3 * a], m.pos[3 * a + 1], m.pos[3 * a + 2], m.pos[3 * b], m.pos[3 * b + 1], m.pos[3 * b + 2]);
    cols.push(...c, ...c);
  };
  if (mode === 'prob' && S.prob) {
    // show the upper range only, scaled around the current threshold, so the map stays readable
    const thr = +$('thr').value, lo = thr * 0.6;
    for (let e = 0; e < m.nE; e++) if (!m.boundary[e] && S.prob[e] > lo) pushEdge(e, probColor(Math.min(1, (S.prob[e] - lo) / (1 - lo))));
  } else if (mode === 'orig' && S.mesh.uvLabels) {
    const c = new THREE.Color(themeColors().seam); for (let e = 0; e < m.nE; e++) if (S.mesh.uvLabels[e] === 1) pushEdge(e, [c.r, c.g, c.b]);
  } else {
    const seams = effectiveSeams();
    const red = new THREE.Color(themeColors().seam), blue = new THREE.Color(themeColors().accent);
    for (let e = 0; e < m.nE; e++) {
      if (S.userSeam[e] < 0) pushEdge(e, [blue.r, blue.g, blue.b]);
      else if (seams[e]) pushEdge(e, [red.r, red.g, red.b]);
    }
  }
  if (pts.length) {
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    lg.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    seamLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: true }));
    seamLines.renderOrder = 2;
    scene.add(seamLines);
  }
  $('legend').hidden = mode !== 'prob';
}

// ---------------------------------------------------------------- UV view (2D canvas)
const cuv = $('cuv');
const uvView = { zoom: 1, ox: 0, oy: 0 };
function drawUV() {
  const r = cuv.parentElement.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cuv.width = Math.max(1, Math.round(r.width * dpr)); cuv.height = Math.max(1, Math.round(r.height * dpr));
  const g = cuv.getContext('2d');
  const T = themeColors();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = T.view; g.fillRect(0, 0, cuv.width, cuv.height);
  const W = r.width, H = r.height;
  const size = Math.min(W, H - 40) * 0.86 * uvView.zoom;
  const x0 = (W - size) / 2 + uvView.ox, y0 = (H - size) / 2 + 14 + uvView.oy;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  // texture square
  g.fillStyle = T.panel; g.fillRect(x0, y0, size, size);
  g.strokeStyle = T.line; g.lineWidth = 1;
  for (let i = 1; i < 8; i++) { const t = (i / 8) * size; g.beginPath(); g.moveTo(x0 + t, y0); g.lineTo(x0 + t, y0 + size); g.moveTo(x0, y0 + t); g.lineTo(x0 + size, y0 + t); g.stroke(); }
  g.strokeStyle = T.muted; g.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);
  g.fillStyle = T.muted; g.font = '11px "IBM Plex Mono", monospace';
  g.fillText('0,0', x0, y0 + size + 14); g.fillText('1,1', x0 + size - 22, y0 - 5);
  const uv = S.mode === 'orig' ? S.origUV : S.result && S.result.cornerUV;
  if (!uv || !S.mesh) {
    g.fillStyle = T.muted; g.font = '13px "IBM Plex Sans JP", sans-serif';
    g.fillText(S.mode === 'orig' ? '元のUVはありません' : '「UV展開を実行」でここにレイアウトが出ます', x0 + 16, y0 + size / 2);
    return;
  }
  const m = S.mesh, nF = m.nF;
  const X = u => x0 + u * size, Y = v => y0 + (1 - v) * size;
  // fill per chart
  if (S.mode !== 'orig' && S.result) {
    const byChart = new Map();
    for (let f = 0; f < nF; f++) { const c = S.result.faceChart[f]; if (!byChart.has(c)) byChart.set(c, []); byChart.get(c).push(f); }
    for (const [c, fs] of byChart) {
      const col = chartColor(c);
      g.fillStyle = `rgba(${col.r * 255 | 0},${col.g * 255 | 0},${col.b * 255 | 0},0.55)`;
      g.beginPath();
      for (const f of fs) { g.moveTo(X(uv[6 * f]), Y(uv[6 * f + 1])); g.lineTo(X(uv[6 * f + 2]), Y(uv[6 * f + 3])); g.lineTo(X(uv[6 * f + 4]), Y(uv[6 * f + 5])); g.closePath(); }
      g.fill();
    }
  }
  // wireframe (skip when too dense to read)
  if (nF < 60000) {
    g.strokeStyle = T.ink; g.globalAlpha = Math.max(0.12, Math.min(0.45, 3000 / nF)); g.lineWidth = 0.6;
    g.beginPath();
    for (let f = 0; f < nF; f++) { g.moveTo(X(uv[6 * f]), Y(uv[6 * f + 1])); g.lineTo(X(uv[6 * f + 2]), Y(uv[6 * f + 3])); g.lineTo(X(uv[6 * f + 4]), Y(uv[6 * f + 5])); g.closePath(); }
    g.stroke(); g.globalAlpha = 1;
  }
  // chart borders = seams (edges whose two faces disagree in UV)
  g.strokeStyle = T.seam; g.lineWidth = 1.2; g.beginPath();
  for (let e = 0; e < m.nE; e++) {
    const f0 = m.edgeF[2 * e], f1 = m.edgeF[2 * e + 1];
    let border = f1 < 0;
    if (!border) {
      if (S.mode === 'orig') border = m.uvLabels && m.uvLabels[e] === 1;
      else border = !!S.result.seam[e];
    }
    if (!border) continue;
    for (const f of f1 < 0 ? [f0] : [f0, f1]) {
      const a = m.edgeV[2 * e], b = m.edgeV[2 * e + 1];
      const ka = cornerIdx(m, f, a), kb = cornerIdx(m, f, b);
      g.moveTo(X(uv[6 * f + 2 * ka]), Y(uv[6 * f + 2 * ka + 1])); g.lineTo(X(uv[6 * f + 2 * kb]), Y(uv[6 * f + 2 * kb + 1]));
    }
  }
  g.stroke();
}
function cornerIdx(m, f, v) { return m.faces[3 * f] === v ? 0 : m.faces[3 * f + 1] === v ? 1 : 2; }

cuv.addEventListener('wheel', e => {
  e.preventDefault();
  const k = Math.exp(-e.deltaY * 0.0015);
  const r = cuv.getBoundingClientRect();
  const cx = e.clientX - r.left - r.width / 2 - uvView.ox, cy = e.clientY - r.top - r.height / 2 - uvView.oy;
  uvView.ox -= cx * (k - 1); uvView.oy -= cy * (k - 1);
  uvView.zoom = Math.max(0.3, Math.min(40, uvView.zoom * k));
  drawUV();
}, { passive: false });
{
  let drag = null;
  cuv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, ox: uvView.ox, oy: uvView.oy }; cuv.setPointerCapture(e.pointerId); });
  cuv.addEventListener('pointermove', e => { if (!drag) return; uvView.ox = drag.ox + e.clientX - drag.x; uvView.oy = drag.oy + e.clientY - drag.y; drawUV(); });
  cuv.addEventListener('pointerup', () => { drag = null; });
  cuv.addEventListener('dblclick', () => { uvView.zoom = 1; uvView.ox = uvView.oy = 0; drawUV(); });
}

// ---------------------------------------------------------------- edge picking (seam editing)
const raycaster = new THREE.Raycaster();
function pickEdge(ev) {
  if (!displayMesh) return -1;
  const r = c3d.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(displayMesh, false)[0];
  if (!hit) return -1;
  const f = hit.faceIndex, m = S.mesh, p = hit.point;
  let best = -1, bd = Infinity;
  for (let k = 0; k < 3; k++) {
    const e = m.faceEdges[3 * f + k];
    const a = m.edgeV[2 * e], b = m.edgeV[2 * e + 1];
    const A = new THREE.Vector3(m.pos[3 * a], m.pos[3 * a + 1], m.pos[3 * a + 2]), B = new THREE.Vector3(m.pos[3 * b], m.pos[3 * b + 1], m.pos[3 * b + 2]);
    const d = new THREE.Line3(A, B).closestPointToPoint(p, true, new THREE.Vector3()).distanceTo(p);
    if (d < bd) { bd = d; best = e; }
  }
  return best;
}
function showHover(e) {
  if (hoverLine) { scene.remove(hoverLine); hoverLine.geometry.dispose(); hoverLine = null; }
  if (e < 0) return;
  const m = S.mesh, a = m.edgeV[2 * e], b = m.edgeV[2 * e + 1];
  const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(m.pos[3 * a], m.pos[3 * a + 1], m.pos[3 * a + 2]), new THREE.Vector3(m.pos[3 * b], m.pos[3 * b + 1], m.pos[3 * b + 2])]);
  hoverLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: themeColors().ink, depthTest: false }));
  hoverLine.renderOrder = 3; scene.add(hoverLine);
}
{
  let down = null;
  c3d.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
  c3d.addEventListener('pointermove', e => { if (S.editing && !e.buttons) showHover(pickEdge(e)); });
  c3d.addEventListener('pointerup', e => {
    if (!S.editing || !down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    const edge = pickEdge(e);
    if (edge < 0 || S.mesh.boundary[edge]) return;
    const cur = effectiveSeams()[edge];
    S.userSeam[edge] = cur ? -1 : 1;
    refresh3D();
    scheduleUnwrap();
  });
}
let unwrapTimer = 0;
function scheduleUnwrap() { clearTimeout(unwrapTimer); unwrapTimer = setTimeout(() => runUnwrap(), 350); }

// ---------------------------------------------------------------- loading
async function geometryFromObject(root) {
  root.updateMatrixWorld(true);
  const geos = [];
  let allUV = true;
  root.traverse(o => {
    if (!o.isMesh || !o.geometry || !o.geometry.attributes.position) return;
    let g = o.geometry.clone();
    for (const k of Object.keys(g.attributes)) if (!['position', 'uv'].includes(k)) g.deleteAttribute(k);
    g.morphAttributes = {};
    if (!g.attributes.uv) allUV = false;
    g.applyMatrix4(o.matrixWorld); // also fixes winding for mirrored transforms
    if (o.matrixWorld.determinant() < 0 && g.index) {
      const idx = g.index.array; for (let i = 0; i < idx.length; i += 3) { const t = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = t; }
    }
    if (!g.index) g.setIndex([...Array(g.attributes.position.count).keys()]);
    geos.push(g);
  });
  if (!geos.length) throw new Error('メッシュが見つかりませんでした');
  if (!allUV) for (const g of geos) g.deleteAttribute('uv');
  const merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false);
  return merged;
}

async function loadGeometry(geo, name) {
  setBusy(true, '解析中…');
  try {
    const pos = geo.attributes.position, idx = geo.index;
    const positions = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) { positions[3 * i] = pos.getX(i); positions[3 * i + 1] = pos.getY(i); positions[3 * i + 2] = pos.getZ(i); }
    const indices = idx ? Uint32Array.from(idx.array) : Uint32Array.from({ length: pos.count }, (_, i) => i);
    let cornerUV = null;
    if (geo.attributes.uv) {
      const uv = geo.attributes.uv; cornerUV = new Float32Array(indices.length * 2);
      for (let i = 0; i < indices.length; i++) { cornerUV[2 * i] = uv.getX(indices[i]); cornerUV[2 * i + 1] = uv.getY(indices[i]); }
    }
    const info = await engine.call('load', { positions, indices, cornerUV });
    S.mesh = info; S.name = name; S.prob = null; S.result = null;
    S.userSeam = new Int8Array(info.nE);
    S.origUV = null;
    if (info.hasUV) { // original UVs per surviving face come from the engine's welded mesh; rebuild them here
      S.origUV = await engine.call('origUV', {});
    }
    document.querySelector('[data-mode="orig"]').hidden = !info.hasUV;
    if (S.mode === 'orig' && !info.hasUV) setMode('checker');
    $('learnUV').disabled = !info.hasUV;
    $('modelInfo').textContent = `${name} — 頂点 ${info.nV.toLocaleString()} / 面 ${info.nF.toLocaleString()}${info.hasUV ? ` / 元UVのシーム ${info.uvSeams.toLocaleString()} 本` : ''}`;
    $('sF').textContent = info.nF.toLocaleString();
    buildDisplayMesh();
    uvView.zoom = 1; uvView.ox = uvView.oy = 0;
    await runPredict();
    await runUnwrap();
  } catch (e) {
    toast('読み込めませんでした: ' + e.message);
    console.error(e);
  } finally { setBusy(false); }
}

async function loadFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  try {
    let root;
    if (ext === 'obj') root = new OBJLoader().parse(await file.text());
    else if (ext === 'glb' || ext === 'gltf') {
      const data = ext === 'glb' ? await file.arrayBuffer() : await file.text();
      root = (await new GLTFLoader().parseAsync(data, '')).scene;
    } else throw new Error('対応形式は OBJ / GLB / glTF(埋め込み) です');
    await loadGeometry(await geometryFromObject(root), file.name);
  } catch (e) { toast('読み込めませんでした: ' + e.message); console.error(e); }
}

// ---------------------------------------------------------------- actions
async function runPredict() {
  const r = await engine.call('predict', {});
  S.prob = r.prob;
  const box = $('evalBox');
  if (r.eval) {
    box.hidden = false;
    box.innerHTML = `<dt>元UVとの一致 AUC</dt><dd>${r.eval.auc.toFixed(3)}</dd><dt>適合率 / 再現率</dt><dd>${(r.eval.precision * 100).toFixed(0)}% / ${(r.eval.recall * 100).toFixed(0)}%</dd>`;
  } else box.hidden = true;
  refresh3D();
}

let unwrapRunning = false, unwrapAgain = false;
async function runUnwrap() {
  if (!S.mesh) return;
  if (unwrapRunning) { unwrapAgain = true; return; }
  unwrapRunning = true;
  setBusy(true, '展開中…');
  try {
    const r = await engine.call('unwrap', { params: params(), userSeam: S.userSeam }, p => setBusy(true, p.stage === 'pack' ? 'パッキング中…' : `展開中… ${Math.round(p.frac * 100)}%`));
    S.result = r;
    const st = r.stats;
    $('sS').textContent = st.seamEdges.toLocaleString() + (st.autoCuts ? ` (+自動 ${st.autoCuts})` : '');
    $('sC').textContent = st.charts.toLocaleString();
    $('sD').textContent = st.distortion.toFixed(3) + (st.flips ? ` · 反転 ${st.flips}` : '');
    $('sP').textContent = (st.packingEfficiency * 100).toFixed(1) + '%';
    $('sT').textContent = (st.ms / 1000).toFixed(2) + ' s';
    for (const id of ['expObj', 'expGlb', 'expPng', 'learnSeams']) $(id).disabled = false;
    refresh3D(); drawUV();
  } catch (e) { toast('展開に失敗しました: ' + e.message); console.error(e); }
  finally {
    unwrapRunning = false; setBusy(false);
    if (unwrapAgain) { unwrapAgain = false; runUnwrap(); }
  }
}

async function runTrain(source) {
  if (!S.mesh) return;
  const prog = $('trainProg'); prog.hidden = false; prog.value = 0;
  for (const id of ['learnUV', 'learnSeams']) $(id).disabled = true;
  try {
    const args = { source, epochs: 15 };
    if (source === 'seams') args.seams = S.result ? S.result.seam : effectiveSeams();
    const r = await engine.call('train', args, p => { prog.value = (p.stage === 'stage1' ? 0 : 0.5) + p.frac * 0.5; $('trainInfo').textContent = `学習中 ${p.stage === 'stage1' ? '第1段' : '第2段'}… loss ${p.loss.toFixed(3)}`; });
    S.prob = r.prob; S.trainedCount++;
    if (r.threshold) { $('thr').value = r.threshold; $('thrOut').value = r.threshold.toFixed(2); }
    try { localStorage.setItem(WEIGHTS_KEY, JSON.stringify({ weights: r.weights, count: S.trainedCount, threshold: r.threshold })); } catch { /* storage unavailable */ }
    $('modelState').textContent = `追加学習 ${S.trainedCount}回`; $('modelState').classList.add('on');
    const b = r.before, a = r.after;
    $('trainInfo').textContent = b ? `学習しました。このモデルでのAUC ${b.auc.toFixed(3)} → ${a.auc.toFixed(3)}、F1 ${b.f1.toFixed(2)} → ${a.f1.toFixed(2)}` : '学習しました。';
    S.result = null; refresh3D();
    await runUnwrap();
  } catch (e) { toast('学習できませんでした: ' + e.message); }
  finally { prog.hidden = true; $('learnUV').disabled = !S.mesh.hasUV; $('learnSeams').disabled = !S.result; }
}

// ---------------------------------------------------------------- export
// Inside the claude.ai viewer, files go through the `downloads` capability (formats it won't
// save directly are wrapped in a .zip); anywhere else a plain link download is used.
const SAVE_OK = /\.(png|json|zip|txt)$/i;
let downloadsCap;
async function download(blob, name) {
  if (downloadsCap === undefined) downloadsCap = window.claude?.use ? await window.claude.use('downloads').catch(() => null) : null;
  if (downloadsCap) {
    let data = blob, fname = name;
    if (!SAVE_OK.test(name)) { data = await zipFiles([{ name, data: blob }]); fname = name + '.zip'; }
    try { await downloadsCap.save({ filename: fname, data }); return true; }
    catch (e) { if (e && e.code !== 'declined') toast('保存できませんでした: ' + (e.message || e.code)); return false; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  return true;
}
const baseName = () => (S.name || 'model').replace(/\.[^.]+$/, '').replace(/^サンプル:\s*/, '').replace(/[^\p{L}\p{N}\-_]+/gu, '_').replace(/^_+|_+$/g, '') || 'model';
$('expObj').onclick = async () => { if (await download(new Blob([toOBJ(S.mesh, S.result.cornerUV)], { type: 'text/plain' }), baseName() + '_uv.obj')) toast('OBJを書き出しました'); };
$('expGlb').onclick = async () => {
  const g = displayMesh.geometry.clone();
  g.attributes.uv.array.set(S.result.cornerUV);
  g.deleteAttribute('color');
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: checkerTex }));
  const buf = await new GLTFExporter().parseAsync(mesh, { binary: true });
  if (await download(new Blob([buf], { type: 'model/gltf-binary' }), baseName() + '_uv.glb')) toast('GLBを書き出しました（チェッカーテクスチャ付き）');
};
$('expPng').onclick = () => {
  const N = +$('res').value, cv = document.createElement('canvas'); cv.width = cv.height = N;
  const g = cv.getContext('2d'); const uv = S.result.cornerUV, m = S.mesh;
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, N, N);
  for (let f = 0; f < m.nF; f++) {
    const c = chartColor(S.result.faceChart[f]);
    g.fillStyle = `rgb(${c.r * 255 | 0},${c.g * 255 | 0},${c.b * 255 | 0})`;
    g.beginPath(); g.moveTo(uv[6 * f] * N, (1 - uv[6 * f + 1]) * N); g.lineTo(uv[6 * f + 2] * N, (1 - uv[6 * f + 3]) * N); g.lineTo(uv[6 * f + 4] * N, (1 - uv[6 * f + 5]) * N); g.closePath(); g.fill();
  }
  cv.toBlob(async b => { if (await download(b, baseName() + '_uvlayout.png')) toast('UV図を書き出しました'); });
};

// ---------------------------------------------------------------- weights persistence
$('saveW').onclick = async () => {
  let json; try { json = localStorage.getItem(WEIGHTS_KEY); } catch { json = null; }
  const w = json ? JSON.parse(json).weights : (await import('./weights.js')).default;
  download(new Blob([JSON.stringify(w)], { type: 'application/json' }), 'seamline-model.json');
};
$('loadW').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const w = JSON.parse(await f.text());
    await engine.call('setWeights', w);
    try { localStorage.setItem(WEIGHTS_KEY, JSON.stringify({ weights: w, count: 1 })); } catch { /* ignore */ }
    S.trainedCount = 1; $('modelState').textContent = '読み込んだモデル'; $('modelState').classList.add('on');
    toast('モデルを読み込みました'); if (S.mesh) { await runPredict(); await runUnwrap(); }
  } catch (err) { toast('モデルファイルを読めませんでした: ' + err.message); }
  e.target.value = '';
};
$('resetW').onclick = async () => {
  await engine.call('setWeights', null);
  try { localStorage.removeItem(WEIGHTS_KEY); } catch { /* ignore */ }
  S.trainedCount = 0; $('modelState').textContent = 'ベースモデル'; $('modelState').classList.remove('on');
  toast('ベースモデルに戻しました'); if (S.mesh) { await runPredict(); await runUnwrap(); }
};

// ---------------------------------------------------------------- UI wiring
function setBusy(on, text) { $('busy').hidden = !on; if (text) $('busy').textContent = text; }
let toastT = 0;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, 3600); }
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll('.seg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
  refresh3D(); drawUV();
}
document.querySelectorAll('.seg button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
$('thr').oninput = () => { $('thrOut').value = (+$('thr').value).toFixed(2); S.result = null; refresh3D(); };
$('thr').onchange = () => runUnwrap();
$('dist').oninput = () => { $('distOut').value = (+$('dist').value).toFixed(2); };
$('dist').onchange = () => runUnwrap();
$('pad').onchange = $('res').onchange = () => runUnwrap();
$('run').onclick = () => runUnwrap();
$('edit').onclick = () => {
  S.editing = !S.editing; $('edit').setAttribute('aria-pressed', String(S.editing));
  if (!S.editing) showHover(-1);
  if (S.editing && S.mode === 'prob') setMode('checker');
};
$('clearEdits').onclick = () => { if (!S.mesh) return; S.userSeam.fill(0); refresh3D(); runUnwrap(); };
$('learnUV').onclick = () => runTrain('uv');
$('learnSeams').onclick = () => runTrain('seams');
$('file').onchange = e => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ''; };
$('sample').onchange = () => loadSample($('sample').value);
const pane = $('pane3d');
pane.addEventListener('dragover', e => { e.preventDefault(); pane.classList.add('drop'); });
pane.addEventListener('dragleave', () => pane.classList.remove('drop'));
pane.addEventListener('drop', e => { e.preventDefault(); pane.classList.remove('drop'); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { refresh3D(); drawUV(); });
new MutationObserver(() => { refresh3D(); drawUV(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

function loadSample(key) {
  const { geometry, label } = makeSample(key, THREE, mergeGeometries);
  return loadGeometry(geometry, label);
}

// ---------------------------------------------------------------- boot
(async () => {
  setBusy(true, 'AIモデルを準備中…');
  await engine.init();
  try {
    const meta = await engine.call('baseMeta', {});
    if (meta && meta.defaultThreshold) { $('thr').value = meta.defaultThreshold; $('thrOut').value = (+meta.defaultThreshold).toFixed(2); }
  } catch { /* keep slider default */ }
  try {
    const saved = JSON.parse(localStorage.getItem(WEIGHTS_KEY) || 'null');
    if (saved && saved.weights) {
      await engine.call('setWeights', saved.weights);
      if (saved.threshold) { $('thr').value = saved.threshold; $('thrOut').value = (+saved.threshold).toFixed(2); }
      S.trainedCount = saved.count || 1; $('modelState').textContent = `追加学習 ${S.trainedCount}回`; $('modelState').classList.add('on');
    }
  } catch { /* no storage */ }
  resize();
  await loadSample('mug');
})();
