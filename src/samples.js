// Procedural sample models (no bundled third-party assets). UVs are stripped so the tool makes its own.
import { GARMENTS } from './garments.js';

const GARMENT_LABELS = { tshirt: 'Tシャツ', pants: 'パンツ', skirt: 'スカート', beanie: 'ニット帽' };

export function makeSample(key, THREE, mergeGeometries) {
  if (GARMENTS[key]) {
    const { positions, indices } = GARMENTS[key]();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    return { geometry, label: 'サンプル: ' + GARMENT_LABELS[key] };
  }
  const clean = g => { g = g.index ? g : g; for (const k of Object.keys(g.attributes)) if (k !== 'position') g.deleteAttribute(k); return g; };
  const merge = gs => mergeGeometries(gs.map(g => clean(g.index ? g : g)), false);
  let geometry, label;
  switch (key) {
    case 'robot': {
      const parts = [];
      const box = (w, h, d, x, y, z) => { const g = new THREE.BoxGeometry(w, h, d, 3, 3, 3); g.translate(x, y, z); parts.push(g); };
      const cyl = (r, h, x, y, z, rotZ = 0) => { const g = new THREE.CylinderGeometry(r, r, h, 20, 4); if (rotZ) g.rotateZ(rotZ); g.translate(x, y, z); parts.push(g); };
      box(0.9, 1.0, 0.55, 0, 1.0, 0);          // torso
      box(0.6, 0.5, 0.5, 0, 1.8, 0);           // head
      cyl(0.08, 0.15, 0, 1.5, 0);              // neck
      cyl(0.11, 0.8, -0.62, 1.05, 0); cyl(0.11, 0.8, 0.62, 1.05, 0);   // arms
      cyl(0.14, 0.9, -0.24, 0.15, 0); cyl(0.14, 0.9, 0.24, 0.15, 0);   // legs
      const ant = new THREE.SphereGeometry(0.07, 16, 10); ant.translate(0, 2.18, 0); parts.push(ant);
      cyl(0.02, 0.2, 0, 2.06, 0);
      geometry = merge(parts); label = 'サンプル: ロボット'; break;
    }
    case 'knot': geometry = clean(new THREE.TorusKnotGeometry(0.8, 0.26, 200, 24)); label = 'サンプル: トーラスノット'; break;
    case 'capsule': geometry = clean(new THREE.CapsuleGeometry(0.5, 1.1, 10, 28)); label = 'サンプル: カプセル'; break;
    case 'sphere': geometry = clean(new THREE.IcosahedronGeometry(1, 12)); label = 'サンプル: 球'; break;
    case 'mug':
    default: {
      // closed lathe profile: outer wall, rim, inner wall, inner floor, base
      const pts = [];
      const R = 0.62, Ri = 0.55, H = 1.25, B = 0.08;
      pts.push(new THREE.Vector2(0.0001, 0));
      for (let i = 0; i <= 3; i++) pts.push(new THREE.Vector2(R * 0.92 + R * 0.08 * (i / 3), 0.02 * (1 - i / 3)));
      for (let i = 1; i <= 12; i++) pts.push(new THREE.Vector2(R, (H * i) / 12));
      pts.push(new THREE.Vector2((R + Ri) / 2, H + 0.025));
      for (let i = 0; i <= 12; i++) pts.push(new THREE.Vector2(Ri, H - ((H - B) * i) / 12));
      pts.push(new THREE.Vector2(Ri * 0.5, B)); pts.push(new THREE.Vector2(0.0001, B));
      const body = new THREE.LatheGeometry(pts, 48);
      const handle = new THREE.TorusGeometry(0.3, 0.07, 14, 28, Math.PI * 1.15);
      handle.rotateZ(-Math.PI * 0.575); handle.translate(R + 0.05, H * 0.52, 0);
      geometry = merge([body, handle]); label = 'サンプル: マグカップ';
    }
  }
  return { geometry, label };
}
