// OBJ writer: welded positions, one vt per distinct (chart corner) UV.
export function toOBJ(mesh, cornerUV) {
  const lines = ['# Seamline UV export', `# vertices ${mesh.nV}, faces ${mesh.nF}`];
  for (let v = 0; v < mesh.nV; v++) lines.push(`v ${fmt(mesh.pos[3 * v])} ${fmt(mesh.pos[3 * v + 1])} ${fmt(mesh.pos[3 * v + 2])}`);
  const vtIndex = new Map();
  const vt = [];
  const faceVT = new Int32Array(3 * mesh.nF);
  for (let i = 0; i < 3 * mesh.nF; i++) {
    const u = cornerUV[2 * i], w = cornerUV[2 * i + 1];
    const key = `${mesh.faces[i]}:${u.toFixed(6)}:${w.toFixed(6)}`;
    let id = vtIndex.get(key);
    if (id === undefined) { id = vt.length; vtIndex.set(key, id); vt.push(`vt ${u.toFixed(6)} ${w.toFixed(6)}`); }
    faceVT[i] = id;
  }
  lines.push(...vt);
  for (let f = 0; f < mesh.nF; f++) {
    const a = 3 * f;
    lines.push(`f ${mesh.faces[a] + 1}/${faceVT[a] + 1} ${mesh.faces[a + 1] + 1}/${faceVT[a + 1] + 1} ${mesh.faces[a + 2] + 1}/${faceVT[a + 2] + 1}`);
  }
  return lines.join('\n') + '\n';
}
const fmt = x => +x.toFixed(6);

// Minimal ZIP (stored, no compression) so formats the artifact viewer won't save directly can be wrapped.
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
export async function zipFiles(files) { // [{name, data: Blob}]
  const parts = [], central = []; let offset = 0;
  for (const f of files) {
    const data = new Uint8Array(await f.data.arrayBuffer()), name = new TextEncoder().encode(f.name), crc = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint32(14, crc, true);
    h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, name.length, true);
    parts.push(h, name, data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint32(16, crc, true);
    c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, offset, true);
    central.push(c, name);
    offset += 30 + name.length + data.length;
  }
  const size = central.reduce((s, p) => s + p.byteLength, 0);
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, size, true); e.setUint32(16, offset, true);
  return new Blob([...parts, ...central, e], { type: 'application/zip' });
}
