// Downloads the Khronos glTF sample models used for training/evaluation into training/models/.
// Cross-platform (Node 18+). Usage: node training/fetch-models.mjs
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';

export const MODELS = [
  'ABeautifulGame', 'AnisotropyBarnLamp', 'AntiqueCamera', 'Avocado', 'BarramundiFish', 'BoomBox', 'BrainStem',
  'CarConcept', 'CesiumMan', 'CesiumMilkTruck', 'ChairDamaskPurplegold', 'ChronographWatch', 'CommercialRefrigerator',
  'Corset', 'DamagedHelmet', 'DiffuseTransmissionTeacup', 'DragonAttenuation', 'Duck', 'Fox', 'GlamVelvetSofa',
  'GlassHurricaneCandleHolder', 'IridescenceSuzanne', 'IridescentDishWithOlives', 'Lantern', 'LightsPunctualLamp',
  'MaterialsVariantsShoe', 'MosquitoInAmber', 'PotOfCoals', 'RiggedFigure', 'ScatteringSkull', 'SheenChair',
  'SheenWoodLeatherSofa', 'ToyCar', 'WaterBottle',
];

const base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models';
const dir = new URL('./models/', import.meta.url);
mkdirSync(dir, { recursive: true });
let failed = 0;
for (const m of MODELS) {
  const dest = new URL(`${m}.glb`, dir);
  if (existsSync(dest)) { console.log('have ', m); continue; }
  try {
    const res = await fetch(`${base}/${m}/glTF-Binary/${m}.glb`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
    console.log('got  ', m);
  } catch (e) { failed++; console.log('FAIL ', m, e.message); }
}
console.log(failed ? `${failed} downloads failed` : 'all models present');
process.exitCode = failed ? 1 : 0;
