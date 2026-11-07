// Checks the statistics helpers against textbook values.
import { wilcoxon, mcnemar, holm, median } from '../tools/statlib.mjs';
let failed = false;
const check = (name, got, want, tol = 1e-3) => { const ok = Math.abs(got - want) <= tol; if (!ok) failed = true; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${got.toFixed(4)} (expected ${want})`); };
// Darwin's Zea mays differences (Hollander & Wolfe); exact two-sided p = 0.04126
check('wilcoxon exact', wilcoxon([6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]).p, 0.04126, 5e-4);
// all positive, n = 5: p = 2 / 2^5 = 0.0625
check('wilcoxon all positive n=5', wilcoxon([1, 2, 3, 4, 5]).p, 0.0625);
// McNemar exact: 1 vs 9 discordant pairs -> p = 2 * (1 + 10) / 1024
const a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], b = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
check('mcnemar exact', mcnemar(a.map(Boolean), b.map(Boolean)).p, 22 / 1024);
const t = [{ p: 0.01 }, { p: 0.04 }, { p: 0.03 }]; holm(t);
check('holm smallest', t[0].pHolm, 0.03); check('holm middle', t[2].pHolm, 0.06); check('holm largest', t[1].pHolm, 0.06);
check('median even', median([4, 1, 3, 2]), 2.5);
process.exitCode = failed ? 1 : 0;
