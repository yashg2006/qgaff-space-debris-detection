"""Image-level robustness study on the Roboflow test split (123 images, one annotated object each).
For every perturbation: detection P/R/mAP50 (YOLOv8n, conf 0.25 / 0.001, IoU 0.5), class-agnostic localisation
recall, and drift of the 256-d GAP backbone feature (layer 8) used as QGAFF input.
usage: python run_study.py <best.pt> <test_dir_with_images_and_labels> <out.json>"""
import sys, glob, os, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_np import YOLOv8
from perturb import CONDITIONS

W, TD, OUT = sys.argv[1:4]
SEEDS = [0, 1, 2]
m = YOLOv8(W)


def read_gt(p, shape):
    H, Wd = shape[:2]
    c, x, y, w, h = [float(v) for v in open(p).read().split()[:5]]
    return int(c), np.array([(x - w / 2) * Wd, (y - h / 2) * H, (x + w / 2) * Wd, (y + h / 2) * H])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def ap50(records, n_gt):
    # records: list of (conf, is_tp)
    if n_gt == 0: return np.nan
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


imgs = sorted(glob.glob(os.path.join(TD, 'images', '*')))
data = []
for p in imgs:
    im = cv2.imread(p)
    lb = os.path.join(TD, 'labels', os.path.splitext(os.path.basename(p))[0] + '.txt')
    data.append((os.path.basename(p), im, *read_gt(lb, im.shape)))
labels = np.array([d[2] for d in data])


def gap(m):
    return m.l8.mean((1, 2))


def silhouette_cos(F, y):
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    D = 1 - Fn @ Fn.T
    s = []
    for i in range(len(y)):
        same = (y == y[i]); same[i] = False
        if not same.any(): continue
        a = D[i, same].mean()
        b = min(D[i, y == c].mean() for c in np.unique(y) if c != y[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def evaluate(fn, seed):
    recs = {c: [] for c in range(11)}; n_gt = {c: 0 for c in range(11)}
    tp25 = fp25 = fn25 = 0; loc = 0; feats = []; best = []
    for name, im, c, box in data:
        pim = fn(im, seed)
        pr = m.predict(pim, conf=0.001, iou=0.6)
        feats.append(gap(m)); n_gt[c] += 1
        used = False
        for b, s, k in sorted(pr, key=lambda r: -r[1]):
            ok = (k == c) and iou(b, box) >= .5 and not used
            if ok: used = True
            recs[k].append((s, int(ok)))
        hi = [r for r in pr if r[1] >= .25]
        # P/R at conf .25
        hit = any(k == c and iou(b, box) >= .5 for b, s, k in hi)
        tp25 += hit; fn25 += (not hit); fp25 += len(hi) - (1 if hit else 0)
        loc += any(iou(b, box) >= .5 for b, s, k in hi)
    aps = [ap50(recs[c], n_gt[c]) for c in range(11)]
    F = np.array(feats)
    return dict(mAP50=float(np.nanmean(aps)), P=tp25 / max(tp25 + fp25, 1), R=tp25 / (tp25 + fn25),
                loc_recall=loc / len(data)), F


res = {}; clean_F = None
for key, label, fn, stoch in CONDITIONS:
    seeds = SEEDS if stoch else [0]
    rows = []; sims = []; sils = []
    for sd in seeds:
        t = time.time(); r, F = evaluate(fn, sd)
        if key == 'clean': clean_F = F
        cs = (F * clean_F).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(clean_F, axis=1) + 1e-9)
        r['feat_cos_to_clean'] = float(cs.mean()); r['feat_silhouette'] = silhouette_cos(F, labels)
        rows.append(r); print(key, sd, {k: round(v, 3) for k, v in r.items()}, round(time.time() - t), 's', flush=True)
    res[key] = dict(label=label, n_runs=len(rows), **{k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]})
    json.dump(res, open(OUT, 'w'), indent=1)
print('done')
