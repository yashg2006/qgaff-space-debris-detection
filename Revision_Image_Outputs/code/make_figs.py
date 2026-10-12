import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from yolo_np import YOLOv8
from perturb import CONDITIONS
U = "./"  # project root; run from there
m = YOLOv8(U + "runs/detect/train3/weights/best.pt")
NAMES = m.names
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
GT_C, PR_C = '#1b9e77', '#d95f02'
CONF, IOU = 0.25, 0.5

def read_gt(p, shape, ):
    H, W = shape[:2]; out = []
    for l in open(p).read().strip().splitlines():
        c, x, y, w, h = [float(v) for v in l.split()[:5]]
        out.append((int(c), np.array([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H])))
    return out

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix*iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0

def draw(ax, img, gts, preds, title, note=None, names=NAMES, lw=1.4):
    ax.imshow(img[:, :, ::-1]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    for c, b in gts:
        ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=GT_C, lw=lw, ls='--'))
    for b, s, c in preds:
        ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=PR_C, lw=lw))
        ax.text(b[0], max(b[1]-4, 10), f"{names.get(c, c)} {s:.2f}", color='white', fontsize=6.5,
                bbox=dict(fc=PR_C, ec='none', pad=1.2))
    ax.set_title(title, fontsize=8, pad=3)
    if note: ax.set_xlabel(note, fontsize=7, labelpad=2)

def run_sample(imgfile, lblfile, tag):
    im = cv2.imread(imgfile); gts = read_gt(lblfile, im.shape)
    outs = []
    for key, label, fn, st in CONDITIONS:
        pim = fn(im, 0)
        pr = m.predict(pim, conf=CONF, iou=0.5)
        outs.append((key, label, pim, pr))
    # Fig A: perturbed gallery (no boxes)
    fig, axs = plt.subplots(2, 7, figsize=(14, 4.6))
    for ax, (key, label, pim, pr) in zip(axs.ravel(), outs):
        ax.imshow(pim[:, :, ::-1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontsize=8)
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(f'figs/{tag}_gallery.png', dpi=170); plt.close(fig)
    # Fig B: detection overlays
    fig, axs = plt.subplots(2, 7, figsize=(14, 5.4))
    rows = []
    for ax, (key, label, pim, pr) in zip(axs.ravel(), outs):
        gt_c, gt_b = gts[0]
        best = max(pr, key=lambda r: r[1]) if pr else None
        if best is None: note = "no detection"; ious = 0
        else:
            ious = max(iou(b, gt_b) for b, s, c in pr)
            note = f"conf {best[1]:.2f} | IoU {iou(best[0], gt_b):.2f}" + ("" if best[2] == gt_c else " | wrong class")
        rows.append(dict(key=key, label=label, n_det=len(pr), top_conf=(best[1] if best else 0), top_iou=(iou(best[0], gt_b) if best else 0),
                         cls_ok=bool(best and best[2] == gt_c)))
        draw(ax, pim, gts, pr, label, note)
    fig.tight_layout(); fig.savefig(f'figs/{tag}_detections.png', dpi=170); plt.close(fig)
    json.dump(rows, open(f'figs/{tag}_rows.json', 'w'), indent=1)
    return im, gts, outs

R = U + "Roboflow Dataset/test/"
n1 = "img021927_jpg.rf.91830c181251b09d0664c76dbafb9d5c"; n2 = "img022036_jpg.rf.5617f92defc23d1b04aa2ee1958f74d4"
im1, gts1, outs1 = run_sample(R + f"images/{n1}.jpg", R + f"labels/{n1}.txt", 's1')
im2, gts2, outs2 = run_sample(R + f"images/{n2}.jpg", R + f"labels/{n2}.txt", 's2')

# Fig C: brightness series with histograms (sample 1)
keys = ['bm60', 'bm40', 'bm20', 'clean', 'bp30', 'bp60']
sel = [o for k in keys for o in outs1 if o[0] == k]
fig, axs = plt.subplots(2, 6, figsize=(13, 4.6), gridspec_kw=dict(height_ratios=[3, 1.1]))
for j, (key, label, pim, pr) in enumerate(sel):
    gt_c, gt_b = gts1[0]
    best = max(pr, key=lambda r: r[1]) if pr else None
    note = "no detection" if best is None else f"conf {best[1]:.2f} | IoU {iou(best[0], gt_b):.2f}"
    draw(axs[0, j], pim, gts1, pr, label, note)
    g = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)
    axs[1, j].hist(g.ravel(), bins=64, range=(0, 255), color='#555', log=True)
    axs[1, j].set_xlim(0, 255); axs[1, j].set_xlabel(f"mean intensity {g.mean():.0f}", fontsize=7)
    axs[1, j].tick_params(labelsize=6); axs[1, j].set_yticks([])
    for s in ('top', 'right', 'left'): axs[1, j].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('figs/s1_brightness.png', dpi=170); plt.close(fig)

# Fig D: dataset-level debris identification (GT vs prediction)
cases = [
 ("Roboflow (in-domain)", R + f"images/{n1}.jpg", R + f"labels/{n1}.txt", NAMES),
 ("Roboflow (in-domain)", R + f"images/{n2}.jpg", R + f"labels/{n2}.txt", NAMES),
 ("SPARK-2022 (cross-domain)", U + "spark-2022-stream-1/val_subset/images/img000098.jpg", U + "spark-2022-stream-1/val_subset/labels/img000098.txt", {i: f"cls{i}" for i in range(11)}),
 ("Streaks/Zenodo (cross-domain)", U + "runs/cross_domain.py/StreaksYoloDataset/test/images/6.jpeg", U + "runs/cross_domain.py/StreaksYoloDataset/test/labels/6.txt", {0: 'streak'}),
]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9)); info = []
for ax, (t, ip, lp, nm) in zip(axs, cases):
    im = cv2.imread(ip); g = read_gt(lp, im.shape); pr = m.predict(im, conf=CONF, iou=0.5)
    pr_low = m.predict(im, conf=0.05, iou=0.5)
    note = (f"{len(pr)} detection(s) at conf≥0.25" if pr else f"no detection at conf≥0.25 ({len(pr_low)} at ≥0.05)")
    draw(ax, im, g, pr, t, note, names=NAMES)
    info.append(dict(case=t, n_det=len(pr), n_det_low=len(pr_low), best=[(b.round(0).tolist(), round(s, 2), NAMES[c]) for b, s, c in pr_low[:3]], gt=[b.round(0).tolist() for c, b in g]))
fig.tight_layout(); fig.savefig('figs/datasets_detection.png', dpi=170); plt.close(fig)
json.dump(info, open('figs/datasets_info.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
