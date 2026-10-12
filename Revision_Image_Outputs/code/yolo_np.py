"""Minimal numpy/OpenCV inference for an Ultralytics YOLOv8 detection checkpoint (best.pt).
Loads the trained weights without torch/ultralytics and reproduces the standard YOLOv8 forward pass
(Conv-BN-SiLU, C2f, SPPF, Upsample, Concat, Detect with DFL), letterbox pre-processing and class-aware NMS."""
import pickle, zipfile, io
import numpy as np, cv2


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        if isinstance(st, dict):
            self.__dict__.update(st)
        else:
            self.state = st


def _stub_class(name):
    return type(name, (_Stub,), {})


class _Storage:
    def __init__(self, arr):
        self.arr = arr


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, *a):
    size = tuple(size)
    n = int(np.prod(size)) if size else 1
    flat = storage.arr[offset:offset + n] if len(size) else storage.arr[offset:offset + 1]
    return np.array(flat, dtype=np.float32).reshape(size) if len(size) else np.float32(flat[0])


def _rebuild_parameter(data, requires_grad, hooks, *a):
    return data


class _Unp(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return _rebuild_parameter
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'torch' and name == 'Size':
            return tuple
        if module == 'collections' and name == 'OrderedDict':
            from collections import OrderedDict
            return OrderedDict
        if module in ('__builtin__', 'builtins') and name == 'set':
            return set
        return _stub_class(name)

    def persistent_load(self, pid):
        typ, st, key, loc, numel = pid
        dt = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'LongStorage': np.int64,
              'DoubleStorage': np.float64}[st]
        raw = self.zf.read(f'{self.prefix}/data/{key}')
        return _Storage(np.frombuffer(raw, dtype=dt))


def load_checkpoint(path):
    zf = zipfile.ZipFile(path)
    pk = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    prefix = pk.rsplit('/', 1)[0]
    return _Unp(io.BytesIO(zf.read(pk)), zf, prefix).load()


def _mods(m):
    return m.__dict__.get('_modules', {})


def _par(m, k):
    return m.__dict__['_parameters'].get(k)


def _fuse_conv(cm):
    """Ultralytics Conv block (conv + bn + SiLU) -> (W, b, stride, pad)."""
    c = _mods(cm)['conv']
    W = _par(c, 'weight').astype(np.float32)
    bn = _mods(cm).get('bn')
    if bn is not None:
        g, be = _par(bn, 'weight'), _par(bn, 'bias')
        mu, var = bn.__dict__['_buffers']['running_mean'], bn.__dict__['_buffers']['running_var']
        eps = bn.__dict__.get('eps', 1e-3)
        s = g / np.sqrt(var + eps)
        W = W * s[:, None, None, None]
        b = be - mu * s
    else:
        b = _par(c, 'bias')
        b = np.zeros(W.shape[0], np.float32) if b is None else b
    return dict(W=W.astype(np.float32), b=b.astype(np.float32), s=c.__dict__['stride'][0],
                p=c.__dict__['padding'][0],
                act=('act' in _mods(cm) and type(_mods(cm)['act']).__name__ == 'SiLU'))


def conv2d(x, W, b, s, p):
    # x: (C,H,W) ; W: (O,C,k,k)
    O, C, k, _ = W.shape
    if k == 1 and s == 1:
        out = W.reshape(O, C) @ x.reshape(C, -1)
        return out.reshape(O, x.shape[1], x.shape[2]) + b[:, None, None]
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    H = (xp.shape[1] - k) // s + 1
    Wd = (xp.shape[2] - k) // s + 1
    sh = (C, k, k, H, Wd)
    st = (xp.strides[0], xp.strides[1], xp.strides[2], xp.strides[1] * s, xp.strides[2] * s)
    cols = np.lib.stride_tricks.as_strided(xp, sh, st).reshape(C * k * k, H * Wd)
    out = W.reshape(O, -1) @ cols
    return out.reshape(O, H, Wd) + b[:, None, None]


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Conv:
    def __init__(self, cm):
        self.d = _fuse_conv(cm)

    def __call__(self, x):
        d = self.d
        y = conv2d(x, d['W'], d['b'], d['s'], d['p'])
        return silu(y) if d['act'] else y


class Bottleneck:
    def __init__(self, m):
        mm = _mods(m)
        self.cv1, self.cv2, self.add = Conv(mm['cv1']), Conv(mm['cv2']), m.__dict__.get('add', False)

    def __call__(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f:
    def __init__(self, m):
        mm = _mods(m)
        self.c = m.__dict__['c']
        self.cv1, self.cv2 = Conv(mm['cv1']), Conv(mm['cv2'])
        self.m = [Bottleneck(b) for b in _mods(mm['m']).values()]

    def __call__(self, x):
        y = self.cv1(x)
        ys = [y[:self.c], y[self.c:]]
        for b in self.m:
            ys.append(b(ys[-1]))
        return self.cv2(np.concatenate(ys, 0))


class SPPF:
    def __init__(self, m):
        mm = _mods(m)
        self.cv1, self.cv2 = Conv(mm['cv1']), Conv(mm['cv2'])
        self.k = 5

    def _mp(self, x):
        k, p = self.k, self.k // 2
        xp = np.pad(x, ((0, 0), (p, p), (p, p)), constant_values=-np.inf)
        return np.stack([cv2.dilate(c, np.ones((k, k), np.uint8)) for c in x]) if False else \
            np.stack([cv2.dilate(xp[i], np.ones((k, k), np.uint8))[p:-p, p:-p] for i in range(x.shape[0])])

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self._mp(x); y2 = self._mp(y1); y3 = self._mp(y2)
        return self.cv2(np.concatenate([x, y1, y2, y3], 0))


class Detect:
    def __init__(self, m):
        mm = _mods(m)
        self.nc, self.reg = m.__dict__['nc'], m.__dict__['reg_max']
        self.stride = np.array(m.__dict__['stride'], dtype=np.float32)
        self.cv2 = [self._branch(b) for b in _mods(mm['cv2']).values()]
        self.cv3 = [self._branch(b) for b in _mods(mm['cv3']).values()]

    @staticmethod
    def _branch(seq):
        layers = list(_mods(seq).values())
        out = [Conv(l) for l in layers[:-1]]
        last = layers[-1]
        W = _par(last, 'weight'); b = _par(last, 'bias')
        return out, (W.astype(np.float32), b.astype(np.float32))

    @staticmethod
    def _run(br, x):
        convs, (W, b) = br
        for c in convs:
            x = c(x)
        return conv2d(x, W, b, 1, 0)

    def __call__(self, feats):
        boxes, clss, anchors, strides = [], [], [], []
        for i, f in enumerate(feats):
            bx = self._run(self.cv2[i], f)
            cl = self._run(self.cv3[i], f)
            H, W = f.shape[1:]
            boxes.append(bx.reshape(4 * self.reg, -1)); clss.append(cl.reshape(self.nc, -1))
            gy, gx = np.meshgrid(np.arange(H) + .5, np.arange(W) + .5, indexing='ij')
            anchors.append(np.stack([gx.ravel(), gy.ravel()], 0)); strides.append(np.full(H * W, self.stride[i]))
        box = np.concatenate(boxes, 1); cls = np.concatenate(clss, 1)
        anc = np.concatenate(anchors, 1); st = np.concatenate(strides)
        b = box.reshape(4, self.reg, -1)
        b = np.exp(b - b.max(1, keepdims=True)); b /= b.sum(1, keepdims=True)
        d = (b * np.arange(self.reg, dtype=np.float32)[None, :, None]).sum(1)  # (4,N) ltrb
        x1y1 = anc - d[:2]; x2y2 = anc + d[2:]
        xyxy = np.concatenate([x1y1, x2y2], 0) * st[None]
        return xyxy.T, (1 / (1 + np.exp(-np.clip(cls, -60, 60)))).T  # (N,4), (N,nc)


class YOLOv8:
    def __init__(self, path):
        ck = load_checkpoint(path)
        model = ck['model']
        self.names = model.__dict__.get('names', None)
        seq = _mods(model)['model']
        self.layers = []
        for l in _mods(seq).values():
            t = type(l).__name__
            f, = [l.__dict__['f']]
            if t == 'Conv': mod = Conv(l)
            elif t == 'C2f': mod = C2f(l)
            elif t == 'SPPF': mod = SPPF(l)
            elif t == 'Detect': mod = Detect(l)
            elif t in ('Upsample', 'Concat'): mod = t
            else: raise RuntimeError(t)
            self.layers.append((t, f, mod, l.__dict__['i']))

    def forward(self, x):
        ys = []
        for t, f, mod, i in self.layers:
            if t == 'Detect':
                return mod([ys[j] for j in f])
            if t == 'Concat':
                inp = [ys[j] if j != -1 else x for j in f]
                x = np.concatenate(inp, 0)
            elif t == 'Upsample':
                x = np.repeat(np.repeat(x, 2, 1), 2, 2)
            else:
                if f != -1:
                    x = ys[f] if isinstance(f, int) else [ys[j] for j in f]
                x = mod(x)
            ys.append(x)
            if len(ys) == 9:
                self.l8 = x  # backbone layer 8 (C2f, 256 ch) used by the GAP feature hook

    @staticmethod
    def letterbox(img, size=640):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = round(h * r), round(w * r)
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
        top, left = (size - nh) // 2, (size - nw) // 2
        out = np.full((size, size, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = im
        return out, r, left, top

    def predict(self, bgr, conf=0.25, iou=0.7, size=640):
        lb, r, left, top = self.letterbox(bgr, size)
        x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        xyxy, cls = self.forward(np.ascontiguousarray(x))
        score = cls.max(1); cid = cls.argmax(1)
        keep = score > conf
        xyxy, score, cid = xyxy[keep], score[keep], cid[keep]
        res = []
        if len(score):
            off = cid[:, None] * 4096.0
            wh = np.concatenate([xyxy[:, :2] + off, xyxy[:, 2:] - xyxy[:, :2]], 1)
            idx = cv2.dnn.NMSBoxes(wh.tolist(), score.tolist(), conf, iou)
            for i in np.array(idx).reshape(-1):
                b = xyxy[i].copy()
                b[[0, 2]] = (b[[0, 2]] - left) / r; b[[1, 3]] = (b[[1, 3]] - top) / r
                H, W = bgr.shape[:2]
                b = np.clip(b, 0, [W, H, W, H])
                res.append((b, float(score[i]), int(cid[i])))
        return res
