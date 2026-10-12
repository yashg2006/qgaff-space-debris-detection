"""Image-level perturbations (applied to the actual pixels, not to feature vectors).
All functions take / return uint8 BGR images. Noise sigmas are expressed on the 0-1 intensity scale."""
import numpy as np, cv2


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0 + rng.normal(0, sigma, img.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def salt_pepper(img, density=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < density / 2] = 0
    out[(m >= density / 2) & (m < density)] = 255
    return out


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def brightness(img, delta):
    """delta = -0.4 -> multiply intensities by 0.6 (eclipse); +0.3 -> x1.3 (solar glare)."""
    return np.clip(img.astype(np.float32) * (1 + delta), 0, 255).astype(np.uint8)


def contrast(img, factor):
    x = img.astype(np.float32)
    return np.clip((x - x.mean()) * factor + x.mean(), 0, 255).astype(np.uint8)


# key, label, function(img, seed) , stochastic?
CONDITIONS = [
    ("clean",     "Original (clean)",           lambda im, s: im,                         False),
    ("g005",      "Gaussian noise σ=0.05",      lambda im, s: gaussian_noise(im, .05, s), True),
    ("g015",      "Gaussian noise σ=0.15",      lambda im, s: gaussian_noise(im, .15, s), True),
    ("g025",      "Gaussian noise σ=0.25",      lambda im, s: gaussian_noise(im, .25, s), True),
    ("sp2",       "Salt-and-pepper 2%",         lambda im, s: salt_pepper(im, .02, s),    True),
    ("blur1",     "Gaussian blur σ=1",          lambda im, s: blur(im, 1.0),              False),
    ("blur2",     "Gaussian blur σ=2",          lambda im, s: blur(im, 2.0),              False),
    ("bm60",      "Brightness −60%",            lambda im, s: brightness(im, -.6),        False),
    ("bm40",      "Brightness −40% (eclipse)",  lambda im, s: brightness(im, -.4),        False),
    ("bm20",      "Brightness −20%",            lambda im, s: brightness(im, -.2),        False),
    ("bp30",      "Brightness +30% (glare)",    lambda im, s: brightness(im, .3),         False),
    ("bp60",      "Brightness +60%",            lambda im, s: brightness(im, .6),         False),
    ("c05",       "Contrast ×0.5",              lambda im, s: contrast(im, .5),           False),
    ("c15",       "Contrast ×1.5",              lambda im, s: contrast(im, 1.5),          False),
]
