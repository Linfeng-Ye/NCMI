# five_base_clusters_merge_to_three_tsne.py
# Build 5 natural-looking base clusters (no rings), merge them into 3 final classes, visualize via t-SNE.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
RNG_SEED = 13
N_TOTAL_POINTS = 3000     # overall budget (incl. optional outliers)
N_DIM = 5                 # high-D makes t-SNE more meaningful
OUTLIER_FRACTION = 0.04   # set to 0 to remove background outliers

# Merge scheme: base cluster ids -> final class id (0,1,2)
MERGE_MAP = {
    0: 0,  # base 0 & 1 -> class 0
    1: 0,
    2: 1,  # base 2 & 3 -> class 1
    3: 1,
    4: 2   # base 4 -> class 2
}

# --------------- Helpers ----------------
rng = np.random.default_rng(RNG_SEED)

def random_rotation_matrix(d, rng):
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q

def make_gaussian_blob(n, d, rng):
    center = rng.normal(scale=3.0, size=d)
    R = random_rotation_matrix(d, rng)
    eig = 10 ** rng.uniform(-0.2, 0.7, size=d)  # anisotropy
    cov = R @ np.diag(eig) @ R.T
    X = rng.normal(size=(n, d)) @ cov + center
    return X

def make_streak(n, d, rng):
    """Elongated curvy open band (no closed loops)."""
    t = np.linspace(0, 1, n)
    base = np.stack([6*t - 3, np.sin(3*np.pi*t), 0.5*np.cos(2*np.pi*t)], axis=1)
    if d > 3:
        extra = rng.normal(scale=0.3, size=(n, d-3))
        base = np.concatenate([base, extra], axis=1)
    noise = rng.normal(scale=0.15, size=(n, d))
    R = random_rotation_matrix(d, rng)
    shift = rng.normal(scale=2.5, size=d)
    return base @ R + noise + shift

def make_half_moon_band(n, d, rng):
    """Open crescent-like band (not a ring)."""
    angles = rng.uniform(-0.8*np.pi, 0.8*np.pi, size=n)  # open arc (not closed)
    r = 1.2 + 0.3 * rng.random(size=n)
    x = r * np.cos(angles)
    y = 0.6 * r * np.sin(angles)
    pts = np.stack([x, y], axis=1)
    if d > 2:
        extra = rng.normal(scale=0.1, size=(n, d-2))
        pts = np.concatenate([pts, extra], axis=1)
    R = random_rotation_matrix(d, rng)
    shift = rng.normal(scale=3.0, size=d)
    return pts @ R + shift + rng.normal(scale=0.06, size=(n, d))

def make_fuzzy_patch(n, d, rng):
    """Small dense cloud."""
    center = rng.normal(scale=4.0, size=d)
    spread = 0.3 + 1.2 * rng.random()
    return rng.normal(scale=spread, size=(n, d)) + center

# --------------- Build 5 base clusters ----------------
N_BASE = 5
n_core = int(N_TOTAL_POINTS * (1 - OUTLIER_FRACTION))

# variable base sizes that sum to n_core
raw = rng.lognormal(mean=np.log(n_core / N_BASE) - 0.2, sigma=0.6, size=N_BASE)
sizes = np.maximum(220, (raw / raw.sum() * n_core).astype(int))
sizes[-1] += (n_core - sizes.sum())  # fix rounding to exact total

makers = [make_gaussian_blob, make_streak, make_half_moon_band, make_fuzzy_patch]
# Assign a maker to each base cluster (allow repeats)
assigned_makers = rng.choice(makers, size=N_BASE, replace=True)

X_list, y_base_list = [], []
for base_id in range(N_BASE):
    Xb = assigned_makers[base_id](sizes[base_id], N_DIM, rng)
    yb = np.full((sizes[base_id],), base_id)  # labels: 0..4 (base)
    X_list.append(Xb)
    y_base_list.append(yb)

# --------------- Optional uniform outliers ----------------
n_out = N_TOTAL_POINTS - int(np.sum(sizes))
if n_out > 0:
    all_stack = np.vstack(X_list)
    lo = all_stack.min(axis=0) - 1.0
    hi = all_stack.max(axis=0) + 1.0
    X_noise = rng.uniform(lo, hi, size=(n_out, N_DIM))
    y_noise_base = np.full((n_out,), -1)  # outliers keep base label -1
    X_list.append(X_noise)
    y_base_list.append(y_noise_base)

X = np.vstack(X_list)
y_base = np.concatenate(y_base_list)

# --------------- Merge to 3 final classes ----------------
# Map base labels {0..4} using MERGE_MAP; keep -1 for outliers if present.
y_final = np.array([MERGE_MAP.get(int(b), -1) for b in y_base])

# --------------- Scale + t-SNE ----------------
X = StandardScaler().fit_transform(X)
tsne = TSNE(
    n_components=2,
    perplexity=5,  # heuristic
    learning_rate="auto",
    init="pca",
    n_iter=1000,
    random_state=RNG_SEED
)
X2 = tsne.fit_transform(X)

# --------------- Report sizes ----------------
def counts(arr):
    uniq, cnt = np.unique(arr, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))

print("Base cluster sizes:", {int(k): int(v) for k, v in counts(y_base).items()})
print("Final class sizes:", {int(k): int(v) for k, v in counts(y_final).items()})

# --------------- Visualize ----------------
plt.figure(figsize=(7.6, 6.4))

# Plot final classes 0,1,2; then outliers on top
CC = ['AT', 'BT', 'MT']
for cls in [0, 1, 2]:
    idx = (y_final == cls)
    if np.any(idx):
        plt.scatter(X2[idx, 0], X2[idx, 1], s=10, alpha=0.9, label=CC[cls])

# Show outliers, if any
# if np.any(y_final == -1):
#     idx = (y_final == -1)
#     plt.scatter(X2[idx, 0], X2[idx, 1], s=8, marker="x", alpha=0.9, label="outliers")

# plt.title("Five base clusters merged into three classes — t-SNE view (no rings)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(loc="best", frameon=True)
plt.tight_layout()
plt.show()
