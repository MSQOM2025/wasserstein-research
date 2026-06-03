import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from scipy.ndimage import rotate, shift


# ==========================================
# Section 1: Statistical Core Engine
# ==========================================
class WDCorResearch:
    def __init__(self, n_projections=500, seed=42):
        self.k = n_projections
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _distance_covariance(self, D):
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        centered_D = H @ D @ H
        return np.sqrt(np.sum(centered_D * centered_D) / (n**2))

    def compute_dcor(self, Dx, Dy):
        n = Dx.shape[0]

        A = Dx - Dx.mean(0, keepdims=True) - Dx.mean(1, keepdims=True) + Dx.mean()
        B = Dy - Dy.mean(0, keepdims=True) - Dy.mean(1, keepdims=True) + Dy.mean()

        v2_xy = np.sum(A * B) / (n**2)
        v2_xx = self._distance_covariance(Dx)**2
        v2_yy = self._distance_covariance(Dy)**2

        denom = np.sqrt(v2_xx * v2_yy)
        return np.sqrt(np.maximum(0, v2_xy)) / np.sqrt(denom) if denom > 1e-10 else 0.0

    def fast_sliced_wasserstein_matrix(self, X_batch):
        """
        Compute approximate pairwise distances using Sliced Wasserstein metric.
        """
        n, m, d = X_batch.shape

        rng = np.random.default_rng(self.seed)
        thetas = rng.standard_normal(size=(self.k, d))
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)

        projected = np.matmul(X_batch, thetas.T)
        projected.sort(axis=1)

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            diff = np.abs(projected[i:i+1] - projected)
            dist_matrix[i] = np.mean(np.sum(diff, axis=1), axis=1)

        return dist_matrix


# ==========================================
# Section 2: Image Geometry Processing
# ==========================================
def img_to_cloud(img, rng, centering=True, scale=1.0, jitter=True):
    coords = []
    r_idx, c_idx = np.where(img > 2)

    for r, c in zip(r_idx, c_idx):
        intensity = int(img[r, c])
        for _ in range(intensity):
            rn = r + (rng.uniform(-0.4, 0.4) if jitter else 0)
            cn = c + (rng.uniform(-0.4, 0.4) if jitter else 0)
            coords.append([rn * scale, cn * scale])

    coords = np.array(coords, dtype=float)

    if len(coords) < 10:
        return np.zeros((100, 2))

    if centering:
        coords -= np.mean(coords, axis=0)

    idx = rng.choice(len(coords), 100, replace=len(coords) < 100)
    return coords[idx]


def cloud_to_pixels(cloud, bins=8):
    hist, _, _ = np.histogram2d(
        cloud[:, 0], cloud[:, 1],
        bins=bins,
        range=[[-5, 13], [-5, 13]]
    )
    return hist.flatten()


# ==========================================
# Section 3: Experimental Pipeline
# ==========================================
def run_reproducible_study():
    SEED = 42
    np.random.seed(SEED)

    print("Running UCI Digits experimental pipeline...")

    digits, _ = load_digits(return_X_y=True)
    n_samples = 150
    X_raw = digits[:n_samples].reshape(-1, 8, 8)

    model = WDCorResearch(n_projections=500, seed=SEED)

    scenarios = ['Rigid Rotation', 'Random Shift', 'Geometric Expansion']
    results = {'Euclidean': [], 'W-dCor': []}

    for scenario in scenarios:
        X_pixels, Y_pixels = [], []
        X_clouds, Y_clouds = [], []

        rng = np.random.default_rng(SEED)

        for i in range(n_samples):
            x_cloud = img_to_cloud(X_raw[i], rng)
            X_clouds.append(x_cloud)
            X_pixels.append(X_raw[i].flatten())

            if scenario == 'Rigid Rotation':
                y_img = rotate(X_raw[i], angle=45, reshape=False, order=1)
                y_cloud = img_to_cloud(y_img, rng)
                Y_pixels.append(y_img.flatten())
                Y_clouds.append(y_cloud)

            elif scenario == 'Random Shift':
                sr, sc = rng.uniform(-3, 3, 2)
                y_img = shift(X_raw[i], shift=(sr, sc), order=1)
                y_cloud = img_to_cloud(y_img, rng)
                Y_pixels.append(y_img.flatten())
                Y_clouds.append(y_cloud)

            elif scenario == 'Geometric Expansion':
                factor = 1.0 + (np.sum(X_raw[i]) / 50.0)
                y_cloud = x_cloud * factor
                Y_pixels.append(cloud_to_pixels(y_cloud))
                Y_clouds.append(y_cloud)

        X_c, Y_c = np.array(X_clouds), np.array(Y_clouds)
        X_p, Y_p = np.array(X_pixels), np.array(Y_pixels)

        res_e = model.compute_dcor(
            pairwise_distances(X_p),
            pairwise_distances(Y_p)
        )

        res_w = model.compute_dcor(
            model.fast_sliced_wasserstein_matrix(X_c),
            model.fast_sliced_wasserstein_matrix(Y_c)
        )

        results['Euclidean'].append(res_e)
        results['W-dCor'].append(res_w)

        print(f"{scenario}: Euclidean={res_e:.3f}, W-dCor={res_w:.3f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    x = np.arange(len(scenarios))
    width = 0.35

    plt.bar(x - width/2, results['Euclidean'], width, label='Euclidean')
    plt.bar(x + width/2, results['W-dCor'], width, label='W-dCor')

    plt.xticks(x, scenarios)
    plt.xlabel("Geometric Transformations")
    plt.ylabel("Distance Correlation")
    plt.legend()

    plt.ylim(0, 1.1)
    plt.savefig("results.png", dpi=300)

    print("Saved plot: results.png")


if __name__ == "__main__":
    run_reproducible_study()
