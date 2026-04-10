"""
inspect_rf_dataset.py — Visual + quantitative evaluation of generated RF datasets.

Usage (inside Docker):
    python3 inspect_rf_dataset.py                       # compare mvdr vs ideal, save figures
    python3 inspect_rf_dataset.py --dataset dataset_mvdr_M4
    python3 inspect_rf_dataset.py --dataset dataset_mvdr_M4 dataset_ideal_mpc --compare
    python3 inspect_rf_dataset.py --dataset dataset_mvdr_M4 --anim  # position sweep animation

Outputs (saved as PNG, no display needed):
    eval_grid.png          — 4×3 mosaic: 4 positions, 3 orientations
    eval_compare.png       — side-by-side MVDR vs ideal (if --compare)
    eval_sharpness.png     — sharpness & discriminability metrics
    eval_sweep.png         — spectrum change as RX sweeps along X axis
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive: safe inside Docker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from glob import glob
from tqdm import tqdm

# ── helpers ───────────────────────────────────────────────────────────────────

def load_images(spectrum_dir, indices=None):
    """Load PNG images as float32 arrays (H,W,3) in [0,1]."""
    paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
    if indices is not None:
        paths = [paths[i] for i in indices if i < len(paths)]
    imgs = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        imgs.append(img)
    return imgs, [os.path.basename(p) for p in (sorted(glob(os.path.join(spectrum_dir, "*.png"))) if indices is None else [sorted(glob(os.path.join(spectrum_dir, "*.png")))[i] for i in indices if i < len(sorted(glob(os.path.join(spectrum_dir, "*.png"))))])]

def sharpness(img):
    """Laplacian-based sharpness — higher = sharper blobs."""
    gray = img.mean(axis=2)
    lap = (
        np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4 * gray
    )
    return float(np.var(lap))

def discriminability(imgs):
    """
    Mean pairwise L2 distance between flattened images (subset for speed).
    Higher → positions are more distinguishable → better for localization.
    """
    flat = np.stack([i.mean(axis=2).ravel() for i in imgs])   # grayscale
    n = min(len(flat), 30)
    flat = flat[:n]
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(flat[i] - flat[j]))
    return float(np.mean(dists)) if dists else 0.0

def read_positions_from_images_txt(images_file):
    """Parse COLMAP images.txt to get tvec (world position) per image."""
    positions = []
    with open(images_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 8:
            try:
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                positions.append((tx, ty, tz))
            except ValueError:
                pass
        i += 1  # empty POINTS2D lines already filtered by 'if l.strip()'
    return positions


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_grid(spectrum_dir, out_path, n_pos=4, n_ori=3, title=None):
    """
    Mosaic: rows = positions (evenly spaced), cols = orientations.
    n_ori images per position are consecutive (as generated).
    """
    paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
    total = len(paths)
    if total == 0:
        print(f"  No images found in {spectrum_dir}"); return

    # Pick evenly spaced position indices
    positions_available = total // n_ori
    pos_indices = np.linspace(0, positions_available - 1, min(n_pos, positions_available), dtype=int)

    fig, axes = plt.subplots(n_pos, n_ori, figsize=(n_ori * 3.5, n_pos * 3.5))
    if n_pos == 1: axes = axes[np.newaxis, :]
    if n_ori == 1: axes = axes[:, np.newaxis]

    ori_labels = [f"Ori {i+1} ({int(360*i/n_ori)}°)" for i in range(n_ori)]

    for row, pos_idx in enumerate(pos_indices):
        for col in range(n_ori):
            img_idx = pos_idx * n_ori + col
            img = np.array(Image.open(paths[img_idx]).convert('RGB'))
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(ori_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Pos {pos_idx+1}/{positions_available}", fontsize=8)

    fig.suptitle(title or os.path.basename(spectrum_dir), fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_compare(dir_a, dir_b, out_path, n_examples=5, label_a="A", label_b="B"):
    """
    Side-by-side comparison: pick n evenly spaced positions, show same index from both datasets.
    """
    paths_a = sorted(glob(os.path.join(dir_a, "*.png")))
    paths_b = sorted(glob(os.path.join(dir_b, "*.png")))
    n = min(n_examples, len(paths_a), len(paths_b))
    indices = np.linspace(0, min(len(paths_a), len(paths_b)) - 1, n, dtype=int)

    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))

    for col, idx in enumerate(indices):
        img_a = np.array(Image.open(paths_a[idx]).convert('RGB'))
        img_b = np.array(Image.open(paths_b[idx]).convert('RGB'))
        axes[0, col].imshow(img_a); axes[0, col].axis('off')
        axes[1, col].imshow(img_b); axes[1, col].axis('off')
        axes[0, col].set_title(f"#{idx+1}", fontsize=8)

    axes[0, 0].set_ylabel(label_a, fontsize=10, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel(label_b, fontsize=10, rotation=0, labelpad=50)
    plt.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_sharpness(datasets, out_path):
    """
    Bar chart: mean sharpness and discriminability per dataset.
    Sharpness  → blob edge crispness (MVDR should be sharper than MPC)
    Discrim.   → mean pairwise distance (localizability indicator)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    names, sharpnesses, discrim = [], [], []
    for label, spectrum_dir in datasets.items():
        paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
        if not paths: continue
        # Sample at most 40 images for speed
        sample = [paths[i] for i in np.linspace(0, len(paths)-1, min(40, len(paths)), dtype=int)]
        imgs = [np.array(Image.open(p).convert('RGB'), dtype=np.float32)/255 for p in sample]
        s_vals = [sharpness(im) for im in imgs]
        names.append(label)
        sharpnesses.append(np.mean(s_vals))
        discrim.append(discriminability(imgs))
        print(f"  {label}: sharpness={np.mean(s_vals):.4f}, discriminability={discrim[-1]:.4f}")

    x = np.arange(len(names))
    ax1.bar(x, sharpnesses, color=['steelblue', 'coral', 'seagreen'][:len(names)])
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_title("Sharpness (Laplacian variance)\nHigher = crisper blobs", fontsize=9)
    ax1.set_ylabel("Laplacian var")

    ax2.bar(x, discrim, color=['steelblue', 'coral', 'seagreen'][:len(names)])
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_title("Discriminability (mean pairwise L2)\nHigher = easier to localize", fontsize=9)
    ax2.set_ylabel("Mean L2 dist")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_sweep(spectrum_dir, images_file, out_path, n_ori=3, orientation_idx=0):
    """
    Show spectrum images as RX sweeps — sorted by X position.
    Uses COLMAP images.txt tvec to order by world X coordinate.
    """
    paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
    if not paths or not os.path.exists(images_file):
        print(f"  Missing files for sweep plot"); return

    positions = read_positions_from_images_txt(images_file)
    if len(positions) != len(paths):
        print(f"  Position count mismatch ({len(positions)} vs {len(paths)}), skipping sweep")
        return

    # Group by position (every n_ori consecutive images share a position)
    pos_groups = [(positions[i*n_ori], paths[i*n_ori + orientation_idx])
                  for i in range(len(paths) // n_ori)]

    # Sort by X coordinate
    pos_groups.sort(key=lambda x: x[0][0])

    n_show = min(10, len(pos_groups))
    indices = np.linspace(0, len(pos_groups)-1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.5, 3))
    if n_show == 1: axes = [axes]

    for col, idx in enumerate(indices):
        pos, img_path = pos_groups[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        axes[col].imshow(img)
        axes[col].axis('off')
        axes[col].set_title(f"X={pos[0]:.1f}\nY={pos[1]:.1f}", fontsize=7)

    plt.suptitle(f"Position Sweep (orientation {orientation_idx+1}) — {os.path.basename(spectrum_dir)}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_spectrum_stats(spectrum_dir, out_path):
    """
    Per-image statistics: mean brightness and std dev over all images.
    Flat line = no variation across positions (bad — means dataset is useless).
    Variable line = spectrum encodes position information (good).
    """
    paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
    if not paths: return

    means, stds, maxvals = [], [], []
    for p in tqdm(paths, desc="  Computing stats", leave=False):
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        gray = img.mean(axis=2)
        means.append(gray.mean())
        stds.append(gray.std())
        maxvals.append(gray.max())

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(paths))
    axes[0].plot(x, means, lw=0.8, color='steelblue')
    axes[0].set_ylabel("Mean brightness"); axes[0].set_title("Per-image statistics (flat line = no spatial variation = bad)")
    axes[1].plot(x, stds, lw=0.8, color='coral')
    axes[1].set_ylabel("Std dev (contrast)")
    axes[2].plot(x, maxvals, lw=0.8, color='seagreen')
    axes[2].set_ylabel("Max value"); axes[2].set_xlabel("Image index")

    # Mark orientation boundaries (every 3 images = new position)
    for ax in axes:
        for i in range(0, len(paths), 3):
            ax.axvline(i, color='gray', lw=0.3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect and evaluate RF datasets.")
    parser.add_argument("--dataset", nargs='+',
                        default=["dataset_mvdr_M4"],
                        help="Dataset directory(ies) to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side comparison of first two datasets")
    parser.add_argument("--outdir", default="eval_output",
                        help="Where to save evaluation figures (default: eval_output/)")
    parser.add_argument("--n-ori", type=int, default=3,
                        help="Number of orientations per position (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for ds in args.dataset:
        spectrum_dir = os.path.join(ds, "spectrum")
        images_file  = os.path.join(ds, "images.txt")
        tag = os.path.basename(ds.rstrip('/'))

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        paths = sorted(glob(os.path.join(spectrum_dir, "*.png")))
        print(f"  Images: {len(paths)} ({len(paths)//args.n_ori} positions × {args.n_ori} orientations)")
        if not paths:
            print("  WARNING: No images found!"); continue

        # 1. Visual grid
        print("  [1/4] Generating visual grid...")
        plot_grid(spectrum_dir,
                  out_path=os.path.join(args.outdir, f"eval_grid_{tag}.png"),
                  n_pos=min(4, len(paths)//args.n_ori), n_ori=args.n_ori,
                  title=tag)

        # 2. Position sweep
        print("  [2/4] Generating position sweep...")
        plot_sweep(spectrum_dir, images_file,
                   out_path=os.path.join(args.outdir, f"eval_sweep_{tag}.png"),
                   n_ori=args.n_ori)

        # 3. Per-image stats
        print("  [3/4] Computing per-image statistics...")
        plot_spectrum_stats(spectrum_dir,
                            out_path=os.path.join(args.outdir, f"eval_stats_{tag}.png"))

    # 4. Sharpness / discriminability across all datasets
    datasets_dict = {os.path.basename(d.rstrip('/')): os.path.join(d, "spectrum")
                     for d in args.dataset
                     if os.path.isdir(os.path.join(d, "spectrum"))}
    if datasets_dict:
        print(f"\n  [4/4] Computing sharpness & discriminability...")
        plot_sharpness(datasets_dict,
                       out_path=os.path.join(args.outdir, "eval_sharpness.png"))

    # 5. Side-by-side comparison
    if args.compare and len(args.dataset) >= 2:
        print(f"\n  [5] Generating comparison...")
        plot_compare(
            os.path.join(args.dataset[0], "spectrum"),
            os.path.join(args.dataset[1], "spectrum"),
            out_path=os.path.join(args.outdir, "eval_compare.png"),
            label_a=os.path.basename(args.dataset[0]),
            label_b=os.path.basename(args.dataset[1])
        )

    print(f"\nAll figures saved to: {args.outdir}/")
    print("\nWhat to look for:")
    print("  eval_grid_*     → Blobs should shift position as RX moves; all 3 orientations should differ")
    print("  eval_sweep_*    → Spectrum should change smoothly — not random noise, not identical")
    print("  eval_stats_*    → Std dev should vary (not flat) → dataset has spatial information")
    print("  eval_sharpness  → MVDR sharpness > MPC sharpness; discriminability > 0.05")
    print("  eval_compare    → MVDR blobs should be narrower than MPC blobs")


if __name__ == "__main__":
    main()
