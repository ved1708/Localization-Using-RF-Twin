#!/usr/bin/env python3
"""
calibrate_rf3dgs.py
====================
Domain-gap compensation between RF-3DGS rendered Φ(P) and real / Sionna CSI.

Three methods, ordered by complexity and data requirement:

  Method A — Single-anchor gain calibration  (N=1 known pair)
             Φ_corrected(P) = γ ⊙ Φ_rendered(P)
             γ estimated from one reference measurement.

  Method B — N-anchor affine calibration     (N≥2 known pairs, recommended)
             Φ_corrected(P) = γ ⊙ Φ_rendered(P) + β
             γ, β fit by per-dimension closed-form linear regression.

  Method C — RBF residual field              (N≥3, best when gap is spatial)
             Φ_corrected(P) = Φ_rendered(P) + Σ_k w_k(P) · (Φ_obs_k - Φ_rendered(P_k))
             Inverse-distance-weighted interpolation of residuals — no model
             training, gracefully degrades to Method A/B with few anchors.

Calibration can be optionally composed with zero-shot fingerprint rescaling
(normalise rendered DB and observed feat to zero-mean/unit-variance) which
further reduces global bias without needing ANY known pairs.

Usage
-----
  # One-time calibration at N known positions (inside the docker container):
  python calibrate_rf3dgs.py \
      --model_path RF-3DGS/output/rf_model \
      --csi_data   dataset_csi_60ghz \
      --anchor_positions 1.0,2.5,1.2  3.5,2.5,1.2  6.0,1.0,1.2 \
      --method     affine \
      --out_dir    calibration_results

  # Then run localization with calibration applied:
  python localize_rf3dgs_realcsi.py \
      --model_path RF-3DGS/output/rf_model \
      --csi_data   dataset_csi_60ghz \
      --calibration calibration_results/calibration.npz

Integration into localize_rf3dgs_realcsi.py
--------------------------------------------
Add two lines to the "Evaluate RF-3DGS" loop:

  cal = CalibratedField.load("calibration_results/calibration.npz", field)
  # ... then replace:  obs_feat_raw = torch.from_numpy(phi_test_raw[i])
  #               with:  obs_feat_raw = cal.apply(torch.from_numpy(raw_phi_raw[i]))
  # and replace GD target:  rendered = field(p)
  #                    with:  rendered = cal.render(p)
"""

import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

# ── reuse Gaussian field from localize_rf3dgs_realcsi.py ────────────────────
# Import dynamically so this script works even without localize installed
try:
    from localize_rf3dgs_realcsi import GaussianField, build_database
except ImportError:
    # Inline minimal copy if the import fails (e.g. running standalone)
    from plyfile import PlyData

    class GaussianField(nn.Module):
        def __init__(self, ply_path: str, device: str = "cuda", topk: int = 10**9):
            super().__init__()
            self.device = device
            self.topk   = topk
            v = PlyData.read(ply_path)["vertex"]
            xyz     = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
            opacity = v["opacity"].astype(np.float32)
            scale   = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
            rot     = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
            f_dc    = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
            prop_names = [p.name for p in v.properties]
            n_rest  = sum(1 for n in prop_names if n.startswith("f_rest_"))
            f_rest  = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32) if n_rest > 0 else np.zeros((len(xyz), 0), np.float32)
            features = np.concatenate([f_dc, f_rest], axis=1)
            S_inv_sq = np.exp(-2.0 * scale.astype(np.float64)).clip(0, 1e7).astype(np.float32)
            R = self._quat_to_rotmat(rot)
            RS = R * S_inv_sq[:, None, :]
            Sigma_inv = RS @ R.transpose(0, 2, 1)
            self.register_buffer("xyz",       torch.from_numpy(xyz))
            self.register_buffer("opacity",   torch.from_numpy(opacity))
            self.register_buffer("features",  torch.from_numpy(features))
            self.register_buffer("Sigma_inv", torch.from_numpy(Sigma_inv.astype(np.float32)))
            self.to(device)

        @staticmethod
        def _quat_to_rotmat(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            n = np.sqrt(w*w + x*x + y*y + z*z).clip(1e-8)
            w, x, y, z = w/n, x/n, y/n, z/n
            return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                              2*(x*y+w*z),  1-2*(x*x+z*z), 2*(y*z-w*x),
                              2*(x*z-w*y),  2*(y*z+w*x),  1-2*(x*x+y*y)],
                             axis=1).reshape(-1, 3, 3)

        def forward(self, p):
            squeeze = (p.dim() == 1)
            if squeeze: p = p.unsqueeze(0)
            B = p.shape[0]
            d = p.unsqueeze(1) - self.xyz.unsqueeze(0)
            Sd = torch.einsum("bnij,bnj->bni", self.Sigma_inv.unsqueeze(0).expand(B, -1, -1, -1), d)
            maha = torch.nan_to_num((d * Sd).sum(-1), nan=100.0, posinf=100.0).clamp(max=30.0)
            w = torch.sigmoid(self.opacity).unsqueeze(0) * torch.exp(-0.5 * maha)
            w = torch.nan_to_num(w, nan=0.0)
            w_sum = w.sum(-1, keepdim=True).clamp(min=1e-8)
            feat = (w.unsqueeze(-1) * self.features.unsqueeze(0).expand(B, -1, -1)).sum(1) / w_sum
            feat = torch.nan_to_num(feat, nan=0.0)
            return feat.squeeze(0) if squeeze else feat

        @property
        def feature_dim(self): return self.features.shape[1]

    @torch.no_grad()
    def build_database(field, positions, batch_size=128):
        N, F = len(positions), field.feature_dim
        db   = torch.zeros(N, F, dtype=torch.float32)
        pos_t = torch.from_numpy(positions).to(field.device)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            db[s:e] = field(pos_t[s:e]).cpu()
        return db


# ════════════════════════════════════════════════════════════════════════════
#  Core calibration class
# ════════════════════════════════════════════════════════════════════════════

class DomainCalibration:
    """
    Encapsulates the domain-gap correction between rendered Φ and real obs.

    The correction maps:  Φ_obs  ←→  Φ_rendered
    so that gradient-descent minimises  ‖field(P) − corrected_obs‖²
    in the RENDERED domain — no changes to GaussianField needed.

    Equivalently (and numerically identical), one can correct on the
    rendered side:  Φ_corrected_render(P) = γ⁻¹ ⊙ (field(P) − β)
    See apply_to_rendered() for that path.
    """

    def __init__(self, gamma: np.ndarray, beta: np.ndarray,
                 method: str,
                 anchor_positions: Optional[np.ndarray] = None,
                 anchor_residuals: Optional[np.ndarray] = None,
                 phi_mean: Optional[np.ndarray] = None,
                 phi_std: Optional[np.ndarray] = None):
        """
        gamma, beta : (F,) affine correction  obs_corrected = (obs - beta) / gamma
        method      : 'gain' | 'affine' | 'rbf'
        anchor_positions : (K, 3) — only used by rbf method at inference
        anchor_residuals : (K, F) — Φ_obs_k - Φ_rendered(P_k), used by rbf
        phi_mean, phi_std: optional z-score normalisation (fit on rendered DB)
        """
        self.gamma            = gamma.astype(np.float32)
        self.beta             = beta.astype(np.float32)
        self.method           = method
        self.anchor_positions = anchor_positions
        self.anchor_residuals = anchor_residuals
        self.phi_mean         = phi_mean
        self.phi_std          = phi_std

    # ------------------------------------------------------------------
    # Observation-side correction  (recommended: correct the observed feat
    # into the rendered domain so GD loss landscape is unchanged)
    # ------------------------------------------------------------------

    def apply_to_observation(self, obs_feat: torch.Tensor) -> torch.Tensor:
        """
        Map a real/Sionna observed Φ_obs into the rendered Φ-space.
        Call this once per test sample before coarse search + GD.
        obs_feat : (F,) tensor on any device.
        Returns  : (F,) corrected tensor (same device).
        """
        device = obs_feat.device
        g = torch.from_numpy(self.gamma).to(device)
        b = torch.from_numpy(self.beta).to(device)
        # Affine de-bias:  Φ_in_rendered_space = (Φ_obs − β) / γ
        corrected = (obs_feat - b) / g.clamp(min=1e-8)

        if self.method == 'rbf' and self.anchor_positions is not None:
            # RBF correction uses position, so this path is only valid
            # AFTER a coarse position estimate is available.
            # Use apply_rbf_at_position() instead in that case.
            pass

        return corrected

    def apply_rbf_at_position(self,
                               obs_feat: torch.Tensor,
                               query_pos: np.ndarray) -> torch.Tensor:
        """
        RBF-corrected observation.  Call AFTER coarse position estimate.
        obs_feat  : (F,) raw observed feature
        query_pos : (3,) estimated position for spatial weighting
        """
        if self.anchor_positions is None or self.anchor_residuals is None:
            return self.apply_to_observation(obs_feat)

        device = obs_feat.device
        d2 = np.sum((self.anchor_positions - query_pos) ** 2, axis=1)  # (K,)
        # Inverse-distance weights (IDW):  w_k = 1/d²  (regularised by 1e-4)
        w  = 1.0 / (d2 + 1e-4)
        w  = w / w.sum()                                               # normalise

        # Weighted residual correction
        delta = (w[:, None] * self.anchor_residuals).sum(0)            # (F,)
        delta_t = torch.from_numpy(delta.astype(np.float32)).to(device)

        # First apply affine, then add RBF residual
        corrected = self.apply_to_observation(obs_feat) + delta_t
        return corrected

    # ------------------------------------------------------------------
    # Render-side correction  (alternative: correct the rendered output)
    # Used if you want to keep obs_feat unchanged and modify field(P)
    # ------------------------------------------------------------------

    def apply_to_rendered(self, rendered_feat: torch.Tensor) -> torch.Tensor:
        """
        Map rendered Φ(P) toward the observation domain.
        Φ_in_obs_space = γ ⊙ Φ(P) + β
        """
        device = rendered_feat.device
        g = torch.from_numpy(self.gamma).to(device)
        b = torch.from_numpy(self.beta).to(device)
        return g * rendered_feat + b

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = dict(gamma=self.gamma, beta=self.beta, method=self.method)
        if self.anchor_positions  is not None: data["anchor_positions"]  = self.anchor_positions
        if self.anchor_residuals  is not None: data["anchor_residuals"]  = self.anchor_residuals
        if self.phi_mean          is not None: data["phi_mean"]          = self.phi_mean
        if self.phi_std           is not None: data["phi_std"]           = self.phi_std
        np.savez_compressed(path, **data)
        print(f"  Calibration saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DomainCalibration":
        d = np.load(path, allow_pickle=True)
        return cls(
            gamma            = d["gamma"],
            beta             = d["beta"],
            method           = str(d["method"]),
            anchor_positions = d["anchor_positions"] if "anchor_positions" in d else None,
            anchor_residuals = d["anchor_residuals"] if "anchor_residuals" in d else None,
            phi_mean         = d["phi_mean"]         if "phi_mean"         in d else None,
            phi_std          = d["phi_std"]          if "phi_std"          in d else None,
        )


# ════════════════════════════════════════════════════════════════════════════
#  Calibration fitting routines
# ════════════════════════════════════════════════════════════════════════════

def calibrate_single_anchor(phi_rendered_cal: np.ndarray,
                             phi_obs_cal: np.ndarray) -> DomainCalibration:
    """
    Method A — Single anchor: estimate per-channel gain γ only, β=0.

    Physics intuition:
      The Gaussian field amplitude may be off by a constant multiplier
      (different training normalization, material mismatch).
      γ_c = Φ_obs_c / Φ_rendered_c   (per channel c)

    Parameters
    ----------
    phi_rendered_cal : (F,) rendered feature at known calibration position
    phi_obs_cal      : (F,) real/Sionna observed feature at same position

    Returns DomainCalibration with method='gain'.
    """
    gamma = phi_obs_cal / np.where(np.abs(phi_rendered_cal) > 1e-8,
                                    phi_rendered_cal,
                                    np.sign(phi_rendered_cal) * 1e-8)
    gamma = np.clip(gamma, 0.01, 100.0)   # sanity clamp
    beta  = np.zeros_like(gamma)
    print(f"\n  [Calibration] Method A — Single-anchor gain")
    print(f"  γ : min={gamma.min():.4f}  mean={gamma.mean():.4f}  max={gamma.max():.4f}")
    return DomainCalibration(gamma, beta, method='gain')


def calibrate_affine(phi_rendered_anchors: np.ndarray,
                     phi_obs_anchors: np.ndarray) -> DomainCalibration:
    """
    Method B — N-anchor affine calibration: fit γ, β per channel by OLS.

    Model: Φ_obs = γ ⊙ Φ_rendered + β  (each channel independently)
    Solved as: [γ_c, β_c] = argmin ‖γ_c·x_c + β_c − y_c‖²
                           = least-squares of [Φ_rendered_c | 1] → Φ_obs_c

    Parameters
    ----------
    phi_rendered_anchors : (K, F)
    phi_obs_anchors      : (K, F)

    Returns DomainCalibration with method='affine'.
    """
    K, F = phi_rendered_anchors.shape
    gamma = np.ones(F,  dtype=np.float32)
    beta  = np.zeros(F, dtype=np.float32)

    # Stack design matrix:  A = [Φ_rendered | 1]  shape (K, 2)
    ones = np.ones((K, 1), dtype=np.float32)
    A    = np.hstack([phi_rendered_anchors, ones])  # (K, F+1) — wrong dimensions!
    # Correct: solve per feature dimension c:
    #   [γ_c, β_c]  =  lstsq( [Φ_rendered[:, c], 1], Φ_obs[:, c] )
    A2 = np.concatenate([phi_rendered_anchors[:, :, None],
                         np.ones((K, F, 1))], axis=2).transpose(1, 0, 2)  # (F, K, 2)
    for c in range(F):
        coef, _, _, _ = np.linalg.lstsq(A2[c], phi_obs_anchors[:, c], rcond=None)
        gamma[c] = coef[0]
        beta[c]  = coef[1]

    gamma = np.clip(gamma, 0.01, 100.0)

    r2_list = []
    for c in range(F):
        pred = gamma[c] * phi_rendered_anchors[:, c] + beta[c]
        var_res = np.var(phi_obs_anchors[:, c] - pred)
        var_tot = np.var(phi_obs_anchors[:, c]).clip(1e-12)
        r2_list.append(1 - var_res / var_tot)
    r2 = np.array(r2_list)

    print(f"\n  [Calibration] Method B — Affine ({K} anchors)")
    print(f"  γ : min={gamma.min():.4f}  mean={gamma.mean():.4f}  max={gamma.max():.4f}")
    print(f"  β : min={beta.min():.4f}   mean={beta.mean():.4f}   max={beta.max():.4f}")
    print(f"  R²: min={r2.min():.3f}     mean={r2.mean():.3f}     max={r2.max():.3f}")

    return DomainCalibration(gamma, beta, method='affine')


def calibrate_rbf(phi_rendered_anchors: np.ndarray,
                  phi_obs_anchors: np.ndarray,
                  anchor_positions: np.ndarray) -> DomainCalibration:
    """
    Method C — RBF residual field.

    Stores per-anchor residuals:  δ_k = Φ_obs_k − Φ_rendered(P_k)
    At query position P:  Φ_corrected_obs(P) ≈ Φ_obs + Σ w_k(P) δ_k
    where w_k = IDW weights.

    Also fits a global affine correction (Method B) as a fallback for
    positions far from all anchors.

    Parameters
    ----------
    phi_rendered_anchors : (K, F)
    phi_obs_anchors      : (K, F)
    anchor_positions     : (K, 3)

    Returns DomainCalibration with method='rbf'.
    """
    residuals = phi_obs_anchors - phi_rendered_anchors   # (K, F) Φ_obs − Φ_rendered
    affine_cal = calibrate_affine(phi_rendered_anchors, phi_obs_anchors)

    print(f"\n  [Calibration] Method C — RBF residual field ({len(anchor_positions)} anchors)")
    res_norm = np.linalg.norm(residuals, axis=1)
    print(f"  Residual ‖δ_k‖: min={res_norm.min():.4f}  "
          f"mean={res_norm.mean():.4f}  max={res_norm.max():.4f}")

    return DomainCalibration(
        gamma            = affine_cal.gamma,
        beta             = affine_cal.beta,
        method           = 'rbf',
        anchor_positions = anchor_positions.astype(np.float32),
        anchor_residuals = residuals.astype(np.float32),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Zero-shot normalisation (no anchor positions required)
# ════════════════════════════════════════════════════════════════════════════

def zeroshot_normalization(db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std of the rendered fingerprint DB.
    These stats can z-score both the DB and the observed feature to remove
    global bias without any known anchor positions.

    Returns (phi_mean, phi_std) each (F,).
    """
    phi_mean = db.mean(axis=0, keepdims=False)
    phi_std  = db.std(axis=0, keepdims=False).clip(1e-8)
    return phi_mean, phi_std


# ════════════════════════════════════════════════════════════════════════════
#  CSI → Φ linear projection fitting
# ════════════════════════════════════════════════════════════════════════════

def fit_linear_projection(csi_ri: np.ndarray,
                          db: np.ndarray,
                          train_idx: np.ndarray,
                          reg: float = 1e-3) -> np.ndarray:
    """
    Fit a linear projection  W : (csi_dim,) → (F,)  that maps raw CSI
    features into the rendered Φ space using ridge regression.

    Model:  Φ_rendered ≈ CSI · W
    Solved: W = (X'X + λI)⁻¹ X'Y
      where X = csi_ri[train_idx]  shape (N_train, csi_dim)
            Y = db[train_idx]      shape (N_train, F)

    This projection bridges the CSI observation space and the Gaussian-field
    feature space so that real/Sionna CSI can be compared against rendered Φ.

    Parameters
    ----------
    csi_ri    : (N, csi_dim) raw CSI real-imag features
    db        : (N, F)      rendered Φ fingerprint database
    train_idx : (N_train,)  indices of training positions
    reg       : ridge regularisation λ (default 1e-3)

    Returns W of shape (csi_dim, F).
    """
    X = csi_ri[train_idx].astype(np.float64)    # (N_train, csi_dim)
    Y = db[train_idx].astype(np.float64)        # (N_train, F)

    csi_dim = X.shape[1]
    A = X.T @ X + reg * np.eye(csi_dim)        # (csi_dim, csi_dim)
    B = X.T @ Y                                 # (csi_dim, F)
    W = np.linalg.solve(A, B)                  # (csi_dim, F)

    # Evaluate fit quality on training set
    Y_pred  = X @ W
    ss_res  = np.sum((Y - Y_pred) ** 2, axis=0)
    ss_tot  = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0).clip(1e-12)
    r2      = 1 - ss_res / ss_tot

    print(f"\n  [Linear Projection] CSI({csi_dim}) → Φ({Y.shape[1]})  "
          f"(ridge λ={reg}, N_train={len(train_idx)})")
    print(f"  R²: min={r2.min():.3f}  mean={r2.mean():.3f}  max={r2.max():.3f}")
    return W.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Integration helpers for localization scripts
# ════════════════════════════════════════════════════════════════════════════

class CalibratedGradientField:
    """
    Drop-in replacement for GaussianField in the GD loop.
    Wraps field(P) → apply_to_rendered(field(P)) so that the GD
    minimises  ‖corrected_render(P) − obs‖² with obs kept as-is.

    Usage:
        cal    = DomainCalibration.load("calibration.npz")
        cfield = CalibratedGradientField(field, cal)
        # Replace  rendered = field(p)
        # With     rendered = cfield(p)
        loss = ((rendered - obs_feat_raw) ** 2).sum()
    """
    def __init__(self, field: GaussianField, cal: DomainCalibration):
        self.field = field
        self.cal   = cal

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        raw = self.field(p)                       # (F,) or (B, F)
        return self.cal.apply_to_rendered(raw)    # same shape


# ════════════════════════════════════════════════════════════════════════════
#  Patch to gradient_descent_localize from localize_rf3dgs_realcsi.py
# ════════════════════════════════════════════════════════════════════════════

def gradient_descent_localize_calibrated(
        field:    GaussianField,
        cal:      DomainCalibration,
        obs_feat: torch.Tensor,          # raw (un-corrected) observed feature
        init_pos: np.ndarray,
        bounds:   np.ndarray,
        n_steps:  int   = 100,
        lr:       float = 0.05,
        use_rbf:  bool  = False,
):
    """
    Gradient descent localization with domain calibration.

    Two equivalent formulations:

      (1) Correct observation → minimize ‖field(P) − corrected_obs‖²
      (2) Correct render      → minimize ‖cal.apply_to_rendered(field(P)) − obs‖²

    This function uses formulation (1): correct obs once, keep field(P) untouched.
    This is preferred because the GD landscape matches what the field was trained on.

    Returns (position, loss_history).
    """
    # Apply affine correction to observation once
    if use_rbf:
        obs_corrected = cal.apply_rbf_at_position(obs_feat, init_pos)
    else:
        obs_corrected = cal.apply_to_observation(obs_feat)

    obs_corrected = obs_corrected.to(field.device)
    p0 = np.clip(init_pos.copy(), bounds[:, 0], bounds[:, 1])
    p  = torch.tensor(p0, dtype=torch.float32, device=field.device, requires_grad=True)

    optimizer    = optim.Adam([p], lr=lr)
    loss_history = []

    for step in range(n_steps):
        optimizer.zero_grad()
        rendered = field(p)
        loss     = ((rendered - obs_corrected) ** 2).sum()
        loss.backward()
        nn.utils.clip_grad_norm_([p], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            p.data.clamp_(
                torch.tensor(bounds[:, 0], device=field.device),
                torch.tensor(bounds[:, 1], device=field.device))
        loss_history.append(loss.item())

    return p.detach().cpu().numpy(), loss_history


# ════════════════════════════════════════════════════════════════════════════
#  CLI — standalone calibration script
# ════════════════════════════════════════════════════════════════════════════

def _load_csi_dataset(data_path: str):
    proc_path = os.path.join(data_path, "processed_features.npz")
    norm_path = os.path.join(data_path, "normalized_dataset.npz")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"CSI dataset not found: {proc_path}")
    proc      = np.load(proc_path)
    norm_data = np.load(norm_path)
    csi_ri    = proc["features_real_imag"].astype(np.float32)   # (N, 128)
    positions = proc["positions"].astype(np.float32)             # (N, 3)
    train_idx = norm_data["train_indices"]
    return csi_ri, positions, train_idx


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate RF-3DGS Φ-space to real / Sionna CSI observations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",      default="RF-3DGS/output/rf_model")
    parser.add_argument("--iteration",       type=int, default=-1,
                        help="PLY iteration (-1 = latest)")
    parser.add_argument("--csi_data",        default="dataset_csi_60ghz",
                        help="CSI dataset with processed_features.npz")
    parser.add_argument("--anchor_positions", nargs="+", default=None,
                        help="N calibration positions as 'x,y,z' strings. "
                             "If omitted, N random training positions are used.")
    parser.add_argument("--n_random_anchors", type=int, default=5,
                        help="Number of random training anchors if not specified explicitly")
    parser.add_argument("--method",          choices=["gain", "affine", "rbf"],
                        default="affine")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--out_dir",         default="calibration_results")
    parser.add_argument("--db_cache",        default=None,
                        help="Path to fingerprint_db.npz from a previous run "
                             "(avoids recomputing the full DB)")
    parser.add_argument("--proj_reg",         type=float, default=1e-3,
                        help="Ridge regularisation λ for CSI→Φ projection fitting")
    parser.add_argument("--refit_projection", action="store_true",
                        help="Re-fit CSI→Φ projection even if linear_proj.npy exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Find PLY ─────────────────────────────────────────────────────────
    if args.iteration == -1:
        pc_dir = os.path.join(args.model_path, "point_cloud")
        iters  = [int(d.split("_")[1]) for d in os.listdir(pc_dir)
                  if d.startswith("iteration_")]
        args.iteration = max(iters)

    ply_path = os.path.join(args.model_path, "point_cloud",
                            f"iteration_{args.iteration}", "point_cloud.ply")
    print(f"\n  PLY : {ply_path}")

    # ── Load CSI dataset ─────────────────────────────────────────────────
    csi_ri, positions, train_idx = _load_csi_dataset(args.csi_data)
    print(f"  CSI dataset: {len(positions)} positions, {csi_ri.shape[1]} CSI dims")

    # ── Parse anchor positions ────────────────────────────────────────────
    if args.anchor_positions is not None:
        anchor_pos = np.array(
            [[float(v) for v in s.split(",")] for s in args.anchor_positions],
            dtype=np.float32,
        )
    else:
        # Random subset of training positions
        rng = np.random.default_rng(42)
        idx = rng.choice(train_idx, size=min(args.n_random_anchors, len(train_idx)),
                         replace=False)
        anchor_pos = positions[idx]
        print(f"  Using {len(anchor_pos)} random training positions as anchors")

    print(f"  Anchor positions ({len(anchor_pos)}):")
    for i, p in enumerate(anchor_pos):
        print(f"    [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # ── Load Gaussian field ───────────────────────────────────────────────
    field = GaussianField(ply_path, device=args.device)
    field.eval()
    for param in field.parameters():
        param.requires_grad = False

    # ── Build or load Φ fingerprint DB ────────────────────────────────────
    db_cache = args.db_cache or os.path.join(args.out_dir, "fingerprint_db.npz")
    if os.path.exists(db_cache):
        print(f"\n  Loading cached DB: {db_cache}")
        cache  = np.load(db_cache)
        db     = torch.from_numpy(cache["db"])
        db_pos = cache["db_pos"]
    else:
        print(f"\n  Building fingerprint DB at {len(positions)} positions…")
        t0     = time.time()
        db     = build_database(field, positions)
        db_pos = positions
        np.savez_compressed(db_cache, db=db.numpy(), db_pos=db_pos)
        print(f"  DB built in {time.time()-t0:.1f}s → {db_cache}")

    # ── Obtain rendered Φ at anchor positions ───────────────────────────────
    anchor_pos_t = torch.from_numpy(anchor_pos).to(args.device)
    with torch.no_grad():
        phi_rendered_anchors = field(anchor_pos_t).cpu().numpy()   # (K, F)

    # ── Fit or load CSI → Φ linear projection ────────────────────────────────
    # This projection W maps raw CSI features (128-dim) into the rendered Φ
    # space (48-dim) so that real/Sionna observations are comparable to field(P).
    proj_path = os.path.join(args.out_dir, "linear_proj.npy")
    if os.path.exists(proj_path) and not args.refit_projection:
        print(f"\n  Loading existing CSI→Φ projection: {proj_path}")
        localize_proj = np.load(proj_path)   # (csi_dim, F)
    else:
        localize_proj = fit_linear_projection(
            csi_ri, db.numpy(), train_idx, reg=args.proj_reg)
        np.save(proj_path, localize_proj)
        print(f"  Projection saved → {proj_path}")

    # ── Get "observed" Φ at anchor positions via projection ───────────────────
    # For each anchor, find nearest dataset position and project its CSI → Φ.
    phi_obs_anchors = np.zeros_like(phi_rendered_anchors)
    for i, apos in enumerate(anchor_pos):
        d2     = np.sum((positions - apos) ** 2, axis=1)
        nn_idx = int(d2.argmin())
        phi_obs_anchors[i] = csi_ri[nn_idx] @ localize_proj

    F = phi_rendered_anchors.shape[1]
    print(f"\n  Φ dimension         : {F}")
    print(f"  Rendered range      : [{phi_rendered_anchors.min():.4f}, "
          f"{phi_rendered_anchors.max():.4f}]")
    print(f"  Observed range      : [{phi_obs_anchors.min():.4f}, "
          f"{phi_obs_anchors.max():.4f}]")

    # ── Fit calibration ───────────────────────────────────────────────────
    if args.method == 'gain':
        # Average gain across all anchors
        gammas = phi_obs_anchors / np.where(np.abs(phi_rendered_anchors) > 1e-8,
                                             phi_rendered_anchors,
                                             1e-8)
        gamma  = np.clip(gammas.mean(axis=0), 0.01, 100.0)
        beta   = np.zeros(F, dtype=np.float32)
        cal    = DomainCalibration(gamma, beta, method='gain')
        print(f"  γ mean={gamma.mean():.4f}  std={gamma.std():.4f}")

    elif args.method == 'affine':
        cal = calibrate_affine(phi_rendered_anchors, phi_obs_anchors)

    else:  # rbf
        cal = calibrate_rbf(phi_rendered_anchors, phi_obs_anchors, anchor_pos)

    # Attach zero-shot normalisation stats (optional, from full DB)
    phi_mean, phi_std = zeroshot_normalization(db.numpy())
    cal.phi_mean = phi_mean
    cal.phi_std  = phi_std

    # ── Save ──────────────────────────────────────────────────────────────
    cal_path = os.path.join(args.out_dir, "calibration.npz")
    cal.save(cal_path)

    # ── Quick self-consistency test ───────────────────────────────────────
    print("\n  Self-consistency check on anchor positions:")
    errs = []
    for i, apos in enumerate(anchor_pos):
        phi_r = phi_rendered_anchors[i]
        phi_o = phi_obs_anchors[i]
        # Correct obs → rendered domain
        phi_r_t   = torch.from_numpy(phi_r)
        phi_o_t   = torch.from_numpy(phi_o)
        corrected = cal.apply_to_observation(phi_o_t)
        err_before = float(torch.norm(phi_r_t - phi_o_t).item())
        err_after  = float(torch.norm(phi_r_t - corrected).item())
        print(f"    Anchor {i}: ‖Φ_r − Φ_o‖ = {err_before:.4f}  →  after cal: {err_after:.4f}")
        errs.append(err_after)
    print(f"  Mean residual ‖Φ_r − corrected_obs‖ = {np.mean(errs):.4f}")
    print(f"\n  Done. Calibration saved to {cal_path}")
    print(f"""
  ── Calibration outputs in {args.out_dir}/ ─────────────────────────────
    fingerprint_db.npz  — rendered Φ at all {len(positions)} dataset positions
    linear_proj.npy     — CSI({csi_ri.shape[1]}) → Φ({F}) projection matrix W
    calibration.npz     — domain-gap correction (γ, β, method={args.method})

  ── How to use in your localization script ────────────────────────────

    from calibrate_rf3dgs import (
        GaussianField, build_database,
        DomainCalibration, gradient_descent_localize_calibrated
    )

    # Load artefacts
    field   = GaussianField(ply_path, device="cuda")
    db      = np.load("{args.out_dir}/fingerprint_db.npz")["db"]       # (N, F)
    db_pos  = np.load("{args.out_dir}/fingerprint_db.npz")["db_pos"]   # (N, 3)
    W       = np.load("{args.out_dir}/linear_proj.npy")                 # (128, F)
    cal     = DomainCalibration.load("{args.out_dir}/calibration.npz")

    # For each test CSI observation:
    csi_feat    = ...                             # (128,) raw CSI real-imag
    obs_phi     = torch.from_numpy(csi_feat @ W)  # project to Φ space
    obs_cor     = cal.apply_to_observation(obs_phi)  # domain correction

    # Coarse nearest-neighbour search in DB
    db_t    = torch.from_numpy(db)
    dists   = ((db_t - obs_cor.unsqueeze(0)) ** 2).sum(1)
    coarse  = db_pos[dists.argmin().item()]

    # Fine gradient-descent refinement
    bounds  = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    fine, _ = gradient_descent_localize_calibrated(
                  field, cal, obs_phi, coarse, bounds,
                  n_steps=200, lr=0.02, use_rbf=(cal.method == 'rbf'))
    print("Estimated position:", fine)
  ──────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
