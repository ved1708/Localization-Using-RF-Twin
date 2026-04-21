import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation 
from scipy.ndimage import map_coordinates
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera as SionnaCamera, RadioMaterial
from sionna.rt.antenna import tr38901_pattern
import sionna
import sionna.channel

# --- GPU Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# --- COLMAP Structures ---
class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

class colmap_Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

# --- Geometric Helper Functions ---
def euler_to_quaternion(euler):
    # rotation from colmap camera default(initial direction) to sionna array initial direction
    R_posz2posx = Rotation.from_euler('ZYX', [-np.pi/2,0.0,-np.pi/2])
    # rotation of sionna from array initial direction to sampling direction
    yaw, pitch, roll = euler 
    R_posx2array = Rotation.from_euler('ZYX',[yaw, pitch, roll]) 
    # For intrinsic rotations, the rightmost rotation matrix corresponds to the first rotation applied.
    R_w2c =  R_posx2array * R_posz2posx
    R_c2w = R_w2c.inv()
    q = R_c2w.as_quat()
    # colmap requires qw qx qy qz scalar first quaternion
    qvec_c2w = [q[3],q[0],q[1],q[2]] 
    return R_c2w, qvec_c2w

def calculate_camera_intrinsics(width, height, focal_length):
    fx = focal_length
    fy = focal_length
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

def save_intrinsics_text(path, cameras):
    with open(path, "w") as fid:
        for cam_id, cam in cameras.items():
            params_str = " ".join(map(str, cam.params))
            fid.write(f"{cam_id} {cam.model} {cam.width} {cam.height} {params_str}\n")

def save_extrinsics_text(path, images):
    with open(path, "w") as fid:
        for img_id, img in images.items():
            qvec_str = " ".join(map(str, img.qvec))
            tvec_str = " ".join(map(str, img.tvec))
            fid.write(f"{img_id} {qvec_str} {tvec_str} {img.camera_id} {img.name}\n")
            fid.write("\n")

# --- Image Transform Helpers ---
def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]
    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)
    return out

class Equirectangular:
    def __init__(self, img):
        self._img = img
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        
        # Scipy Rotation replacement for cv2.Rodrigues
        r1 = Rotation.from_rotvec(y_axis * np.radians(THETA))
        R1 = r1.as_matrix()
        
        r2 = Rotation.from_rotvec(np.dot(R1, x_axis) * np.radians(PHI))
        R2 = r2.as_matrix()
        
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        
        # Scipy map_coordinates replacement for cv2.remap
        # XY is (H, W, 2) where XY[..., 0] is X (col), XY[..., 1] is Y (row)
        map_cols = XY[..., 0] # indices for width
        map_rows = XY[..., 1] # indices for height
        
        # Stack coordinates for map_coordinates: (2, H, W) -> [rows, cols]
        coords = np.stack([map_rows, map_cols], axis=0)
        
        out_channels = []
        for i in range(self._img.shape[2]):
            # map_coordinates handles interpolation. mode='wrap' behaves like BORDER_WRAP?
            # cv2.BORDER_WRAP means cyclic. scipy 'wrap' means cyclic.
            out_channels.append(map_coordinates(self._img[..., i], coords, order=1, mode='wrap'))
            
        persp = np.stack(out_channels, axis=-1)
        return persp

def equirectangular_to_perspective(img, fov, theta, phi, height, width):
    eq = Equirectangular(img)
    return eq.GetPerspective(FOV=fov, THETA=-theta, PHI=90-phi, height=height, width=width)

# --- Core MPC Spectrum Generation ---
def plot_spatial_spectrum(path_instance, image_scale=3, kernel_size=3, kernel_sigma=3):
    theta_r = path_instance.theta_r.numpy()  
    phi_r = path_instance.phi_r.numpy()      
    intensities = path_instance.a.numpy()  
    # print(f"intensities shape: {intensities.shape}, Intensities sample: {intensities}")
    
    # print(f"theta_r shape: {theta_r.shape}, phi_r shape: {phi_r.shape}, intensities shape: {intensities.shape}")
    
    # Check if empty (no paths)
    if intensities.size == 0:
        return np.ones((180*image_scale, 360*image_scale)) * -160.0 # Return noise floor

    img = np.zeros((360*image_scale, 180*image_scale))
 
    sigma_x, sigma_y = kernel_sigma, kernel_sigma
    theta = theta_r[0, 0, 0, :]*180/np.pi
    phi = phi_r[0, 0, 0, :]*180/np.pi
    amps = np.abs(intensities[0, 0, 0, 0, 0, :,0])
    # print(f"Amplitudes shape: {amps.shape}, Amplitudes sample: {amps}")
 
    size_x = int(kernel_size * sigma_x) | 1  
    size_y = int(kernel_size * sigma_y) | 1  
    x = np.linspace(-size_x // 2, size_x // 2, size_x)
    y = np.linspace(-size_y // 2, size_y // 2, size_y)
    x, y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    for idx, intensity in enumerate(amps):
        if intensity > 1e-9: # Threshold for meaningful paths
            path_dot = gauss_kernel*intensity/np.sum(gauss_kernel) 
            phi_idx = int(-phi[idx] + 180)*image_scale
            theta_idx = int(theta[idx])*image_scale
            xmin = max(0, phi_idx - size_x // 2)
            xmax = min(360*image_scale, phi_idx + size_x // 2 + 1)
            ymin = max(0, theta_idx - size_y // 2)
            ymax = min(180*image_scale, theta_idx + size_y // 2 + 1)
 
            gauss_xmin = max(0, size_x // 2 - phi_idx)
            gauss_xmax = min(size_x, 360*image_scale - phi_idx + size_x // 2)
            gauss_ymin = max(0, size_y // 2 - theta_idx)
            gauss_ymax = min(size_y, 180*image_scale - theta_idx + size_y // 2)
 
            # Ensure indices match
            if (xmax > xmin) and (ymax > ymin):
                target_slice = img[xmin:xmax, ymin:ymax]
                source_slice = path_dot[gauss_xmin:gauss_xmax, gauss_ymin:gauss_ymax]
                
                # Double check shapes match before adding
                if target_slice.shape == source_slice.shape:
                    img[xmin:xmax, ymin:ymax] += source_slice

    non_zero_mask = img != 0.0
    zero_mask = img == 0.0

    img[non_zero_mask] = 10*np.log10(img[non_zero_mask]) 
    img[zero_mask] = np.min(img[non_zero_mask])-10 if np.any(non_zero_mask) else -160
    
    return img

def plot_aod_spatial_spectrum(path_instance, image_scale=3, kernel_size=3, kernel_sigma=3):
    """
    AoD (Angle of Departure) spatial spectrum — RGB encoding of departure angles:
      R = departure elevation (theta_t, 0°–180°, low=blue, high=red)
      G = departure azimuth  (phi_t, -180°–180°, mapped to [1,0])
      B = path amplitude (acts as reference channel)
    Returns shape: (360*scale, 180*scale, 3), values in dB.
    """
    # Blob position = arrival direction (same as MPC)
    theta_r = path_instance.theta_r.numpy()[0, 0, 0, :] * 180 / np.pi  # arrival elevation [deg]
    phi_r   = path_instance.phi_r.numpy()[0, 0, 0, :]   * 180 / np.pi  # arrival azimuth   [deg]
    # Color encoding = departure direction (what makes AoD different from MPC)
    theta_t = path_instance.theta_t.numpy()[0, 0, 0, :] * 180 / np.pi  # departure elevation [deg]
    phi_t   = path_instance.phi_t.numpy()[0, 0, 0, :]   * 180 / np.pi  # departure azimuth   [deg]
    amps    = np.abs(path_instance.a.numpy()[0, 0, 0, 0, 0, :, 0])      # path amplitudes

    if amps.size == 0:
        return np.ones((180 * image_scale, 360 * image_scale, 3)) * -160.0

    img_rgb = np.zeros((360 * image_scale, 180 * image_scale, 3))

    sigma_x, sigma_y = kernel_sigma, kernel_sigma
    size_x = int(kernel_size * sigma_x) | 1
    size_y = int(kernel_size * sigma_y) | 1
    x = np.linspace(-size_x // 2, size_x // 2, size_x)
    y = np.linspace(-size_y // 2, size_y // 2, size_y)
    x, y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    for idx, amp in enumerate(amps):
        if amp < 1e-9:
            continue

        # Blob color encodes departure angles (unique per TX-scatterer-RX geometry → discriminative for localization)
        r_value = np.interp(theta_t[idx], (0, 180),    (0, 1))  # departure elevation → R
        g_value = np.interp(phi_t[idx],   (-180, 180),  (1, 0)) # departure azimuth   → G

        path_dot = gauss_kernel * amp / np.sum(gauss_kernel)

        # Blob position = arrival direction (same coordinate system as MPC)
        phi_idx   = int(-phi_r[idx]   + 180) * image_scale
        theta_idx = int( theta_r[idx]       ) * image_scale

        xmin = max(0, phi_idx   - size_x // 2)
        xmax = min(360 * image_scale, phi_idx   + size_x // 2 + 1)
        ymin = max(0, theta_idx - size_y // 2)
        ymax = min(180 * image_scale, theta_idx + size_y // 2 + 1)

        gauss_xmin = max(0, size_x // 2 - phi_idx)
        gauss_xmax = min(size_x, 360 * image_scale - phi_idx + size_x // 2)
        gauss_ymin = max(0, size_y // 2 - theta_idx)
        gauss_ymax = min(size_y, 180 * image_scale - theta_idx + size_y // 2)

        if (xmax > xmin) and (ymax > ymin):
            s = path_dot[gauss_xmin:gauss_xmax, gauss_ymin:gauss_ymax]
            if s.shape == (xmax - xmin, ymax - ymin):
                img_rgb[xmin:xmax, ymin:ymax, 0] += s * r_value  # R = elevation-weighted amp
                img_rgb[xmin:xmax, ymin:ymax, 1] += s * g_value  # G = azimuth-weighted amp
                img_rgb[xmin:xmax, ymin:ymax, 2] += s            # B = amplitude reference

    # dB conversion per channel (same convention as plot_spatial_spectrum)
    for c in range(3):
        ch = img_rgb[:, :, c]
        nz = ch > 0
        if np.any(nz):
            ch[nz] = 10 * np.log10(ch[nz])
            ch[~nz] = np.min(ch[nz]) - 10
        else:
            ch[:] = -160.0
        img_rgb[:, :, c] = ch

    return img_rgb  # shape: (360*scale, 180*scale, 3)


def plot_delay_spatial_spectrum(path_instance, image_scale=4, kernel_size=3, kernel_sigma=3, delay_max_ns=200):
    """
    Propagation-delay spatial spectrum plotted at arrival direction — RGB encoding:
      R = propagation delay (tau, 0-delay_max_ns ns, normalized to [0,1])
      G = path amplitude (reference channel)
      B = path amplitude (same as G — duplicated for visual balance)
    Returns shape: (360*scale, 180*scale, 3), values in dB.
    """
    theta_r = path_instance.theta_r.numpy()[0, 0, 0, :] * 180 / np.pi  # arrival elevation [deg]
    phi_r   = path_instance.phi_r.numpy()[0, 0, 0, :]   * 180 / np.pi  # arrival azimuth   [deg]
    amps    = np.abs(path_instance.a.numpy()[0, 0, 0, 0, 0, :, 0])      # path amplitudes
    tau_ns  = path_instance.tau.numpy()[0, 0, 0, :] * 1e9               # seconds → nanoseconds

    if amps.size == 0:
        return np.ones((180 * image_scale, 360 * image_scale, 3)) * -160.0

    img_rgb = np.zeros((360 * image_scale, 180 * image_scale, 3))

    sigma_x, sigma_y = kernel_sigma, kernel_sigma
    size_x = int(kernel_size * sigma_x) | 1
    size_y = int(kernel_size * sigma_y) | 1
    x = np.linspace(-size_x // 2, size_x // 2, size_x)
    y = np.linspace(-size_y // 2, size_y // 2, size_y)
    x, y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    for idx, amp in enumerate(amps):
        if amp < 1e-9:
            continue

        # Normalize delay: 0 ns → R=0, delay_max_ns → R=1
        r_value = np.interp(tau_ns[idx], (0, delay_max_ns), (0, 1))

        path_dot = gauss_kernel * amp / np.sum(gauss_kernel)

        phi_idx   = int(-phi_r[idx]   + 180) * image_scale
        theta_idx = int( theta_r[idx]       ) * image_scale

        xmin = max(0, phi_idx   - size_x // 2)
        xmax = min(360 * image_scale, phi_idx   + size_x // 2 + 1)
        ymin = max(0, theta_idx - size_y // 2)
        ymax = min(180 * image_scale, theta_idx + size_y // 2 + 1)

        gauss_xmin = max(0, size_x // 2 - phi_idx)
        gauss_xmax = min(size_x, 360 * image_scale - phi_idx + size_x // 2)
        gauss_ymin = max(0, size_y // 2 - theta_idx)
        gauss_ymax = min(size_y, 180 * image_scale - theta_idx + size_y // 2)

        if (xmax > xmin) and (ymax > ymin):
            s = path_dot[gauss_xmin:gauss_xmax, gauss_ymin:gauss_ymax]
            if s.shape == (xmax - xmin, ymax - ymin):
                img_rgb[xmin:xmax, ymin:ymax, 0] += s * r_value  # R = delay-encoded amp
                img_rgb[xmin:xmax, ymin:ymax, 1] += s            # G = amplitude reference
                img_rgb[xmin:xmax, ymin:ymax, 2] += s            # B = amplitude reference

    # dB conversion per channel
    for c in range(3):
        ch = img_rgb[:, :, c]
        nz = ch > 0
        if np.any(nz):
            ch[nz] = 10 * np.log10(ch[nz])
            ch[~nz] = np.min(ch[nz]) - 10
        else:
            ch[:] = -160.0
        img_rgb[:, :, c] = ch

    return img_rgb  # shape: (360*scale, 180*scale, 3)


def plot_phase_spectrum(path_instance, image_scale=3, kernel_size=3, kernel_sigma=3):
    """
    Phase-aware spatial spectrum plotted at arrival direction — RGB encoding:
      R = amplitude * cos(phase)
      G = amplitude * sin(phase)
      B = amplitude

        Notes:
            - Returns linear channels; normalization is handled downstream in generate_ideal_dataset.
            - R/G are signed (phase-carrying), B is non-negative amplitude.
    Returns shape: (360*scale, 180*scale, 3).
    """
    theta_r = path_instance.theta_r.numpy()[0, 0, 0, :] * 180 / np.pi
    phi_r = path_instance.phi_r.numpy()[0, 0, 0, :] * 180 / np.pi
    a_complex = path_instance.a.numpy()[0, 0, 0, 0, 0, :, 0]

    if a_complex.size == 0:
        return np.ones((180 * image_scale, 360 * image_scale, 3)) * -160.0

    amp = np.abs(a_complex)
    phase = np.angle(a_complex)

    img_rgb = np.zeros((360 * image_scale, 180 * image_scale, 3), dtype=np.float32)

    sigma_x, sigma_y = kernel_sigma, kernel_sigma
    size_x = int(kernel_size * sigma_x) | 1
    size_y = int(kernel_size * sigma_y) | 1
    x = np.linspace(-size_x // 2, size_x // 2, size_x)
    y = np.linspace(-size_y // 2, size_y // 2, size_y)
    x, y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

    for idx, a_val in enumerate(amp):
        if a_val < 1e-9:
            continue

        r_value = np.cos(phase[idx])
        g_value = np.sin(phase[idx])

        path_dot = gauss_kernel * a_val / np.sum(gauss_kernel)

        phi_idx = int(-phi_r[idx] + 180) * image_scale
        theta_idx = int(theta_r[idx]) * image_scale

        xmin = max(0, phi_idx - size_x // 2)
        xmax = min(360 * image_scale, phi_idx + size_x // 2 + 1)
        ymin = max(0, theta_idx - size_y // 2)
        ymax = min(180 * image_scale, theta_idx + size_y // 2 + 1)

        gauss_xmin = max(0, size_x // 2 - phi_idx)
        gauss_xmax = min(size_x, 360 * image_scale - phi_idx + size_x // 2)
        gauss_ymin = max(0, size_y // 2 - theta_idx)
        gauss_ymax = min(size_y, 180 * image_scale - theta_idx + size_y // 2)

        if (xmax > xmin) and (ymax > ymin):
            s = path_dot[gauss_xmin:gauss_xmax, gauss_ymin:gauss_ymax]
            if s.shape == (xmax - xmin, ymax - ymin):
                img_rgb[xmin:xmax, ymin:ymax, 0] += s * r_value
                img_rgb[xmin:xmax, ymin:ymax, 1] += s * g_value
                img_rgb[xmin:xmax, ymin:ymax, 2] += s

    return img_rgb


def normalize_phase_spectrum(spec_rgb, rg_percentile=99.0, b_lo_percentile=10.0, b_hi_percentile=99.0,
                             rg_scale=None, b_lo_db=None, b_hi_db=None):
    """
    Normalize phase spectrum for visualization to [0,1] RGB.
      - R/G: symmetric normalization around zero, then mapped to [0,1] via 0.5 offset.
      - B: log-compressed amplitude (dB) with robust percentile clipping.
    """
    r = spec_rgb[:, :, 0]
    g = spec_rgb[:, :, 1]
    b = spec_rgb[:, :, 2]

    if rg_scale is None:
        rg_abs = np.concatenate([np.abs(r).ravel(), np.abs(g).ravel()])
        rg_abs_nz = rg_abs[rg_abs > 0]
        if rg_abs_nz.size > 0:
            rg_scale = np.percentile(rg_abs_nz, rg_percentile)
        else:
            rg_scale = 1.0
    rg_scale = max(float(rg_scale), 1e-12)

    r_norm = 0.5 + 0.5 * np.clip(r / rg_scale, -1.0, 1.0)
    g_norm = 0.5 + 0.5 * np.clip(g / rg_scale, -1.0, 1.0)

    b_db = 10.0 * np.log10(np.maximum(b, 1e-30))
    if b_lo_db is None or b_hi_db is None:
        b_finite = b_db[np.isfinite(b_db)]
        if b_finite.size > 0:
            b_lo_db = np.percentile(b_finite, b_lo_percentile)
            b_hi_db = np.percentile(b_finite, b_hi_percentile)
        else:
            b_lo_db, b_hi_db = -120.0, -40.0
    b_lo_db = float(b_lo_db)
    b_hi_db = float(b_hi_db)
    if b_hi_db <= b_lo_db:
        b_hi_db = b_lo_db + 1.0
    b_norm = np.clip((b_db - b_lo_db) / (b_hi_db - b_lo_db + 1e-9), 0.0, 1.0)

    return np.stack([r_norm, g_norm, b_norm], axis=-1).astype(np.float32)


# =============================================================================
# MVDR Beamforming Dataset Functions
# =============================================================================

def merge_paths_to_time_grid(a, tau, time_grid):
    """Merge multi-path complex amplitudes onto a fixed time grid."""
    a = tf.convert_to_tensor(a, dtype=tf.complex64)
    tau = tf.convert_to_tensor(tau, dtype=time_grid.dtype)
    time_grid_tensor = tf.constant(time_grid, dtype=tau.dtype)
    if tf.rank(tau) == 0:
        tau = tf.expand_dims(tau, axis=0)
    indices = tf.searchsorted(time_grid_tensor, tau, side='left') - 1
    indices = tf.clip_by_value(indices, 0, len(time_grid) - 1)
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    if tf.rank(indices) == 0:
        indices = tf.expand_dims(indices, axis=0)
    result = tf.tensor_scatter_nd_add(
        tf.zeros(len(time_grid), dtype=tf.complex64),
        tf.expand_dims(indices, axis=1), a)
    return result


def paths_to_response(paths, time_interval=0.1):
    """
    Convert Sionna paths to a baseband time-domain array response matrix.
    Returns: complex tensor of shape (M², L) where L = number of time steps.
    Requires synthetic_array=True so paths.cir() returns per-element CIRs.
    """
    paths.normalize_delays = False
    bb_a, tau = paths.cir()         # bb_a: [1, 1, M², 1, 1, num_paths, 1]
    t = tf.squeeze(tau) / 1e-9      # delays in nanoseconds
    max_spread = int(np.ceil(float(np.max(t.numpy()))))
    if max_spread == 0:
        max_spread = 1
    time_grid = np.arange(0, max_spread, time_interval).astype(np.float32)
    if len(time_grid) == 0:
        time_grid = np.array([0.0], dtype=np.float32)

    a_np = bb_a.numpy()         # [1, 1, M², 1, 1, num_paths, 1]
    num_elements = bb_a.shape[2]
    a_out = np.zeros((num_elements, len(time_grid)), dtype=np.complex64)
    for i in range(num_elements):
        a_temp = a_np[0, 0, i, 0, 0, :, 0]
        a_out[i] = merge_paths_to_time_grid(a_temp, t, time_grid).numpy()
    return tf.convert_to_tensor(a_out)   # (M², L)


def array_manifold_vector(M, theta_grid, phi_grid):
    """
    Compute MxM UPA array manifold vector with tr38901 element pattern.
    Returns: complex tensor of shape (M², H, W) for scanning over (theta, phi) grid.
    Element positions follow the tutorial's λ/2 spacing convention.
    """
    theta_grid = tf.cast(theta_grid, tf.float32)
    phi_grid   = tf.cast(phi_grid,   tf.float32)

    v = tf.sin(theta_grid) * tf.sin(phi_grid)
    w = tf.cos(theta_grid)
    v_grid = tf.expand_dims(v, 0)   # (1, H, W)
    w_grid = tf.expand_dims(w, 0)

    values   = 0.25 + 0.5 * np.arange(M / 2)
    y_i = np.concatenate((-values[::-1], values))
    z_i = np.concatenate((values[::-1], -values))
    y_i_grid, z_i_grid = np.meshgrid(y_i, z_i)
    y_i_flat = np.reshape(y_i_grid.T, (M**2,))
    z_i_flat = np.reshape(z_i_grid.T, (M**2,))
    y_i_tf = tf.cast(tf.reshape(y_i_flat, (M**2, 1, 1)), tf.float32)
    z_i_tf = tf.cast(tf.reshape(z_i_flat, (M**2, 1, 1)), tf.float32)

    phase   = 2 * np.pi * (y_i_tf * v_grid + z_i_tf * w_grid)
    c_phase = tf.complex(tf.zeros_like(phase), phase)

    # Apply tr38901 element radiation pattern
    c_theta, _ = tr38901_pattern(theta_grid, phi_grid, slant_angle=0.0, polarization_model=2)
    c_G_grid  = tf.expand_dims(c_theta, 0)   # (1, H, W)
    alpha_i   = c_G_grid * tf.exp(c_phase)   # (M², H, W)
    return alpha_i


def compute_angle_matrices_mvdr(width, height, fov):
    """
    Map each pixel of a PINHOLE camera image to (theta, phi) in camera-local spherical coords.
    theta: zenith [0, π], phi: azimuth [-π, π].
    Used by array_manifold_vector to steer MVDR across the image FOV.
    """
    focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    x = np.linspace(-width  / 2, width  / 2, width)
    y = np.linspace( height / 2, -height / 2, height)
    xx, yy = np.meshgrid(x, y)
    directions = np.stack([xx / focal, yy / focal, np.ones_like(xx)], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    phi   =  np.arctan2(-directions[..., 0],  directions[..., 2])  # azimuth
    theta =  np.pi / 2 - np.arcsin(directions[..., 1])             # zenith
    return theta, phi


def MVDR_spectrum(paths, M, theta_grid, phi_grid, time_interval=0.1):
    """
    Minimum Variance Distortionless Response (MVDR) beamforming spectrum.
    R = x xH (spatial covariance matrix from all paths)
    P_mvdr(θ,φ) = 1 / (aH R⁻¹ a)
    Returns: (p_mvdr, p_mvdr_dB) both shape (H, W).
    REQUIREMENT: rank(R) = M² → need num_paths >> M².
    """
    x = paths_to_response(paths, time_interval)  # (M², L)
    x_H = tf.transpose(tf.math.conj(x))          # (L, M²)
    R   = tf.matmul(x, x_H)                      # (M², M²)
    # Adaptive diagonal loading: only activates when R has negative eigenvalues
    # (i.e. truly rank-deficient, num_paths < M²). When R is full-rank (normal case
    # with 20+ paths), eps=0 → no loading → physical nulls fully preserved.
    # Fixed loading (e.g. 1e-3×trace) fills nulls even for well-conditioned R.
    eigvals  = tf.math.real(tf.linalg.eigvalsh(R))          # sorted ascending
    min_eig  = tf.reduce_min(eigvals)
    max_eig  = tf.reduce_max(eigvals)
    # Load just enough to make min eigenvalue positive — zero loading if already PD
    eps = tf.maximum(-min_eig + 1e-10 * max_eig, 0.0)
    eps = tf.cast(eps, tf.complex64)
    R_loaded = R + eps * tf.eye(tf.shape(R)[0], dtype=tf.complex64)
    R_inv = tf.linalg.inv(R_loaded)               # (M², M²)
    R_inv = tf.expand_dims(tf.expand_dims(R_inv, -1), -1)  # (M², M², 1, 1)

    a_scan   = array_manifold_vector(M, theta_grid, phi_grid)          # (M², H, W)
    a_scan_H = tf.math.conj(tf.expand_dims(a_scan, axis=1))            # (M², 1, H, W)

    aH_Rinv   = tf.reduce_sum(a_scan_H * R_inv, axis=0)                # (M², H, W)
    aH_Rinv_a = tf.reduce_sum(aH_Rinv * a_scan, axis=0)               # (H, W)

    p_mvdr    = 1.0 / tf.abs(aH_Rinv_a)                               # (H, W)
    p_mvdr_dB = 10.0 * np.log10(p_mvdr.numpy() + 1e-30)
    return p_mvdr, p_mvdr_dB


def CBF_spectrum(paths, M, theta_grid, phi_grid, time_interval=0.1):
    """
    Angle-Delay CBF (delay-resolved beamforming)

    P(θ,φ) = max_l |aᴴ x_l|²

    where x_l is the array snapshot at delay bin l.
    """

    # array response over delays
    x = paths_to_response(paths, time_interval)   # (M², L)

    # steering vectors
    a_scan = array_manifold_vector(M, theta_grid, phi_grid)   # (M², H, W)
    aH = tf.math.conj(tf.expand_dims(a_scan, axis=1))         # (M²,1,H,W)

    # expand x for broadcasting
    x_exp = tf.expand_dims(tf.expand_dims(x, -1), -1)         # (M²,L,1,1)

    # beamform per delay tap
    bf = tf.reduce_sum(aH * x_exp, axis=0)                    # (L,H,W)

    power = tf.abs(bf)**2                                     # (L,H,W)

    # collapse delay dimension
    p_cbf = tf.reduce_max(power, axis=0)                      # (H,W)

    p_cbf_dB = 10*np.log10(p_cbf.numpy() + 1e-30)

    return p_cbf, p_cbf_dB


def generate_mvdr_dataset(scene, rx_locs, tx_loc, output_dir, M=4, time_interval=0.1, method='mvdr'):
    """
    Array beamforming dataset for RF-3DGS.
    method='mvdr' — MVDR (Capon): P = 1/(aᴴR⁻¹a), sharp adaptive nulls, needs num_paths >> M²
    method='cbf'  — CBF (Bartlett): P = aᴴRa, always stable, fully deterministic
    Requires: M×M tr38901 rx_array with synthetic_array=True.
    """
    spectrum_dir = os.path.join(output_dir, 'spectrum')
    cameras_file = os.path.join(output_dir, 'cameras.txt')
    images_file  = os.path.join(output_dir, 'images.txt')
    os.makedirs(spectrum_dir, exist_ok=True)

    if 'tx' in scene.transmitters: scene.remove('tx')
    scene.add(Transmitter(name='tx', position=tx_loc))

    width, height = 600, 600
    h_fov_deg = 120
    h_fov_rad = np.deg2rad(h_fov_deg)
    focal_length = width / (2 * np.tan(h_fov_rad / 2))
    fx, fy, cx, cy = calculate_camera_intrinsics(width, height, focal_length)
    camera_id = 1
    cameras = {camera_id: Camera(camera_id, "PINHOLE", width, height, [fx, fy, cx, cy])}
    images   = {}

    print(f"{method.upper()} Setup: {width}×{height}, H-FOV={h_fov_deg}°, M={M} ({M}×{M} array = {M**2} elements)")

    jet_colormap = plt.get_cmap('jet')
    # Camera-local angle grid — same for every pose (array points along +x, same FOV)
    theta_grid, phi_grid = compute_angle_matrices_mvdr(width=width, height=height, fov=h_fov_deg)

    # ------------------------------------------------------------------
    # 1. Global normalization pass (sample a few positions × all orientations)
    # ------------------------------------------------------------------
    spectrum_fn = MVDR_spectrum if method == 'mvdr' else CBF_spectrum
    print(f"Computing global {method.upper()} dB range...")
    spec_max, spec_min = -np.inf, np.inf
    sample_indices = np.linspace(0, len(rx_locs)-1, min(4, len(rx_locs)), dtype=int)

    for s_idx in tqdm(sample_indices, desc=f"Sampling {method.upper()} stats"):
        for angle in [0, 2*np.pi/3, 4*np.pi/3]:
            if 'rx' in scene.receivers: scene.remove('rx')
            rx = Receiver(name='rx', position=rx_locs[s_idx],
                          look_at=None, orientation=tf.Variable([angle, 0, 0], dtype=tf.float32))
            scene.add(rx)
            paths = scene.compute_paths(max_depth=2, reflection=True, diffraction=False,
                                        scattering=True, scat_keep_prob=0.5, num_samples=int(5e5))
            if paths.a.shape[-2] == 0:
                continue
            try:
                _, dB = spectrum_fn(paths, M=M, theta_grid=theta_grid, phi_grid=phi_grid,
                                    time_interval=time_interval)
                if np.max(dB) > spec_max: spec_max = np.max(dB)
                if np.min(dB) < spec_min: spec_min = np.min(dB)
            except Exception as e:
                print(f"  stats {method.upper()} failed: {e}")

    if spec_max == -np.inf:
        spec_max, spec_min = -40.0, -160.0
    print(f"Global MVDR range: [{spec_min:.1f}, {spec_max:.1f}] dB")

    # ------------------------------------------------------------------
    # 2. Generation pass
    # ------------------------------------------------------------------
    image_counter = 1
    for i, rx_loc in enumerate(tqdm(rx_locs, desc="Generating MVDR views", unit="pos")):
        for angle in tqdm([0, 2*np.pi/3, 4*np.pi/3],
                          desc=f"  Orientations {i+1}/{len(rx_locs)}", leave=False, unit="view"):
            if 'rx' in scene.receivers: scene.remove('rx')
            rx = Receiver(name='rx', position=rx_loc,
                          look_at=None, orientation=tf.Variable([angle, 0, 0], dtype=tf.float32))
            scene.add(rx)
            paths = scene.compute_paths(max_depth=2, reflection=True, diffraction=False,
                                        scattering=True, scat_keep_prob=0.5, num_samples=int(5e5))
            try:
                _, spec_dB = spectrum_fn(paths, M=M, theta_grid=theta_grid, phi_grid=phi_grid,
                                         time_interval=time_interval)
            except Exception as e:
                print(f"  {method.upper()} failed at pos {i}, angle {angle:.2f}: {e}")
                spec_dB = np.ones((height, width)) * spec_min

            spec_norm   = np.clip((spec_dB - spec_min) / (spec_max - spec_min + 1e-9), 0, 1)
            img_colored = np.clip(jet_colormap(spec_norm)[:, :, :3], 0, 1)

            img_filename = f"{image_counter:05d}.png"
            plt.imsave(os.path.join(spectrum_dir, img_filename), img_colored)

            orientation   = [angle, 0, 0]
            R_c2w, qvec_c2w = euler_to_quaternion(orientation)
            tvec_c2w = -R_c2w.apply(rx_loc)
            images[image_counter] = colmap_Image(image_counter, qvec_c2w, tvec_c2w,
                                                  camera_id, img_filename, [], [])
            image_counter += 1

    if 'rx' in scene.receivers: scene.remove('rx')
    save_intrinsics_text(cameras_file, cameras)
    save_extrinsics_text(images_file, images)
    print(f"{method.upper()} dataset generation complete.")


def generate_ideal_dataset(scene, rx_locs, tx_loc, output_dir, spectrum_type='mpc'):
    # spectrum_type: 'mpc'   → MPC arrival-angle spectrum  (single-channel grayscale)
    #               'aod'   → AoD departure-angle spectrum (3-channel RGB: R=elev, G=azim, B=amp)
    #               'delay' → Propagation-delay spectrum   (3-channel RGB: R=delay, G=B=amp)
    #               'phase' → Phase-aware spectrum          (3-channel RGB: R=Acos(phi), G=Asin(phi), B=A)
    assert spectrum_type in ('mpc', 'aod', 'delay', 'phase'), \
        f"spectrum_type must be 'mpc', 'aod', 'delay', or 'phase', got '{spectrum_type}'"

    spectrum_dir = os.path.join(output_dir, 'spectrum')
    cameras_file = os.path.join(output_dir, 'cameras.txt')
    images_file  = os.path.join(output_dir, 'images.txt')

    os.makedirs(spectrum_dir, exist_ok=True)
    
    if 'tx' in scene.transmitters: scene.remove('tx')
    scene.add(Transmitter(name='tx', position=tx_loc))

    width, height = 600, 600  # For 600×400: H-FOV=120, V-FOV≈98
    h_fov_deg = 120  # Horizontal FOV
    
    # Calculate vertical FOV based on aspect ratio to maintain square pixels
    aspect_ratio = width / height
    h_fov_rad = np.deg2rad(h_fov_deg)
    v_fov_rad = 2 * np.arctan(np.tan(h_fov_rad / 2) / aspect_ratio)
    v_fov_deg = np.rad2deg(v_fov_rad)
    
    print(f"Camera Setup: {width}x{height}, H-FOV={h_fov_deg}°, V-FOV={v_fov_deg:.1f}°, spectrum_type='{spectrum_type}'")
    
    # Calculate focal length from horizontal FOV
    focal_length = width / (2 * np.tan(h_fov_rad / 2)) 
    fx, fy, cx, cy = calculate_camera_intrinsics(width, height, focal_length)
    print(f"Focal length: f={focal_length:.1f}px")
    camera_id = 1
    cameras = {camera_id: Camera(camera_id, "PINHOLE", width, height, [fx, fy, cx, cy])}
    images = {}
    
    # Add Receiver ONCE
    if 'rx' in scene.receivers: scene.remove('rx')
    rx = Receiver(name='rx', position=[0,0,0])
    scene.add(rx)

    # 1. Determine normalization stats.
    spec_max = None
    spec_min = None
    phase_norm_params = None
    if spectrum_type != 'phase':
        print("Computing global stats (Min/Max dB)...")
        spec_max = -np.inf
        spec_min = np.inf

        # Sample a subset
        sample_indices = np.linspace(0, len(rx_locs)-1, min(50, len(rx_locs)), dtype=int)

        for idx in tqdm(sample_indices, desc="Sampling stats", unit="pos"):
            rx.position = rx_locs[idx]

            # Scat_keep_prob=0.5 for detailed MPC
            # Using 2 depth to capture reflections
            paths = scene.compute_paths(max_depth=2, reflection=True, diffraction=False, scattering=True,
                                        scat_keep_prob=0.5, num_samples=int(5e5))

            # Check if paths found
            if paths.a.shape[-2] == 0:
                continue

            # Dispatch to correct spectrum function
            if spectrum_type == 'aod':
                spec = np.transpose(plot_aod_spatial_spectrum(paths), (1, 0, 2))
            elif spectrum_type == 'delay':
                spec = np.transpose(plot_delay_spatial_spectrum(paths), (1, 0, 2))
            else:  # 'mpc'
                spec = plot_spatial_spectrum(paths).T
            if np.max(spec) > spec_max: spec_max = np.max(spec)
            if np.min(spec) < spec_min: spec_min = np.min(spec)

        # Fallback if no paths found in sample
        if spec_max == -np.inf:
            spec_max = -40
            spec_min = -160

        print(f"Global Max dB: {spec_max}, Global Min dB: {spec_min}")
    else:
        print("Phase mode: computing dataset-level normalization (stable mapping across all training images).")
        sample_indices = np.linspace(0, len(rx_locs)-1, min(50, len(rx_locs)), dtype=int)
        rng = np.random.default_rng(12345)
        rg_samples = []
        b_db_samples = []

        for idx in tqdm(sample_indices, desc="Sampling phase stats", unit="pos"):
            rx.position = rx_locs[idx]
            paths = scene.compute_paths(max_depth=2, reflection=True, diffraction=False, scattering=True,
                                        scat_keep_prob=0.5, num_samples=int(5e5))
            if paths.a.shape[-2] == 0:
                continue

            spec = np.transpose(plot_phase_spectrum(paths), (1, 0, 2))
            r = spec[:, :, 0]
            g = spec[:, :, 1]
            b = spec[:, :, 2]

            rg_abs = np.concatenate([np.abs(r).ravel(), np.abs(g).ravel()])
            rg_abs = rg_abs[rg_abs > 0]
            if rg_abs.size > 0:
                take = min(8000, rg_abs.size)
                rg_samples.append(rng.choice(rg_abs, size=take, replace=False))

            b_db = 10.0 * np.log10(np.maximum(b, 1e-30))
            b_db = b_db[np.isfinite(b_db)]
            if b_db.size > 0:
                take = min(8000, b_db.size)
                b_db_samples.append(rng.choice(b_db, size=take, replace=False))

        if len(rg_samples) > 0:
            rg_scale = float(np.percentile(np.concatenate(rg_samples), 99.0))
        else:
            rg_scale = 1.0
        rg_scale = max(rg_scale, 1e-12)

        if len(b_db_samples) > 0:
            b_all = np.concatenate(b_db_samples)
            b_lo_db = float(np.percentile(b_all, 10.0))
            b_hi_db = float(np.percentile(b_all, 99.0))
        else:
            b_lo_db, b_hi_db = -120.0, -40.0

        if b_hi_db <= b_lo_db:
            b_hi_db = b_lo_db + 1.0

        phase_norm_params = {
            "rg_scale": rg_scale,
            "b_lo_db": b_lo_db,
            "b_hi_db": b_hi_db,
        }

        stats_file = os.path.join(output_dir, "phase_norm_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"rg_scale {rg_scale}\n")
            f.write(f"b_lo_db {b_lo_db}\n")
            f.write(f"b_hi_db {b_hi_db}\n")
        print(f"Phase normalization stats: rg_scale={rg_scale:.4e}, b_lo_db={b_lo_db:.2f}, b_hi_db={b_hi_db:.2f}")
        print(f"Saved phase normalization stats to {stats_file}")

    # 2. Generate Dataset
    image_counter = 1
    for i, rx_loc in enumerate(tqdm(rx_locs, desc="Generating views", unit="pos")):
        rx.position = rx_loc
        
        # Higher sample count for final images
        paths = scene.compute_paths(max_depth=2, reflection=True, diffraction=False, scattering=True, 
                                    scat_keep_prob=0.5, num_samples=int(5e5))
        
        # print("paths:", paths.a.shape[-2])
        
        # Dispatch to correct spectrum function
        if spectrum_type == 'aod':
            spec = np.transpose(plot_aod_spatial_spectrum(paths), (1, 0, 2))
        elif spectrum_type == 'delay':
            spec = np.transpose(plot_delay_spatial_spectrum(paths), (1, 0, 2))
        elif spectrum_type == 'phase':
            spec = np.transpose(plot_phase_spectrum(paths), (1, 0, 2))
        else:  # 'mpc'
            spec = plot_spatial_spectrum(paths).T

        # Normalize for visualization
        if spectrum_type == 'phase':
            spec_norm = normalize_phase_spectrum(
                spec,
                rg_scale=phase_norm_params["rg_scale"],
                b_lo_db=phase_norm_params["b_lo_db"],
                b_hi_db=phase_norm_params["b_hi_db"],
            )
        else:
            spec_norm = np.clip((spec - spec_min) / (spec_max - spec_min + 1e-9), 0, 1)

        # print(f"Normalization: spec={spec.shape}, norm_range=[{np.min(spec_norm):.3f}, {np.max(spec_norm):.3f}]")

        # Prepare for projection (add channel dim for MPC)
        if spectrum_type == 'mpc':
            spec_input = spec_norm[..., None]
        else:
            spec_input = spec_norm

        # Generate 3 orientations
        for angle in tqdm([0, 2*np.pi/3, 4*np.pi/3], desc=f"  Orientations {i+1}/{len(rx_locs)}", leave=False, unit="view"):  # 3×120° = exact 360° azimuth, zero overlap
            # Convert Equirectangular to Perspective (MPC stays scalar; AoD/Delay are RGB)
            persp_norm = equirectangular_to_perspective(spec_input, h_fov_deg, angle*180/np.pi, 90, height, width)
            
            # Quality check: for phase use B channel as signal proxy, otherwise max intensity.
            signal_peak = np.max(persp_norm[..., 2]) if spectrum_type == 'phase' else np.max(persp_norm)
            if signal_peak < 0.1:
                continue

            # MPC is saved as a single-channel grayscale image.
            if spectrum_type == 'mpc':
                persp_img = persp_norm[..., 0]
            else:
                persp_img = persp_norm
            # persp_img = persp_norm

            # Ensure float32 0-1
            persp_img = np.clip(persp_img, 0, 1)
            
            # Save Image
            img_filename = f"{image_counter:05d}.png"
            img_path = os.path.join(spectrum_dir, img_filename)
            if spectrum_type == 'mpc':
                plt.imsave(img_path, persp_img, cmap='gray', vmin=0.0, vmax=1.0)
                # np.save(img_path, persp_img.astype(np.float32))
            else:
                plt.imsave(img_path, persp_img)
            
            # Save Pose
            orientation = [angle, 0, 0] # Yaw, Pitch, Roll
            R_c2w, qvec_c2w = euler_to_quaternion(orientation)
            tvec_w2c = rx_loc
            tvec_c2w = -R_c2w.apply(tvec_w2c)
            
            images[image_counter] = colmap_Image(image_counter, qvec_c2w, tvec_c2w, camera_id, img_filename, [], [])
            image_counter += 1
            
    print("")

    save_intrinsics_text(cameras_file, cameras)
    save_extrinsics_text(images_file, images)
    print("Dataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RF-3DGS training dataset from Sionna ray tracing.")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--ideal", action="store_true",
                      help="Ideal MPC/AoD/Delay spectrum dataset (no array beamforming) [default]")
    mode.add_argument("--mvdr",  action="store_true",
                      help="MVDR (Capon) beamforming: sharp adaptive nulls, needs many paths")
    mode.add_argument("--cbf",   action="store_true",
                      help="CBF (Bartlett/delay-and-sum): always stable, fully deterministic [recommended over --mvdr]")
    parser.add_argument("--spectrum-type", "--spectrum_type", dest="spectrum_type",
                        choices=["mpc", "aod", "delay", "phase"], default="mpc",
                        help="Spectrum type for --ideal mode (default: mpc)")
    parser.add_argument("--mvdr-m", type=int, default=4,
                        help="Array size M for MVDR: uses MxM UPA (default: 4 → 4x4=16 elements)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto based on mode)")
    parser.add_argument("--scene", type=str, default="room_with_cube.xml",
                        help="Sionna scene XML file (default: room_with_cube.xml)")
    args = parser.parse_args()
    if not args.ideal and not args.mvdr and not args.cbf:
        args.ideal = True  # default mode when no flag given

    if not os.path.exists(args.scene):
        print(f"Error: scene file '{args.scene}' not found.")
        exit(1)

    scene = load_scene(args.scene)
    
    # 28 GHz — lower FSPL (+6.6 dB vs 60 GHz), stronger multipath, ITU-R P.2040-2 material params below
    scene.frequency = 3.5e9 
    scene.synthetic_array = True 
    wavelength = 299792458 / scene.frequency
    
    # TX: single iso element — ideal spectra
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)

    if args.ideal:
        # RX: single iso element — ideal MPC/AoD/Delay reads paths directly, no beamforming
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                    vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    else:  # --mvdr or --cbf
        # RX: M×M tr38901 UPA — both CBF and MVDR use the array element pattern
        M = args.mvdr_m
        scene.rx_array = PlanarArray(num_rows=M, num_cols=M, pattern="tr38901", polarization="V",
                                    vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
        method_name = 'MVDR' if args.mvdr else 'CBF'
        print(f"{method_name} mode: {M}x{M} tr38901 array ({M**2} elements) at {scene.frequency/1e9} GHz")

    # Define Materials with Scattering (Required for Ideal MPC)
    global_scattering_coeff = 4
    
    # Create materials but check if they exist (Sionna might load them from XML with generic names)
    # The snippet creates new RadioMaterials with scattering properties.
    
    # Conductivities at 28 GHz via ITU-R P.2040-2: σ(f_GHz) = c * f_GHz^d
    # concrete: c=0.0462, d=0.7822 → σ(28) = 0.0462 * 28^0.7822 = 0.626 S/m
    # wood:     c=0.0047, d=1.0718 → σ(28) = 0.0047 * 28^1.0718 = 0.167 S/m
    # glass:    c=0.0043, d=1.1925 → σ(28) = 0.0043 * 28^1.1925 = 0.229 S/m
    # mat_concrete = RadioMaterial("mat_concrete_scat", relative_permittivity=5.24, conductivity=0.626,
    #                              scattering_coefficient=0.1*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=5))
    # mat_wood = RadioMaterial("mat_wood_scat", relative_permittivity=1.99, conductivity=0.167,
    #                              scattering_coefficient=0.2*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=3))
    # mat_glass = RadioMaterial("mat_glass_scat", relative_permittivity=6.27, conductivity=0.229,
    #                              scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))
    # mat_metal = RadioMaterial("mat_metal_scat", relative_permittivity=1, conductivity=1e7,
    #                              scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))

    # 3.5 GHz — Sub-6 GHz, ITU-R P.2040-2: σ = c × f_GHz^d
    # concrete: c=0.0462, d=0.7822 → σ(3.5) = 0.0462 × 3.5^0.7822 = 0.123 S/m
    # wood:     c=0.0047, d=1.0718 → σ(3.5) = 0.0047 × 3.5^1.0718 = 0.018 S/m
    # glass:    c=0.0043, d=1.1925 → σ(3.5) = 0.0043 × 3.5^1.1925 = 0.019 S/m
    # permittivity (ε = a × f^b): b≈0 for all → frequency-independent

    mat_concrete = RadioMaterial("mat_concrete_scat", relative_permittivity=5.24, conductivity=0.123,
                             scattering_coefficient=0.1*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=5))
    mat_wood = RadioMaterial("mat_wood_scat", relative_permittivity=1.99, conductivity=0.018,
                             scattering_coefficient=0.2*global_scattering_coeff,scattering_pattern=sionna.rt.DirectivePattern(alpha_r=3))
    mat_glass = RadioMaterial("mat_glass_scat", relative_permittivity=6.27, conductivity=0.019,
                             scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))
    mat_metal = RadioMaterial("mat_metal_scat", relative_permittivity=1, conductivity=1e7,
                             scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))

    # Add materials safely
    for mat in [mat_concrete, mat_wood, mat_glass, mat_metal]:
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    # Re-assign materials to objects
    # This overwrites the materials assigned by load_scene from fixed XML
    print("Re-assigning materials for Scattering...")
    for obj_name, obj in scene.objects.items():
        # Clean naming matching
        name = obj_name.lower()
        if "floor" in name or "walls" in name or "ceiling" in name or "pillar" in name:
            obj.radio_material = "mat_concrete_scat"
        elif "furniture" in name or "door" in name:
            obj.radio_material = "mat_wood_scat"
        elif "window" in name:
            obj.radio_material = "mat_glass_scat"
        elif "tv" in name or "led" in name or "cube" in name:
            obj.radio_material = "mat_metal_scat"
        else:
            print(f"Warning: Object {obj_name} using default material.")

    # --- Sampling Campaign ---
    x_range = np.linspace(0.3, 6.7, 12)  # 0.3m to 6.7m
    y_range = np.linspace(0.3, 4.7, 8)   # 0.3m to 4.7m
    z_height = [1.2, 2.5]   # 1.2m
    rx_locs = [[x, y, z] for x in x_range for y in y_range for z in z_height]
    
    tx_pos = [0.01, 2.5, 2.9]  # 3.5 2.5 2.7
    
    print(f"Starting dataset generation for {len(rx_locs)} positions...")
    # spectrum_type options:
    #   'mpc'   — MPC arrival-angle spectrum (single-channel grayscale)
    #   'aod'   — AoD departure-angle spectrum (RGB: R=elev, G=azim, B=amp)
    #   'delay' — Propagation-delay spectrum  (RGB: R=delay, G=B=amp)
    #   'phase' — Phase-aware spectrum        (RGB: R=A*cos(phi), G=A*sin(phi), B=A)
    if args.ideal:
        out_dir = args.output_dir or f"dataset_ideal_{args.spectrum_type}"
        generate_ideal_dataset(scene, rx_locs, tx_pos, out_dir, spectrum_type=args.spectrum_type)
    elif args.cbf:
        out_dir = args.output_dir or f"dataset_cbf_M{args.mvdr_m}"
        generate_mvdr_dataset(scene, rx_locs, tx_pos, out_dir, M=args.mvdr_m, method='cbf')
    else:  # --mvdr
        out_dir = args.output_dir or f"dataset_mvdr_M{args.mvdr_m}"
        generate_mvdr_dataset(scene, rx_locs, tx_pos, out_dir, M=args.mvdr_m, method='mvdr')
