# RF Data Generation Improvements Summary

## Overview
Enhanced `generate_single_rf_dynamic.py` with RF-physics best practices for Radio Radiance Field (RRF) training.

## Improvements Implemented

### 1. ✅ Proper Logarithmic (dB) Scale
**Problem:** RF power varies by 10⁶× between line-of-sight and reflections. Linear scale shows only bright transmitter, hiding reflections.

**Solution:**
- Proper dB conversion: `10 * log10(power + epsilon)`
- Noise floor: -160 dB (appropriate for 60GHz indoor RF)
- Makes weak reflections visible alongside strong paths

**Code:**
```python
NOISE_FLOOR_DB = -160.0
img_dB = 10 * np.log10(img_power + EPSILON)
img_dB = np.maximum(img_dB, NOISE_FLOOR_DB)
```

**Result:** Spectrum range -160 to -60 dB (100 dB dynamic range)

---

### 2. ✅ Phase Information Preserved
**Problem:** `abs(amplitude)` discards phase → neural network can't learn interference patterns (constructive/destructive fading).

**Solution:**
- Save complex RF data: amplitude + phase as separate channels
- Store as `.npz` files with metadata
- Visualization uses `abs()`, training data keeps complex values

**Code:**
```python
# Preserve complex amplitudes during accumulation
complex_amps = intensities[0, 0, 0, 0, 0, :, 0]  # Keep phase
img_complex[xmin:xmax, ymin:ymax] += gauss_norm * complex_amps[idx]

# Save for RRF training
np.savez_compressed(rf_data_path,
                   amplitude=amplitude_persp.astype(np.float32),
                   phase=phase_persp.astype(np.float32),
                   frequency=scene.frequency,
                   ...)
```

**Result:** Phase coverage 94.9%, variance 3.017 rad → interference patterns captured

---

### 3. ✅ Physics-Appropriate Resolution
**Problem:** 60GHz has ~5mm wavelength → physical diffraction limit. High resolutions (1024×1024) imply false precision.

**Solution:**
- Reduced resolution: 300×200 → **128×128**
- Matches antenna aperture physics
- Clean, smooth images > noisy high-res artifacts

**Justification:**
- Angular resolution ≈ λ/D where D = antenna size
- For 5mm wavelength and ~10cm aperture: ~3° resolution
- 128×128 with 90° FOV → 0.7° per pixel (conservative)

**Result:** Physically meaningful resolution without misleading detail

---

### 4. ✅ RF Metadata Saved
**Problem:** Neural network needs context (frequency, wavelength, positions) for proper reconstruction.

**Solution:**
Save metadata in `.npz` files:
- `frequency`: 60 GHz
- `wavelength`: 5.00 mm
- `rx_position`, `tx_position`: [x, y, z] in meters
- `yaw`, `pitch`: viewing angles in degrees
- `spec_max_dB`, `spec_min_dB`: dB range for normalization

**Result:** Self-contained training data with full RF context

---

## Output Files

For each receiver position:
1. **Visualization PNG** (`dynamic_rf_0001.png`)
   - 128×128 color image (Jet colormap)
   - For human inspection
   - Size: ~17-24 KB

2. **Complex RF Data** (`dynamic_rf_0001_complex.npz`)
   - `amplitude`: 128×128 float32 (linear power)
   - `phase`: 128×128 float32 (radians, -π to π)
   - Metadata: frequency, wavelength, positions, angles
   - Size: ~112 KB (compressed)

---

## Verification Results

```
✓ Resolution: 128x128 (physics-appropriate for 60GHz)
✓ Phase data: Preserved (94.9% coverage)
✓ Amplitude range: 8.3e+06x dynamic range (strong interference)
✓ Phase variance: 3.017 rad (captures fading patterns)
✓ Ready for RRF training
```

---

## Usage for RRF Training

### Load Complex Data
```python
import numpy as np

data = np.load('dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001_complex.npz')

amplitude = data['amplitude']  # (128, 128)
phase = data['phase']          # (128, 128)

# Reconstruct complex field
complex_field = amplitude * np.exp(1j * phase)

# Or use amplitude + phase as 2 channels for neural network
rf_input = np.stack([amplitude, phase], axis=-1)  # (128, 128, 2)
```

### Normalize for Training
```python
# Amplitude: Log scale (dB) normalization
amplitude_dB = 10 * np.log10(amplitude + 1e-20)
amplitude_norm = (amplitude_dB - data['spec_min_dB']) / (data['spec_max_dB'] - data['spec_min_dB'])

# Phase: Already in [-π, π], optionally normalize to [0, 1]
phase_norm = (phase + np.pi) / (2 * np.pi)
```

---

## Key Takeaways

1. **Always use dB scale** for RF visualization and normalization
2. **Preserve phase** for neural networks to learn interference
3. **Match resolution to physics** - don't chase megapixels
4. **Include metadata** - frequency, wavelength, positions are crucial

---

## References
- Noise floor: Typical indoor 60GHz thermal noise + receiver sensitivity
- Resolution: Rayleigh criterion (angular resolution ≈ 1.22 λ/D)
- Phase preservation: Essential for Radio Tomography and RRF methods
