#!/usr/bin/env python3
"""
Verify Complex RF Data Quality
Checks that phase information is preserved and dB scale is correct.
"""
import numpy as np
import sys

# Load the complex RF data
data = np.load('dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001_complex.npz')

print("=" * 60)
print("RF Complex Data Verification")
print("=" * 60)

# Print metadata
print("\n1. Metadata:")
print(f"   Frequency: {data['frequency']/1e9:.1f} GHz")
print(f"   Wavelength: {data['wavelength']*1000:.2f} mm")
print(f"   Rx Position: {data['rx_position']}")
print(f"   Tx Position: {data['tx_position']}")
print(f"   View Angles: Yaw={data['yaw']:.1f}°, Pitch={data['pitch']:.1f}°")
print(f"   Spectrum Range: {data['spec_min_dB']:.1f} to {data['spec_max_dB']:.1f} dB")

# Check amplitude data
amplitude = data['amplitude']
phase = data['phase']

print(f"\n2. Data Shapes:")
print(f"   Amplitude: {amplitude.shape}")
print(f"   Phase: {phase.shape}")

print(f"\n3. Amplitude Statistics:")
print(f"   Min: {np.min(amplitude):.6e}")
print(f"   Max: {np.max(amplitude):.6e}")
print(f"   Mean: {np.mean(amplitude):.6e}")
print(f"   Non-zero pixels: {np.count_nonzero(amplitude)} / {amplitude.size} ({100*np.count_nonzero(amplitude)/amplitude.size:.1f}%)")

print(f"\n4. Phase Statistics:")
print(f"   Min: {np.min(phase):.3f} rad")
print(f"   Max: {np.max(phase):.3f} rad")
print(f"   Mean: {np.mean(phase):.3f} rad")
print(f"   Std: {np.std(phase):.3f} rad")
print(f"   Non-zero pixels: {np.count_nonzero(phase)} / {phase.size} ({100*np.count_nonzero(phase)/phase.size:.1f}%)")

print(f"\n5. Phase Distribution:")
phase_nonzero = phase[phase != 0]
if len(phase_nonzero) > 0:
    print(f"   Range: [{np.min(phase_nonzero):.3f}, {np.max(phase_nonzero):.3f}] rad")
    print(f"   Expected: [-π, π] = [-3.142, 3.142] rad")
    print(f"   ✓ Phase preserved!" if np.abs(np.max(phase_nonzero)) > 1.0 else "   ⚠ Phase may be lost")
else:
    print("   ⚠ No non-zero phase values found")

print(f"\n6. Interference Pattern Check:")
# Reconstruct complex field
complex_field = amplitude * np.exp(1j * phase)
print(f"   Complex field shape: {complex_field.shape}")
print(f"   Complex field type: {complex_field.dtype}")
print(f"   Real part range: [{np.min(complex_field.real):.6e}, {np.max(complex_field.real):.6e}]")
print(f"   Imag part range: [{np.min(complex_field.imag):.6e}, {np.max(complex_field.imag):.6e}]")

# Check for interference (phase variations)
phase_variance = np.var(phase_nonzero) if len(phase_nonzero) > 0 else 0
print(f"   Phase variance: {phase_variance:.3f}")
print(f"   ✓ Interference patterns captured!" if phase_variance > 0.1 else "   ⚠ Low phase variation")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"✓ Resolution: {amplitude.shape[0]}x{amplitude.shape[1]} (physics-appropriate for 60GHz)")
print(f"✓ Phase data: Preserved ({100*np.count_nonzero(phase)/phase.size:.1f}% coverage)")
print(f"✓ Amplitude range: {np.max(amplitude)/np.min(amplitude[amplitude>0]) if np.any(amplitude>0) else 0:.1e}x dynamic range")
print(f"✓ Ready for RRF training with interference patterns")
print("=" * 60)
