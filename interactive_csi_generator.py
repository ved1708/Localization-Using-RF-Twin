import sys
import os
import argparse
import numpy as np
from sionna.rt import load_scene, PlanarArray, RadioMaterial
import sionna
import sionna.channel

# We can import the function directly
from generate_csi_dataset import generate_ideal_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="room_with_cube.xml")
    parser.add_argument("--spectrum-type", type=str, default="delay")
    parser.add_argument("--spec-min", type=float, default=None)
    parser.add_argument("--spec-max", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="localisation_frames")
    args = parser.parse_args()

    scene = load_scene(args.scene)
    scene.frequency = 3.5e9 
    scene.synthetic_array = True 
    wavelength = 299792458 / scene.frequency

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                    vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)

    global_scattering_coeff = 4
    mat_concrete = RadioMaterial("mat_concrete_scat", relative_permittivity=5.24, conductivity=0.123,
                             scattering_coefficient=0.1*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=5))
    mat_wood = RadioMaterial("mat_wood_scat", relative_permittivity=1.99, conductivity=0.018,
                             scattering_coefficient=0.2*global_scattering_coeff,scattering_pattern=sionna.rt.DirectivePattern(alpha_r=3))
    mat_glass = RadioMaterial("mat_glass_scat", relative_permittivity=6.27, conductivity=0.019,
                             scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))
    mat_metal = RadioMaterial("mat_metal_scat", relative_permittivity=1, conductivity=1e7,
                             scattering_coefficient=0.025*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=10))

    for mat in [mat_concrete, mat_wood, mat_glass, mat_metal]:
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    for obj_name, obj in scene.objects.items():
        name = obj_name.lower()
        if "floor" in name or "walls" in name or "ceiling" in name or "pillar" in name:
            obj.radio_material = "mat_concrete_scat"
        elif "furniture" in name or "door" in name:
            obj.radio_material = "mat_wood_scat"
        elif "window" in name:
            obj.radio_material = "mat_glass_scat"
        elif "tv" in name or "led" in name or "cube" in name:
            obj.radio_material = "mat_metal_scat"

    tx_pos = [0.01, 2.5, 2.9]
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 3: continue
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        yaw = float(parts[3]) if len(parts) >= 4 else 0.0

        print(f"PROCESSING {x} {y} {z} [Auto-yaw]", flush=True)
        generate_ideal_dataset(scene, [[x, y, z]], tx_pos, args.output_dir, 
                               spectrum_type=args.spectrum_type, 
                               spec_min=args.spec_min, spec_max=args.spec_max, 
                               rx_yaw=None)
        print("DONE", flush=True)

if __name__ == "__main__":
    main()
