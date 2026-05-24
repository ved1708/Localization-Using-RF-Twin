import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial
import sionna

def generate_dataset():
    # Load scene
    scene = load_scene('../room_with_cube.xml')
    scene.frequency = 3.5e9
    scene.synthetic_array = True
    wavelength = 299792458 / scene.frequency

    # TX and RX single isotropic antennas
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                 vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                 vertical_spacing=0.5*wavelength, horizontal_spacing=0.5*wavelength)

    tx = Transmitter("tx", position=[0.01, 2.5, 2.9])
    scene.add(tx)
    rx = Receiver("rx", position=[0, 0, 0])
    scene.add(rx)
    
    global_scattering_coeff = 4
    mat_concrete = RadioMaterial("mat_concrete_scat", relative_permittivity=5.24, conductivity=0.123,
                                 scattering_coefficient=0.1*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=5))
    mat_wood = RadioMaterial("mat_wood_scat", relative_permittivity=1.99, conductivity=0.018,
                             scattering_coefficient=0.2*global_scattering_coeff, scattering_pattern=sionna.rt.DirectivePattern(alpha_r=3))
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

    # Generate locations (25cm grid spacing)
    x_range = np.arange(0.3, 6.7 + 0.01, 0.25)
    y_range = np.arange(0.3, 4.7 + 0.01, 0.25)
    z_heights = [1.0, 1.5, 2.0, 2.5]
    
    rx_locs = []
    for x in x_range:
        for y in y_range:
            for z in z_heights:
                rx_locs.append([float(x), float(y), float(z)])

    dataset = []
    print(f"Starting dataset generation for {len(rx_locs)} positions...")

    for i, loc in enumerate(tqdm(rx_locs)):
        rx.position = loc
        paths = scene.compute_paths(max_depth=3, num_samples=1e6)
        
        # Features (gains and delays)
        a = paths.a.numpy()[0, 0, 0, 0, 0, :, 0] # amplitudes
        tau = paths.tau.numpy()[0, 0, 0, :] # delays
        
        gains = np.abs(a)
        
        mask = gains > 1e-9
        gains = gains[mask]
        delays = tau[mask]
        
        if len(gains) > 0:
            total_power = float(np.sum(gains**2))
            num_paths = int(len(gains))
            # delay spread
            mean_delay = float(np.average(delays, weights=gains**2))
            delay_spread = float(np.sqrt(np.average((delays - mean_delay)**2, weights=gains**2)))
        else:
            total_power = 0.0
            num_paths = 0
            delay_spread = 0.0

        dataset.append({
            'position': loc,
            'features': {
                'path_gains': gains,
                'path_delays': delays,
                'total_power': total_power,
                'num_paths': num_paths,
                'delay_spread': delay_spread
            }
        })
        
    output_pkl = 'rf_dataset.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Dataset generation complete. Saved to {output_pkl}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            pass
    generate_dataset()
