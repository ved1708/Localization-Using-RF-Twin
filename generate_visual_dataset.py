import bpy
import os
import sys
import json
import math
import mathutils
import random
import numpy as np
import argparse
import time

# ================= CONFIGURATION =================
BASE_DIR = "/home/ved/Ved/Project_1"

# Parse command-line arguments (filter to args after '--')
if '--' in sys.argv:
    script_args = sys.argv[sys.argv.index('--') + 1:]
else:
    script_args = sys.argv[1:]

parser = argparse.ArgumentParser(description='Generate 325-image visual dataset for 3DGS training')
parser.add_argument('--meshes_dir', type=str, default='meshes',
                    help='Directory containing PLY mesh files')
parser.add_argument('--output_dir', type=str, default='dataset_visual_1',
                    help='Output directory for dataset')
parser.add_argument('--test_ratio', type=float, default=0.10,
                    help='Test set ratio (0.0 to 0.5 recommended)')
parser.add_argument('--split_mode', type=str, default='spatial', choices=['spatial', 'block', 'periodic'],
                    help='Train/test split strategy')
parser.add_argument('--test_block_size', type=int, default=8,
                    help='Contiguous block size for block split mode')
parser.add_argument('--spatial_min_dist', type=float, default=0.35,
                    help='Minimum camera-position distance between selected test frames in spatial split')
parser.add_argument('--split_seed', type=int, default=42,
                    help='Random seed for deterministic train/test splitting')

# Parse arguments
args = parser.parse_args(script_args)

# Set directories
INPUT_MODELS_DIR = os.path.join(BASE_DIR, args.meshes_dir) if not os.path.isabs(args.meshes_dir) else args.meshes_dir
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

RESOLUTION = 500  # High resolution for better quality

# Lower frame density to reduce overlap/overfitting
PERIMETER_FRAMES = 72
# Strategy 2: moderate coverage in XY + multiple heights with a couple of rotations
# Use all 5×3=15 grid positions to cover the room.
GRID_DETAIL_POSITIONS = 15  # Unique XY positions
GRID_DETAIL_HEIGHTS = [0.7, 1.2, 1.8, 2.6]  # Different camera heights
GRID_DETAIL_ROTATIONS = 2  # Different look directions per height
GRID_DETAIL_FRAMES = GRID_DETAIL_POSITIONS * len(GRID_DETAIL_HEIGHTS) * GRID_DETAIL_ROTATIONS
TOP_DOWN_FRAMES = 56
FOCUS_FRAMES_PER_OBJECT = 14
NUM_IMAGES = PERIMETER_FRAMES + GRID_DETAIL_FRAMES + TOP_DOWN_FRAMES

# Room Dimensions
ROOM_MIN = mathutils.Vector((0.5, 0.5, 0.0))
ROOM_MAX = mathutils.Vector((6.5, 4.5, 3.0)) 
CENTER = (ROOM_MIN + ROOM_MAX) / 2

MATERIAL_COLORS = {
    "walls": (0.85, 0.85, 0.85, 1),
    "floor": (0.25, 0.25, 0.25, 1),
    "ceiling": (0.95, 0.95, 0.95, 1),
    "door": (0.4, 0.2, 0.1, 1),
    "window": (0.7, 0.8, 1.0, 1),
    "furniture": (0.55, 0.35, 0.15, 1),
    "furniture_center": (0.55, 0.35, 0.15, 1),
    "pillar": (0.85, 0.85, 0.85, 1),
    "led_tv": (0.05, 0.05, 0.05, 1),
    "metallic_cube": (0.7, 0.7, 0.7, 1)
}

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 96  # Higher quality with GPU acceleration
    scene.cycles.use_denoising = True  # Re-enabled for cleaner images
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Fast denoiser
    scene.cycles.max_bounces = 3  # Increased for better lighting
    
    # Enable GPU rendering (CRITICAL for speed)
    scene.cycles.device = 'GPU'
    
    # Get cycles preferences and enable GPU
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        
        # Enable CUDA/OptiX (for NVIDIA GPUs)
        cycles_prefs.refresh_devices()
        cycles_prefs.compute_device_type = 'CUDA'
        
        # Enable all GPU devices
        for device in cycles_prefs.devices:
            if device.type in ('CUDA', 'OPTIX', 'HIP'):
                device.use = True
                print(f"Enabled GPU: {device.name} ({device.type})")
    except Exception as e:
        print(f"GPU setup warning: {e}")
        print("Falling back to CPU rendering")
    
    # ---  Neutral Exposure ---
    if hasattr(scene.view_settings, 'view_transform'):
        scene.view_settings.view_transform = 'Standard' 
    
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = False  # Opaque background for realistic camera images
    
    # Set background color (optional - will use world background)
    scene.render.image_settings.color_mode = 'RGB'  # RGB instead of RGBA
    
    # Disable verbose render output
    scene.render.use_stamp = False

def create_high_feature_material(name, color, is_glass=False, is_metal=False):
    """Create realistic material with procedural textures."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    bsdf.inputs['Base Color'].default_value = color
    
    if is_glass:
        # Glass optimized for RRF reconstruction
        # Semi-transparent with visible surface details for better feature matching
        bsdf.inputs['Transmission Weight'].default_value = 0.7  # 70% transparent - good balance
        bsdf.inputs['Roughness'].default_value = 0.25  # Slight frosting for surface visibility
        bsdf.inputs['IOR'].default_value = 1.52  # Real glass IOR for accurate refraction
        
        # Add subtle noise for surface detail
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 80.0
        noise.inputs['Detail'].default_value = 2.0
        
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.color_ramp.elements[0].position = 0.45
        color_ramp.color_ramp.elements[1].position = 0.55
        links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], bsdf.inputs['Alpha'])
        
        mat.blend_method = 'BLEND'
    elif is_metal:
        # Metallic appearance for cube and LED TV
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.5
    else:
        # Standard material with procedural texture
        noise_large = nodes.new(type='ShaderNodeTexNoise')
        noise_large.inputs['Scale'].default_value = 15.0
        
        noise_small = nodes.new(type='ShaderNodeTexNoise')
        noise_small.inputs['Scale'].default_value = 100.0
        
        mix_rgb = nodes.new(type='ShaderNodeMixRGB')
        mix_rgb.blend_type = 'ADD'
        mix_rgb.inputs['Fac'].default_value = 0.5
        links.new(noise_large.outputs['Fac'], mix_rgb.inputs['Color1'])
        links.new(noise_small.outputs['Fac'], mix_rgb.inputs['Color2'])
        
        bump = nodes.new(type='ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.2
        
        links.new(mix_rgb.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        bsdf.inputs['Roughness'].default_value = 0.8

    return mat

def setup_lighting():
    scene = bpy.context.scene
    
    if not scene.world:
        new_world = bpy.data.worlds.new("World")
        scene.world = new_world

    # 1. Background (Fill)
    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree

    # Ensure a World Output node exists
    output = node_tree.nodes.get('World Output')
    if output is None:
        output = node_tree.nodes.new('ShaderNodeOutputWorld')

    # Ensure a Background node exists and is connected
    bg = node_tree.nodes.get('Background')
    if bg is None:
        bg = node_tree.nodes.new('ShaderNodeBackground')

    if output.inputs['Surface'].is_linked:
        # If other nodes are linked, preserve that behavior.
        pass
    else:
        node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

    bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs['Strength'].default_value = 0.72  # bright room fill, not pure overexposure


    # 2. Sun (Key)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.object
    sun.data.energy = 2.5
    sun.data.angle = 0.5 
    sun.rotation_euler = (math.radians(45), math.radians(15), 0)

    # 3. Ceiling Panel
    bpy.ops.object.light_add(type='AREA', location=(3.5, 2.5, 2.9)) 
    ceiling_light = bpy.context.object
    ceiling_light.name = "Ceiling_Panel"
    ceiling_light.data.energy = 250.0
    ceiling_light.data.size = 4.0     
    ceiling_light.data.color = (1.0, 0.98, 0.9) # RGB only

def import_models():
    """Import PLY models and assign realistic materials."""
    if not os.path.exists(INPUT_MODELS_DIR):
        print(f"ERROR: Directory not found: {INPUT_MODELS_DIR}")
        return
    
    files = [f for f in os.listdir(INPUT_MODELS_DIR) if f.endswith('.ply')]
    
    if not files:
        print(f"ERROR: No PLY files found in {INPUT_MODELS_DIR}")
        return
    
    for filename in files:
        filepath = os.path.join(INPUT_MODELS_DIR, filename)
        
        # Blender 5.0 uses different import operator
        try:
            bpy.ops.wm.ply_import(filepath=filepath)
        except AttributeError:
            # Fallback for older Blender versions
            bpy.ops.import_mesh.ply(filepath=filepath)
        
        if not bpy.context.selected_objects:
            print(f"WARNING: Could not import {filename}")
            continue
            
        obj = bpy.context.selected_objects[0]
        obj.name = filename.replace('.ply', '')
        
        # Determine material - keyword-based search
        color = (0.5, 0.5, 0.5, 1)  # Default color
        is_glass = False
        is_metal = False
        fname_lower = filename.lower()
        
        # Search for keyword matches in MATERIAL_COLORS
        for keyword, mapped_color in MATERIAL_COLORS.items():
            if keyword in fname_lower:
                color = mapped_color
                if "glass" in keyword or "window" in keyword:
                    is_glass = True
                break
        
        # Check for metal objects separately (LED TV and metallic cube)
        if "metallic_cube" in fname_lower or "led_tv" in fname_lower:
            is_metal = True
        
        mat = create_high_feature_material(f"Mat_{obj.name}", color, is_glass=is_glass, is_metal=is_metal)
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"✓ Imported: {obj.name} (glass={is_glass}, metal={is_metal})")

def look_at(obj, target_pos):
    """Rotates camera to look at target vector, with target clamped inside room."""
    clamped_target = clamp_to_room(target_pos, margin=0.05)
    direction = clamped_target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clamp_to_room(position, margin=0.3):
    """Clamps camera position to stay within room boundaries with margin."""
    x = max(ROOM_MIN.x + margin, min(position[0], ROOM_MAX.x - margin))
    y = max(ROOM_MIN.y + margin, min(position[1], ROOM_MAX.y - margin))
    z = max(ROOM_MIN.z + margin, min(position[2], ROOM_MAX.z - margin))
    return mathutils.Vector((x, y, z))

def generate_focus_orbit(target_obj, frames_list, images_dir, start_index, num_frames=30):
    """
    Generates a tight spiral orbit around a specific object to capture details.
    Ensures camera stays within room boundaries.
    """
    # Get object location and dimensions
    center = target_obj.location
    
    # Calculate maximum safe radius based on room boundaries and object position
    max_radius_x = min(center.x - ROOM_MIN.x, ROOM_MAX.x - center.x) - 0.5
    max_radius_y = min(center.y - ROOM_MIN.y, ROOM_MAX.y - center.y) - 0.5
    safe_radius = min(max_radius_x, max_radius_y, 1.4)  # Slightly wider orbit to reduce overlap
    safe_radius = max(0.75, safe_radius)  # Keep enough distance from object
    
    print(f"Generating focus scan for: {target_obj.name} at {center}, radius: {safe_radius:.2f}m")

    for i in range(num_frames):
        t = i / num_frames
        angle = t * 2 * math.pi
        
        # Spiral height: Go from High -> Low to see top and under-sides
        # Start at 2.0m (looking down), end at 0.8m (looking level)
        # Constrain z to be within room height
        desired_z = 2.0 - (t * 1.2)
        current_z = max(ROOM_MIN.z + 0.5, min(desired_z, ROOM_MAX.z - 0.3))
        
        # Calculate Camera Position
        cam_x = center.x + math.cos(angle) * safe_radius
        cam_y = center.y + math.sin(angle) * safe_radius
        
        # Clamp to room boundaries with margin
        cam = bpy.context.scene.camera
        cam.location = clamp_to_room((cam_x, cam_y, current_z))
        
        # Look specifically at the object center (not the room center)
        # We add a slight Z-offset to look at the 'mass' of the object, not its feet
        look_target = center + mathutils.Vector((0, 0, 0.5))
        look_at(cam, look_target)
        
        # Render
        render_frame(
            start_index + i,
            cam,
            images_dir,
            frames_list,
            strategy=f"focus_orbit:{target_obj.name}"
        )


def _camera_xyz(frame):
    matrix = frame["transform_matrix"]
    return np.array([matrix[0][3], matrix[1][3], matrix[2][3]], dtype=np.float32)


def _sanitize_frame(frame):
    return {
        "file_path": frame["file_path"],
        "transform_matrix": frame["transform_matrix"]
    }


def _split_periodic(frames, test_ratio):
    num_frames = len(frames)
    num_test = max(1, int(round(num_frames * test_ratio)))
    step = max(1, num_frames // num_test)
    return set(list(range(0, num_frames, step))[:num_test])


def _split_block(frames, test_ratio, block_size, seed):
    rng = random.Random(seed)
    strategy_to_indices = {}
    for idx, frame in enumerate(frames):
        strategy = frame.get("strategy", "unknown")
        strategy_to_indices.setdefault(strategy, []).append(idx)

    selected = set()
    for strategy, indices in strategy_to_indices.items():
        target = max(1, int(round(len(indices) * test_ratio)))
        candidates = []
        last_start = max(0, len(indices) - block_size)
        for start in range(0, last_start + 1, max(1, block_size)):
            candidates.append(indices[start:start + block_size])
        if not candidates:
            candidates = [indices]
        rng.shuffle(candidates)

        taken = 0
        for block in candidates:
            for idx in block:
                if taken >= target:
                    break
                selected.add(idx)
                taken += 1
            if taken >= target:
                break

        if taken < target:
            leftovers = [idx for idx in indices if idx not in selected]
            rng.shuffle(leftovers)
            for idx in leftovers[:target - taken]:
                selected.add(idx)

    return selected


def _split_spatial(frames, test_ratio, min_dist, seed):
    rng = random.Random(seed)
    strategy_to_indices = {}
    for idx, frame in enumerate(frames):
        strategy = frame.get("strategy", "unknown")
        strategy_to_indices.setdefault(strategy, []).append(idx)

    selected = set()

    for strategy, indices in strategy_to_indices.items():
        target = max(1, int(round(len(indices) * test_ratio)))
        if len(indices) <= target:
            selected.update(indices)
            continue

        order = indices[:]
        rng.shuffle(order)

        chosen = []
        chosen_xyz = []

        for idx in order:
            xyz = _camera_xyz(frames[idx])
            if not chosen_xyz:
                chosen.append(idx)
                chosen_xyz.append(xyz)
            else:
                d_min = min(np.linalg.norm(xyz - cxyz) for cxyz in chosen_xyz)
                if d_min >= min_dist:
                    chosen.append(idx)
                    chosen_xyz.append(xyz)
            if len(chosen) >= target:
                break

        if len(chosen) < target:
            leftovers = [idx for idx in indices if idx not in set(chosen)]
            while leftovers and len(chosen) < target:
                if not chosen_xyz:
                    pick = leftovers.pop()
                else:
                    best_idx = None
                    best_score = -1.0
                    for cand in leftovers:
                        xyz = _camera_xyz(frames[cand])
                        score = min(np.linalg.norm(xyz - cxyz) for cxyz in chosen_xyz)
                        if score > best_score:
                            best_score = score
                            best_idx = cand
                    pick = best_idx
                    leftovers.remove(pick)
                chosen.append(pick)
                chosen_xyz.append(_camera_xyz(frames[pick]))

        selected.update(chosen[:target])

    return selected


def build_train_test_split(frames, test_ratio, split_mode, test_block_size, spatial_min_dist, split_seed):
    if not frames:
        return [], []

    test_ratio = max(0.01, min(float(test_ratio), 0.5))

    if split_mode == 'periodic':
        test_indices = _split_periodic(frames, test_ratio)
    elif split_mode == 'block':
        test_indices = _split_block(frames, test_ratio, max(1, int(test_block_size)), split_seed)
    else:
        test_indices = _split_spatial(frames, test_ratio, float(spatial_min_dist), split_seed)

    train_frames = [_sanitize_frame(f) for i, f in enumerate(frames) if i not in test_indices]
    test_frames = [_sanitize_frame(f) for i, f in enumerate(frames) if i in test_indices]

    return train_frames, test_frames

def main():
    t_start = time.time()
    print("=" * 60)
    print("TRAINING DATASET GENERATOR")
    print("=" * 60)
    print(f"Meshes: {INPUT_MODELS_DIR}")
    print(f"Output: {OUTPUT_DATASET_DIR}")
    print(f"Images to generate: {NUM_IMAGES}")
    print("=" * 60)
    
    t_setup = time.time()
    reset_scene()
    setup_render_engine()
    import_models()
    setup_lighting()
    print(f"Setup time: {time.time() - t_setup:.2f}s")
    
    images_dir = os.path.join(OUTPUT_DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 20 # 20mm is good for room scale
    
    frames = []
    
    print("\nGenerating training dataset...")
    
    # === STRATEGY 1: PERIMETER WALK ===
    # Walk around the walls, looking at the center
    print(f"\nStrategy 1: Perimeter Walk ({PERIMETER_FRAMES} frames)")
    t_s1 = time.time()
    for i in range(PERIMETER_FRAMES):
        t = i / float(PERIMETER_FRAMES)
        angle = t * 2 * math.pi
        
        # Oval path slightly smaller than room limits
        radius_x = (ROOM_MAX.x - ROOM_MIN.x) * 0.35  # Reduced from 0.4 for safety
        radius_y = (ROOM_MAX.y - ROOM_MIN.y) * 0.35
        
        x = CENTER.x + math.cos(angle) * radius_x
        y = CENTER.y + math.sin(angle) * radius_y
        z = 1.6 # Eye level
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look at a point slightly offset from center to create parallax
        look_target = CENTER + mathutils.Vector((math.cos(angle * 2.5) * 0.7, math.sin(angle * 2.5) * 0.7, -0.5))
        look_at(cam, look_target)
        
        render_frame(i, cam, images_dir, frames, strategy="perimeter")
    print(f"\nStrategy 1 completed: {time.time() - t_s1:.2f}s")

    # === STRATEGY 2: LOW "DETAIL" PASS ===
    # Grid-based sampling at low height (human/robot perspective)
    print(f"\n\nStrategy 2: Grid Detail Pass ({GRID_DETAIL_FRAMES} frames)")
    t_s2 = time.time()
    # Use a grid large enough to cover the room (5×3=15 possible positions),
    # but only pick the first GRID_DETAIL_POSITIONS of them to keep the total frame count lower.
    grid_x = 5
    grid_y = 3
    strategy2_start = PERIMETER_FRAMES
    
    for pos_ix in range(GRID_DETAIL_POSITIONS):
        # Deterministic grid positions (fewer XY points so we can take multiple heights)
        grid_idx = pos_ix % (grid_x * grid_y)
        ix = grid_idx % grid_x
        iy = grid_idx // grid_x
        
        # Map to room coordinates with safety margin
        x = ROOM_MIN.x + 0.8 + (ROOM_MAX.x - ROOM_MIN.x - 1.6) * (ix / (grid_x - 1))
        y = ROOM_MIN.y + 0.8 + (ROOM_MAX.y - ROOM_MIN.y - 1.6) * (iy / (grid_y - 1))
        x += 0.18 * math.sin(pos_ix * 1.7)
        y += 0.18 * math.cos(pos_ix * 1.3)
        
        # Use a base rotation for this XY position, then sweep a small rotation per height
        base_angle = (pos_ix / float(GRID_DETAIL_POSITIONS)) * 2 * math.pi
        target_radius = 0.7
        target_z = CENTER.z  # Look at center height
        
        for h_ix, z in enumerate(GRID_DETAIL_HEIGHTS):
            for rot_ix in range(GRID_DETAIL_ROTATIONS):
                angle = base_angle + (rot_ix / GRID_DETAIL_ROTATIONS) * (2 * math.pi / GRID_DETAIL_ROTATIONS)
                target_x = CENTER.x + math.cos(angle) * target_radius
                target_y = CENTER.y + math.sin(angle) * target_radius

                cam.location = clamp_to_room((x, y, z))
                look_at(cam, mathutils.Vector((target_x, target_y, target_z)))

                frame_idx = (
                    strategy2_start
                    + pos_ix * len(GRID_DETAIL_HEIGHTS) * GRID_DETAIL_ROTATIONS
                    + h_ix * GRID_DETAIL_ROTATIONS
                    + rot_ix
                )
                render_frame(frame_idx, cam, images_dir, frames, strategy="grid_detail")
    print(f"\nStrategy 2 completed: {time.time() - t_s2:.2f}s")

    # === STRATEGY 3: TOP-DOWN FILLER ===
    # High up, looking down. Fills floor holes.
    print(f"\n\nStrategy 3: Top-Down Coverage ({TOP_DOWN_FRAMES} frames)")
    t_s3 = time.time()
    strategy3_start = PERIMETER_FRAMES + GRID_DETAIL_FRAMES
    for i in range(TOP_DOWN_FRAMES):
        # Zig Zag pattern
        t = i / float(TOP_DOWN_FRAMES)
        x = ROOM_MIN.x + 0.8 + (ROOM_MAX.x - ROOM_MIN.x - 1.6) * t
        y = CENTER.y + math.sin(t * 8 * math.pi) * 1.4
        z = 2.5 # Near Ceiling
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look specifically at the floor ahead
        look_at(cam, mathutils.Vector((x, y, 0.0)))
        render_frame(strategy3_start + i, cam, images_dir, frames, strategy="top_down")
    print(f"\nStrategy 3 completed: {time.time() - t_s3:.2f}s")

    # === STRATEGY 4: OBJECT FOCUS ORBITS ===
    # Automatically find furniture and orbit it
    print("\n\nStrategy 4: Object Focus Orbits")
    t_s4 = time.time()
    
    # 1. Identify objects of interest based on names
    keywords = ["chair", "table", "sofa", "tv", "desk"]
    target_objects = []
    
    for obj in bpy.context.scene.objects:
        # Check if object name contains any keyword (case insensitive)
        if any(k in obj.name.lower() for k in keywords):
            target_objects.append(obj)
    
    print(f"Found {len(target_objects)} objects to orbit")
            
    # 2. Generate orbits for each found object
    frame_counter = PERIMETER_FRAMES + GRID_DETAIL_FRAMES + TOP_DOWN_FRAMES
    frames_per_object = FOCUS_FRAMES_PER_OBJECT
    
    for obj in target_objects:
        generate_focus_orbit(
            target_obj=obj, 
            frames_list=frames, 
            images_dir=images_dir, 
            start_index=frame_counter, 
            num_frames=frames_per_object
        )
        frame_counter += frames_per_object
    print(f"Strategy 4 completed: {time.time() - t_s4:.2f}s")

    # Save JSON in Blender/NeRF format split into train/test
    # 2DGS expects transforms_train.json and transforms_test.json
    print("\n\nSaving transforms JSON...")
    train_frames, test_frames = build_train_test_split(
        frames=frames,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        test_block_size=args.test_block_size,
        spatial_min_dist=args.spatial_min_dist,
        split_seed=args.split_seed
    )
    
    train_data = {"camera_angle_x": cam.data.angle_x, "frames": train_frames}
    test_data = {"camera_angle_x": cam.data.angle_x, "frames": test_frames}
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_test.json"), 'w') as f:
        json.dump(test_data, f, indent=4)
    
    total_time = time.time() - t_start
    num_frames = len(frames)
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Split mode: {args.split_mode}")
    print(f"Test ratio: {args.test_ratio:.2f}")
    print(f"Train frames: {len(train_frames)}")
    print(f"Test frames: {len(test_frames)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Average per frame: {total_time/num_frames:.2f}s")
    print(f"{'='*60}")


def _is_frame_too_black(filepath, threshold=0.02, max_fraction=0.03, grid_samples=32):
    """Return True if the image contains too many near-black pixels.

    We sample a uniform grid across the image (rather than a flat stride) to
    reliably catch black/void regions in corners or along edges.
    """
    try:
        img = bpy.data.images.load(filepath)
    except Exception:
        return False

    width, height = img.size
    if width <= 0 or height <= 0:
        bpy.data.images.remove(img)
        return False

    pixels = list(img.pixels)  # [r,g,b,a, ...]

    # Sample a uniform grid across the image for better coverage.
    samples = 0
    black_count = 0

    # Ensure at least a small grid even for tiny images
    sx = max(4, min(grid_samples, width))
    sy = max(4, min(grid_samples, height))

    for iy in range(sy):
        y = int((iy + 0.5) * height / sy)
        for ix in range(sx):
            x = int((ix + 0.5) * width / sx)
            idx = (y * width + x) * 4
            r = pixels[idx]
            g = pixels[idx + 1]
            b = pixels[idx + 2]
            samples += 1
            if r < threshold and g < threshold and b < threshold:
                black_count += 1

    bpy.data.images.remove(img)

    # If a substantial portion of the sampled pixels are almost black, discard.
    return (black_count / max(1, samples)) > max_fraction


def render_frame(index, cam, images_dir, frames_list, strategy="unknown"):
    bpy.context.view_layer.update()
    filename = f"frame_{index:04d}.png"
    filepath = os.path.join(images_dir, filename)

    # Check if frame already exists
    if os.path.exists(filepath):
        print(f"  Skipping existing frame {index:04d}")
        return

    # Render the frame
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered frame {index:04d}")

    # Discard frames which are mostly black (bad camera placement)
    if _is_frame_too_black(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
        print(f"  Discarded frame {index:04d} (too dark)")
        return

    # Always capture transform matrix (needed for JSON)
    matrix = cam.matrix_world
    frames_list.append({
        "file_path": f"images/frame_{index:04d}",  # No .png extension
        "transform_matrix": [list(row) for row in matrix],
        "strategy": strategy
    })


if __name__ == "__main__":
    main()