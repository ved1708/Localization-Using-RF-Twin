#!/usr/bin/env python3
"""
Generates separate PLY files for each object/material AND a combined PLY file for a complete room scene.

Creates a 7m x 5m x 3m rectangular room with:
- Floor, ceiling, walls (with window and door cutouts)
- Glass window
- Wooden door
- 3 tables with chairs
- Sofa
- LED TV

Separate files go into 'meshes_d/' subfolder for Sionna RF simulation.
Combined file for 3D visualization.
"""
import os
import struct
import numpy as np

# --- Helper Functions ---

def write_ply_simple(filename, vertices, faces):
    """Writes a binary PLY file without material IDs."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))
        for v in vertices:
            f.write(struct.pack('fff', v[0], v[1], v[2]))
        for face in faces:
            f.write(struct.pack('B', 3))
            f.write(struct.pack('iii', face[0], face[1], face[2]))

def write_ply_combined(filename, vertices, faces, materials):
    """Writes a binary PLY file with material IDs."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
property int material_id
end_header
"""
        f.write(header.encode('ascii'))
        for v in vertices:
            f.write(struct.pack('fff', v[0], v[1], v[2]))
        for face, mat_id in zip(faces, materials):
            f.write(struct.pack('B', 3))
            f.write(struct.pack('iii', face[0], face[1], face[2]))
            f.write(struct.pack('i', mat_id))

class MeshBuilder:
    """Helper class to build meshes."""
    def __init__(self):
        self.vertices = []
        self.faces = []
    
    def add_rectangle(self, x_range, y_range, z_val, flip_normal=False):
        x_min, x_max = x_range
        y_min, y_max = y_range
        base_idx = len(self.vertices)
        self.vertices.extend([
            [x_min, y_min, z_val], [x_max, y_min, z_val],
            [x_max, y_max, z_val], [x_min, y_max, z_val],
        ])
        if flip_normal:
            self.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
        else:
            self.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    
    def add_box(self, x_range, y_range, z_range):
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        base_idx = len(self.vertices)
        self.vertices.extend([
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])
        box_faces = [
            [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ]
        for face in box_faces:
            self.faces.append([base_idx + face[0], base_idx + face[1], base_idx + face[2]])

    def add_sphere(self, center, radius, resolution=16):
        """Adds a sphere to the mesh."""
        base_idx = len(self.vertices)
        
        # Generate sphere vertices using parametric equations
        for i in range(resolution + 1):
            lat = np.pi * (-0.5 + float(i) / resolution)
            cos_lat = np.cos(lat)
            sin_lat = np.sin(lat)
            
            for j in range(resolution + 1):
                lon = 2 * np.pi * float(j) / resolution
                cos_lon = np.cos(lon)
                sin_lon = np.sin(lon)
                
                x = center[0] + radius * cos_lat * cos_lon
                y = center[1] + radius * cos_lat * sin_lon
                z = center[2] + radius * sin_lat
                self.vertices.append([x, y, z])

        # Generate sphere faces
        for i in range(resolution):
            for j in range(resolution):
                p1 = base_idx + i * (resolution + 1) + j
                p2 = base_idx + (i + 1) * (resolution + 1) + j
                p3 = base_idx + (i + 1) * (resolution + 1) + (j + 1)
                p4 = base_idx + i * (resolution + 1) + (j + 1)
                
                self.faces.append([p1, p2, p3])
                self.faces.append([p1, p3, p4])

# Material IDs
MAT_GREY = 0; MAT_WHITE = 1; MAT_WOOD = 2; MAT_GLASS = 3; MAT_METAL = 4; MAT_CONCRETE = 5

def create_scene_meshes(output_dir):
    """Generate all scene meshes."""
    print(f"Generating scene meshes in: {output_dir}")
    meshes_dir = os.path.join(output_dir, 'meshes_d')
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Combined mesh
    combined_v, combined_f, combined_m = [], [], []
    
    def add_to_combined(builder, mat_id):
        offset = len(combined_v)
        combined_v.extend(builder.vertices)
        for face in builder.faces:
            combined_f.append([f + offset for f in face])
            combined_m.append(mat_id)
    
    # Floor - Rectangular 7m x 5m
    floor = MeshBuilder()
    floor.add_rectangle((0, 7), (0, 5), 0)
    write_ply_simple(os.path.join(meshes_dir, 'floor.ply'), floor.vertices, floor.faces)
    add_to_combined(floor, MAT_GREY)
    print("✓ Floor (7m x 5m)")
    
    # Ceiling - Rectangular 7m x 5m
    ceiling = MeshBuilder()
    ceiling.add_rectangle((0, 7), (0, 5), 3, flip_normal=True)
    write_ply_simple(os.path.join(meshes_dir, 'ceiling.ply'), ceiling.vertices, ceiling.faces)
    add_to_combined(ceiling, MAT_WHITE)
    print("✓ Ceiling (7m x 5m)")
    
    # Walls for rectangular room (7m x 5m x 3m)
    walls = MeshBuilder()
    
    # Wall 1 (y=0, front wall)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[0, 0, 0], [7, 0, 0], [7, 0, 3], [0, 0, 3]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    
    # Wall 2 (y=5, back wall) with window cutout
    # Window cutout: centered horizontally (2.5m to 5m), height 1m to 2m
    # Bottom section
    base_idx = len(walls.vertices)
    walls.vertices.extend([[0, 5, 0], [7, 5, 0], [7, 5, 1.0], [0, 5, 1.0]])
    walls.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    # Top section
    base_idx = len(walls.vertices)
    walls.vertices.extend([[0, 5, 2.0], [7, 5, 2.0], [7, 5, 3], [0, 5, 3]])
    walls.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    # Left side of window
    base_idx = len(walls.vertices)
    walls.vertices.extend([[0, 5, 1.0], [2.5, 5, 1.0], [2.5, 5, 2.0], [0, 5, 2.0]])
    walls.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    # Right side of window
    base_idx = len(walls.vertices)
    walls.vertices.extend([[5.0, 5, 1.0], [7, 5, 1.0], [7, 5, 2.0], [5.0, 5, 2.0]])
    walls.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    
    # Wall 3 (x=0, left wall)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[0, 0, 0], [0, 5, 0], [0, 5, 3], [0, 0, 3]])
    walls.faces.extend([[base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3]])
    
    # Wall 4 (x=7, right wall) with door cutout
    # Door cutout: centered at y=2.5m, width 0.9m, height 2.1m
    # Left section (y=0 to y=2.05)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[7, 0, 0], [7, 2.05, 0], [7, 2.05, 2.1], [7, 0, 2.1]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    # Right section (y=2.95 to y=5)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[7, 2.95, 0], [7, 5, 0], [7, 5, 2.1], [7, 2.95, 2.1]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    # Top of door (y=2.05 to y=2.95, z=2.1 to z=3)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[7, 2.05, 2.1], [7, 2.95, 2.1], [7, 2.95, 3], [7, 2.05, 3]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    # Upper left section (y=0 to y=2.05, z=2.1 to z=3)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[7, 0, 2.1], [7, 2.05, 2.1], [7, 2.05, 3], [7, 0, 3]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    # Upper right section (y=2.95 to y=5, z=2.1 to z=3)
    base_idx = len(walls.vertices)
    walls.vertices.extend([[7, 2.95, 2.1], [7, 5, 2.1], [7, 5, 3], [7, 2.95, 3]])
    walls.faces.extend([[base_idx, base_idx+2, base_idx+1], [base_idx, base_idx+3, base_idx+2]])
    
    write_ply_simple(os.path.join(meshes_dir, 'walls.ply'), walls.vertices, walls.faces)
    add_to_combined(walls, MAT_WHITE)
    print("✓ Walls")
    
    # Window (horizontal: 2.5m x 1m, 2cm thick) on back wall
    window = MeshBuilder()
    window.add_box((2.5, 5.0), (4.98, 5.0), (1.0, 2.0))
    write_ply_simple(os.path.join(meshes_dir, 'window.ply'), window.vertices, window.faces)
    add_to_combined(window, MAT_GLASS)
    print("✓ Window")
    
    # Door (0.9m x 2.1m, 5cm thick) on right wall
    door = MeshBuilder()
    door.add_box((6.95, 7.0), (2.05, 2.95), (0, 2.1))
    write_ply_simple(os.path.join(meshes_dir, 'door.ply'), door.vertices, door.faces)
    add_to_combined(door, MAT_WOOD)
    print("✓ Door")
    
    # Tables and Chairs - positioned along left wall (x=0), opposite to door
    furniture = MeshBuilder()
    table_width = 0.76  # depth from wall
    table_length = 1.17  # along the wall
    gap = 0.15
    table_starts_y = [0.5, 0.5 + table_length + gap, 0.5 + 2*(table_length + gap)]  # y positions along left wall
    
    for i, table_y_start in enumerate(table_starts_y):
        table_height, table_thickness = 0.76, 0.05
        table_x = (0.1, 0.1 + table_width)  # 10cm from left wall
        table_y = (table_y_start, table_y_start + table_length)
        
        # Table top & legs
        furniture.add_box(table_x, table_y, (table_height - table_thickness, table_height))
        leg_size = 0.05
        furniture.add_box((table_x[0], table_x[0]+leg_size), (table_y[0], table_y[0]+leg_size), (0, table_height - table_thickness))
        furniture.add_box((table_x[1]-leg_size, table_x[1]), (table_y[0], table_y[0]+leg_size), (0, table_height - table_thickness))
        furniture.add_box((table_x[0], table_x[0]+leg_size), (table_y[1]-leg_size, table_y[1]), (0, table_height - table_thickness))
        furniture.add_box((table_x[1]-leg_size, table_x[1]), (table_y[1]-leg_size, table_y[1]), (0, table_height - table_thickness))
        
        # Chair - positioned away from wall (facing the table)
        chair_width, chair_depth = 0.45, 0.43
        chair_seat_height, chair_seat_thickness = 0.46, 0.05
        chair_backrest_height, chair_backrest_thickness = 0.40, 0.05
        chair_y_center = (table_y[0] + table_y[1]) / 2
        chair_y = (chair_y_center - chair_width/2, chair_y_center + chair_width/2)
        chair_x_front = table_x[1] + 0.05
        chair_x_back = chair_x_front + chair_depth
        
        furniture.add_box((chair_x_front, chair_x_back), chair_y, (chair_seat_height - chair_seat_thickness, chair_seat_height))
        furniture.add_box((chair_x_back - chair_backrest_thickness, chair_x_back), chair_y, (chair_seat_height, chair_seat_height + chair_backrest_height))
        
        leg_size = 0.04
        furniture.add_box((chair_x_back-leg_size, chair_x_back), (chair_y[0], chair_y[0]+leg_size), (0, chair_seat_height - chair_seat_thickness))
        furniture.add_box((chair_x_back-leg_size, chair_x_back), (chair_y[1]-leg_size, chair_y[1]), (0, chair_seat_height - chair_seat_thickness))
        furniture.add_box((chair_x_front, chair_x_front+leg_size), (chair_y[0], chair_y[0]+leg_size), (0, chair_seat_height - chair_seat_thickness))
        furniture.add_box((chair_x_front, chair_x_front+leg_size), (chair_y[1]-leg_size, chair_y[1]), (0, chair_seat_height - chair_seat_thickness))
        
        print(f"✓ Table{i+1} and Chair{i+1}")
    
    write_ply_simple(os.path.join(meshes_dir, 'furniture.ply'), furniture.vertices, furniture.faces)
    add_to_combined(furniture, MAT_WOOD)
    
    # # Pillar - 15cm x 15cm, positioned 30cm from middle chair (chair 2)
    # pillar = MeshBuilder()
    # pillar_size = 0.35
    # middle_chair_x = table_starts_y[1]  # y position of middle table
    # middle_chair_center_y = middle_chair_x + table_length / 2
    # pillar_distance = 0.50  # 30cm from chair
    # middle_chair_back_x = 0.1 + table_width + 0.05 + 0.43  # chair back position
    # pillar_x_center = middle_chair_back_x + pillar_distance
    # pillar_x = (pillar_x_center - pillar_size/2, pillar_x_center + pillar_size/2)
    # pillar_y = (middle_chair_center_y - pillar_size/2, middle_chair_center_y + pillar_size/2)
    
    # pillar.add_box(pillar_x, pillar_y, (0, 3.0))  # Floor to ceiling
    # write_ply_simple(os.path.join(meshes_dir, 'pillar.ply'), pillar.vertices, pillar.faces)
    # add_to_combined(pillar, MAT_CONCRETE)
    # print("✓ Pillar (15cm x 15cm, concrete)")
    
    # Center coffee table and sofas
    furniture_center = MeshBuilder()
    
    # Center coffee table - small height table at room center
    center_table_width = 1.2
    center_table_length = 0.6
    center_table_height = 0.4
    center_table_thickness = 0.05
    room_center_x = 3.5  # Center of 7m room
    room_center_y = 2.5  # Center of 5m room
    
    ct_x = (room_center_x - center_table_width/2, room_center_x + center_table_width/2)
    ct_y = (room_center_y - center_table_length/2, room_center_y + center_table_length/2)
    
    # Coffee table top
    furniture_center.add_box(ct_x, ct_y, (center_table_height - center_table_thickness, center_table_height))
    # Coffee table legs
    leg_size = 0.05
    furniture_center.add_box((ct_x[0], ct_x[0]+leg_size), (ct_y[0], ct_y[0]+leg_size), (0, center_table_height - center_table_thickness))
    furniture_center.add_box((ct_x[1]-leg_size, ct_x[1]), (ct_y[0], ct_y[0]+leg_size), (0, center_table_height - center_table_thickness))
    furniture_center.add_box((ct_x[0], ct_x[0]+leg_size), (ct_y[1]-leg_size, ct_y[1]), (0, center_table_height - center_table_thickness))
    furniture_center.add_box((ct_x[1]-leg_size, ct_x[1]), (ct_y[1]-leg_size, ct_y[1]), (0, center_table_height - center_table_thickness))
    print("✓ Center coffee table")
    
    # Two sofas facing each other (north and south of center table)
    sofa_length = 2.0
    sofa_depth = 0.7
    sofa_seat_height = 0.4
    sofa_seat_thickness = 0.1
    sofa_backrest_height = 0.5
    sofa_backrest_thickness = 0.12
    
    # Sofa 1 - South side (lower y), facing north
    sofa1_y_back = ct_y[0] - 0.4  # 40cm gap from table
    sofa1_y_front = sofa1_y_back - sofa_depth
    sofa1_x = (room_center_x - sofa_length/2, room_center_x + sofa_length/2)
    
    # Sofa 1 seat
    furniture_center.add_box(sofa1_x, (sofa1_y_front, sofa1_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, sofa_seat_height))
    # Sofa 1 backrest
    furniture_center.add_box(sofa1_x, (sofa1_y_front, sofa1_y_front + sofa_backrest_thickness), 
                     (sofa_seat_height, sofa_seat_height + sofa_backrest_height))
    # Sofa 1 armrests
    armrest_width = 0.1
    armrest_height = sofa_seat_height + 0.15
    furniture_center.add_box((sofa1_x[0], sofa1_x[0] + armrest_width), (sofa1_y_front, sofa1_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, armrest_height))
    furniture_center.add_box((sofa1_x[1] - armrest_width, sofa1_x[1]), (sofa1_y_front, sofa1_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, armrest_height))
    print("✓ Sofa 1 (south side)")
    
    # Sofa 2 - North side (higher y), facing south
    sofa2_y_front = ct_y[1] + 0.4  # 40cm gap from table
    sofa2_y_back = sofa2_y_front + sofa_depth
    sofa2_x = (room_center_x - sofa_length/2, room_center_x + sofa_length/2)
    
    # Sofa 2 seat
    furniture_center.add_box(sofa2_x, (sofa2_y_front, sofa2_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, sofa_seat_height))
    # Sofa 2 backrest
    furniture_center.add_box(sofa2_x, (sofa2_y_back - sofa_backrest_thickness, sofa2_y_back), 
                     (sofa_seat_height, sofa_seat_height + sofa_backrest_height))
    # Sofa 2 armrests
    furniture_center.add_box((sofa2_x[0], sofa2_x[0] + armrest_width), (sofa2_y_front, sofa2_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, armrest_height))
    furniture_center.add_box((sofa2_x[1] - armrest_width, sofa2_x[1]), (sofa2_y_front, sofa2_y_back), 
                     (sofa_seat_height - sofa_seat_thickness, armrest_height))
    print("✓ Sofa 2 (north side)")
    
    write_ply_simple(os.path.join(meshes_dir, 'furniture_center.ply'), furniture_center.vertices, furniture_center.faces)
    add_to_combined(furniture_center, MAT_WOOD)
    
    # LED TV on back wall (y=5), facing sofa
    led_tv = MeshBuilder()
    # TV dimensions: 55" screen = 1.2m wide x 0.7m tall x 0.05m thick
    tv_width = 1.2
    tv_height = 0.7
    tv_thickness = 0.05
    tv_center_height = 1.5  # Center of TV at 1.5m height
    
    # Position: On back wall (y=5), aligned with sofa area
    tv_x_center = 3.5  # Center of room
    tv_y = (tv_thickness, 0)  # On back wall at y=5
    tv_x = (tv_x_center - tv_width/2, tv_x_center + tv_width/2)
    tv_z = (tv_center_height - tv_height/2, tv_center_height + tv_height/2)
    
    # TV screen (thin box)
    led_tv.add_box(tv_x, tv_y, tv_z)
    
    write_ply_simple(os.path.join(meshes_dir, 'led_tv.ply'), led_tv.vertices, led_tv.faces)
    add_to_combined(led_tv, MAT_METAL)
    print("✓ LED TV on back wall (facing sofa)")

    # # ---------------------------------------------------
    # # Standing Person (RF human phantom approximation)
    # # ---------------------------------------------------
    # person = MeshBuilder()

    # # Dimensions based on RF human-body approximation
    # person_height = 1.40
    # torso_height = 1.0
    # leg_height = 0.9
    # body_width = 0.45
    # body_depth = 0.28
    # head_size = 0.20

    # # Position person slightly offset from center table
    # person_center_x = 4.8
    # person_center_y = 2.5

    # px = (person_center_x - body_width/2, person_center_x + body_width/2)
    # py = (person_center_y - body_depth/2, person_center_y + body_depth/2)

    # # Torso
    # person.add_box(px, py, (leg_height, leg_height + torso_height))

    # # Head
    # head_radius = head_size / 2
    # head_center = [person_center_x, person_center_y, leg_height + torso_height + head_radius]
    # person.add_sphere(head_center, head_radius, resolution=16)

    # # Legs
    # leg_gap = 0.05
    # leg_width = (body_width - leg_gap) / 2

    # # Left leg
    # person.add_box(
    #     (person_center_x - body_width/2, person_center_x - body_width/2 + leg_width),
    #     py,
    #     (0, leg_height)
    # )

    # # Right leg
    # person.add_box(
    #     (person_center_x + body_width/2 - leg_width, person_center_x + body_width/2),
    #     py,
    #     (0, leg_height)
    # )

    # write_ply_simple(os.path.join(meshes_dir, 'person.ply'),
    #                  person.vertices, person.faces)

    # add_to_combined(person, MAT_CONCRETE)
    # print("✓ Person (standing)")

    # ---------------------------------------------------
    # Metallic Cube on Table
    # ---------------------------------------------------
    metallic_cube = MeshBuilder()
    
    # Cube dimensions: 20cm x 20cm x 20cm
    cube_size = 0.20
    
    # Position on center coffee table
    center_table_center_x = room_center_x  # 3.5
    center_table_center_y = room_center_y  # 2.5
    center_table_top_height = center_table_height  # 0.4
    
    cube_x = (center_table_center_x - cube_size/2, center_table_center_x + cube_size/2)
    cube_y = (center_table_center_y - cube_size/2, center_table_center_y + cube_size/2)
    cube_z = (center_table_top_height, center_table_top_height + 0.50)
    
    metallic_cube.add_box(cube_x, cube_y, cube_z)
    
    write_ply_simple(os.path.join(meshes_dir, 'metallic_cube.ply'),
                     metallic_cube.vertices, metallic_cube.faces)
    
    add_to_combined(metallic_cube, MAT_METAL)
    print("✓ Metallic cube on center coffee table")
    
    # Write combined PLY
    combined_path = os.path.join(output_dir, 'room_5x3x3_combined.ply')
    write_ply_combined(combined_path, combined_v, combined_f, combined_m)
    print(f"\n✅ Combined: {combined_path} ({len(combined_v)} vertices, {len(combined_f)} faces)")
    print(f"✅ Separate meshes in: {meshes_dir}/")

def create_scene_xml(output_path):
    """Generate scene XML with separate objects."""
    print(f"\nGenerating scene XML: {output_path}")
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<scene version="2.1.0">
    <emitter type="constant">
        <rgb name="radiance" value="1.0"/>
    </emitter>

    <!-- Materials -->
    <bsdf type="diffuse" id="mat_concrete">
        <rgb name="reflectance" value="0.5 0.5 0.5"/>
    </bsdf>
    <bsdf type="diffuse" id="mat_wood">
        <rgb name="reflectance" value="0.6 0.4 0.2"/>
    </bsdf>
    <bsdf type="diffuse" id="mat_glass">
        <rgb name="reflectance" value="0.1 0.1 0.9"/>
    </bsdf>
    <bsdf type="diffuse" id="mat_metal">
        <rgb name="reflectance" value="0.8 0.8 0.8"/>
    </bsdf>

    <!-- Shapes with Material References -->
    <shape type="ply" id="floor">
        <string name="filename" value="meshes_d/floor.ply"/>
        <ref id="mat_concrete"/>
    </shape>
    
    <shape type="ply" id="ceiling">
        <string name="filename" value="meshes_d/ceiling.ply"/>
        <ref id="mat_concrete"/>
    </shape>
    
    <shape type="ply" id="walls">
        <string name="filename" value="meshes_d/walls.ply"/>
        <ref id="mat_concrete"/>
    </shape>
    
    <shape type="ply" id="window">
        <string name="filename" value="meshes_d/window.ply"/>
        <ref id="mat_glass"/>
    </shape>
    
    <shape type="ply" id="door">
        <string name="filename" value="meshes_d/door.ply"/>
        <ref id="mat_wood"/>
    </shape>
    
    <shape type="ply" id="furniture">
        <string name="filename" value="meshes_d/furniture.ply"/>
        <ref id="mat_wood"/>
    </shape>
    
    # <shape type="ply" id="pillar">
    #     <string name="filename" value="meshes_d/pillar.ply"/>
    #     <ref id="mat_concrete"/>
    # </shape>
    
    <shape type="ply" id="furniture_center">
        <string name="filename" value="meshes_d/furniture_center.ply"/>
        <ref id="mat_wood"/>
    </shape>
    
    <shape type="ply" id="led_tv">
        <string name="filename" value="meshes_d/led_tv.ply"/>
        <ref id="mat_metal"/>
    </shape>
    
    <shape type="ply" id="metallic_cube">
        <string name="filename" value="meshes_d/metallic_cube.ply"/>
        <ref id="mat_metal"/>
    </shape>
</scene>
"""
    with open(output_path, 'w') as f:
        f.write(xml_content)
    print("✅ Scene XML created")

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    create_scene_meshes(project_dir)
    create_scene_xml(os.path.join(project_dir, "room_with_cube.xml"))
    print("\n--- Complete ---")
    print("Rectangular room: 7m x 5m x 3m")
    print("Room dimensions: x=0-7m, y=0-5m, z=0-3m")
    print("Combined PLY: room_5x3x3_combined.ply (for 3D viewers)")
    print("Separate PLYs: meshes_d/ (for Sionna RF simulation)")
    