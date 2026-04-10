import re

path = "/home/ved/Ved/Project_1/RF-3DGS/gradient_descent_localization.py"
with open(path, "r") as f:
    orig = f.read()

part1 = """    print(f"--- Phase 0: Verifying Top Candidates with Native RF-3DGS Render Model ---")
    best_candidate_idx = 0
    best_candidate_loss = float('inf')
    
    for i, (pos, yaw) in enumerate(coarse_candidates):
        l = evaluate_loss_at(pos[0], pos[1], pos[2], yaw)
        print(f"  Candidate {i+1} [Pos={pos}, Yaw={yaw}°] -> Deep Model Loss: {l:.6f}")
        if l < best_candidate_loss:
            best_candidate_loss = l
            best_candidate_idx = i
            
    coarse_position, coarse_yaw_deg = coarse_candidates[best_candidate_idx]
    print(f"  --> SELECTED BEST STARTING CANDIDATE: Pos={coarse_position}, Yaw={coarse_yaw_deg}°\\n")

    current_pos = np.array(coarse_position)
    best_yaw_history = []
    
    MAX_DIRECTION_CHECKS = 7
    MIN_CONSECUTIVE_SAME_YAW = 5
    
    print(f"--- Phase 1: Finding Best Angle and Moving ---")
    for step in range(MAX_DIRECTION_CHECKS):
        best_loss = float('inf')
        best_yaw = 0
        
        for y in yaws_to_test:
            l = evaluate_loss_at(current_pos[0], current_pos[1], current_pos[2], y)
            if l < best_loss:
                best_loss = l
                best_yaw = y
                
        best_yaw_history.append(best_yaw)
        print(f"  Step {step+1}: best angle = {best_yaw}°, loss = {best_loss:.6f} at coords {current_pos}")
        
        # Check if direction has stabilized for 5 iterations
        if len(best_yaw_history) >= MIN_CONSECUTIVE_SAME_YAW and len(set(best_yaw_history[-MIN_CONSECUTIVE_SAME_YAW:])) == 1:
            print(f"  --> Direction established to {best_yaw}° (stable for {MIN_CONSECUTIVE_SAME_YAW} iterations). Proceeding to target optimization.")
            break
            
        def step_objective(pos):
            return evaluate_loss_at(pos[0], pos[1], pos[2], best_yaw)
            
        # Move slightly in that position's error basin (optimising Rx position keeping angle fixed)
        res = minimize(step_objective, current_pos, method='Nelder-Mead', options={'maxiter': 5, 'maxfev': 15})
        current_pos = res.x
        
    final_target_yaw = best_yaw_history[-1]
    
    print(f"\\n--- Phase 2: Full Rx Position Optimization (Fixed at Dir {final_target_yaw}°) ---")"""

part2 = """    MAX_DIRECTION_CHECKS = 7
    MIN_CONSECUTIVE_SAME_YAW = 5

    print(f"--- Phase 1: Exploring Best Angle and Error Basin for each Candidate ---")
    
    best_candidate_post_phase1_loss = float('inf')
    best_candidate_post_phase1_pos = None
    best_candidate_post_phase1_yaw = None
    
    for i, (pos, yaw) in enumerate(coarse_candidates):
        print(f"\\n  [Candidate {i+1}]: Initial Coords={pos}, Yaw={yaw}°")
        current_pos = np.array(pos)
        best_yaw_history = []
        best_loss = float('inf')
        best_yaw = yaw
        
        for step in range(MAX_DIRECTION_CHECKS):
            best_loss = float('inf')
            best_yaw = 0
            
            for y in yaws_to_test:
                l = evaluate_loss_at(current_pos[0], current_pos[1], current_pos[2], y)
                if l < best_loss:
                    best_loss = l
                    best_yaw = y
                    
            best_yaw_history.append(best_yaw)
            
            # Check if stabilized
            if len(best_yaw_history) >= MIN_CONSECUTIVE_SAME_YAW and len(set(best_yaw_history[-MIN_CONSECUTIVE_SAME_YAW:])) == 1:
                break
                
            def step_objective(p):
                return evaluate_loss_at(p[0], p[1], p[2], best_yaw)
                
            res = minimize(step_objective, current_pos, method='Nelder-Mead', options={'maxiter': 5, 'maxfev': 15})
            current_pos = res.x
            
        print(f"    -> After basin exploration: Coords=[{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}], Best Yaw={best_yaw}°, Deep Loss={best_loss:.6f}")
        
        if best_loss < best_candidate_post_phase1_loss:
            best_candidate_post_phase1_loss = best_loss
            best_candidate_post_phase1_pos = current_pos
            best_candidate_post_phase1_yaw = best_yaw

    print(f"\\n  --> WINNER CANDIDATE FOR PHASE 2: Coords=[{best_candidate_post_phase1_pos[0]:.4f}, {best_candidate_post_phase1_pos[1]:.4f}, {best_candidate_post_phase1_pos[2]:.4f}], Yaw={best_candidate_post_phase1_yaw}°, Loss={best_candidate_post_phase1_loss:.6f}")

    current_pos = np.array(best_candidate_post_phase1_pos)
    final_target_yaw = best_candidate_post_phase1_yaw

    print(f"\\n--- Phase 2: Full Rx Position Optimization (Fixed at Dir {final_target_yaw}°) ---")"""

if part1 in orig:
    res = orig.replace(part1, part2)
    with open(path, "w") as f:
        f.write(res)
    print("PATCH APPLIED SUCCESSFULLY")
else:
    print("PATCH FAILED TO FIND TARGET STRING")
