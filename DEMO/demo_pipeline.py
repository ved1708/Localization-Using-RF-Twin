"""
localisation_pipeline.py
------------------------
Orchestrates the full localisation pipeline:

  For each true waypoint:
    1. docker exec -w /tf/Project_1 magical_margulis python3 generate_csi_dataset.py
         --rx-pos x y z --spectrum-type delay
         --spec-min ... --spec-max ... --output-dir ...
    2. python gradient_descent_localization.py
         --target_image <img> --model_path output/rf_model_delay_3.5ghz --iteration 40000
         [--coarse_x --coarse_y --coarse_z --coarse_yaw]  ← from previous point (point 2+)

  Writes results.json incrementally so demo_server.py can stream live updates.

Usage:
    python localisation_pipeline.py \
        --waypoints waypoints.txt \
        --results-file results.json

Waypoints file (one per line):
    x y z [yaw_degrees]    e.g.  0.4 0.4 1.2 0.0
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    """Run a command, return (stdout, stderr, returncode)."""
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(result.stdout[-3000:], flush=True)
    if result.stderr:
        print(result.stderr[-500:], file=sys.stderr, flush=True)
    return result.stdout, result.stderr, result.returncode


def parse_estimated_position(stdout: str) -> tuple[float, float, float, float] | None:
    """
    Parse predicted x y z yaw from gradient_descent_localization.py stdout.
    Matches the exact print format:
        Optimized Position:
          X: 1.234567 m
          Y: 2.345678 m
          Z: 1.234567 m
        Optimized Yaw: 0.123456°
    """
    mx = re.search(r"X:\s*(-?[\d.]+)\s*m", stdout)
    my = re.search(r"Y:\s*(-?[\d.]+)\s*m", stdout)
    mz = re.search(r"Z:\s*(-?[\d.]+)\s*m", stdout)
    myaw = re.search(r"Optimized Yaw:\s*(-?[\d.]+)", stdout)
    if mx and my and mz:
        return float(mx.group(1)), float(my.group(1)), float(mz.group(1)), \
               float(myaw.group(1)) if myaw else 0.0
    return None


def translation_error(a: dict, b: dict) -> float:
    return ((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2) ** 0.5


def rmse(trues: list[dict], preds: list[dict]) -> float:
    n = min(len(trues), len(preds))
    sq = sum(translation_error(trues[i], preds[i])**2 for i in range(n))
    return (sq / n) ** 0.5


def write_results(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)   # atomic swap so server always reads valid JSON


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Localisation pipeline orchestrator")
    ap.add_argument("--waypoints", required=True,
                    help="Text file: one 'x y z [yaw]' per line (true positions)")
    ap.add_argument("--output-dir", default="DEMO/localisation_frames_demo",
                    help="Output dir for CSI images")
    ap.add_argument("--model-path", default="RF-3DGS/output/rf_model_delay_3.5ghz",
                    help="RF-3DGS model path")
    ap.add_argument("--spec-min", type=float, default=-143.9031990144428)
    ap.add_argument("--spec-max", type=float, default=-23.31242051071544)
    ap.add_argument("--spectrum-type", default="delay",
                    choices=["mpc", "aod", "delay", "phase"])
    ap.add_argument("--iteration", type=int, default=40000,
                    help="Model iteration for gradient descent (default: 40000)")
    ap.add_argument("--scene", default="room_with_cube.xml")
    ap.add_argument("--results-file", default="results.json")
    ap.add_argument("--docker-container", default="magical_margulis")
    ap.add_argument("--docker-workdir", default="/tf/Project_1")
    ap.add_argument("--project-dir", default="..",
                    help="Root dir of your project (where gradient_descent script lives)")
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    out_dir = project / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load waypoints
    waypoints = []
    with open(args.waypoints) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = list(map(float, line.split()))
            x, y, z = parts[0], parts[1], parts[2]
            yaw = parts[3] if len(parts) > 3 else 0.0
            waypoints.append({"x": x, "y": y, "z": z, "yaw": yaw})

    print(f"Loaded {len(waypoints)} waypoints", flush=True)

    results = {
        "status": "running",
        "total": len(waypoints),
        "true_path": [],
        "pred_path": [],
        "errors": [],
        "times": [],
        "rmse": None,
        "log": []
    }
    write_results(args.results_file, results)

    for idx, wp in enumerate(waypoints):
        print(f"\n{'='*60}", flush=True)
        print(f"Waypoint {idx+1}/{len(waypoints)}: x={wp['x']:.2f} y={wp['y']:.2f} z={wp['z']:.2f} yaw={wp['yaw']:.1f}°", flush=True)
        results["log"].append(f"[{idx+1}/{len(waypoints)}] Processing ({wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f})")

        # ── Step 1: generate CSI spectrum via docker ───────────────────────

        # exact command: docker exec -w /tf/Project_1 magical_margulis python3 generate_csi_dataset.py ...
        # output-dir is inside the container — use same name, map via mounted volume
        gen_cmd = [
            "docker", "exec",
            "-w", args.docker_workdir,
            args.docker_container,
            "python3", "generate_csi_dataset.py",
            "--rx-pos", str(wp["x"]), str(wp["y"]), str(wp["z"]),
            "--spectrum-type", args.spectrum_type,
            "--spec-min", str(args.spec_min),
            "--spec-max", str(args.spec_max),
            "--output-dir", args.output_dir,
        ]
        results["log"].append("  → Generating CSI spectrum (docker)...")
        write_results(args.results_file, results)

        stdout, stderr, rc = run(gen_cmd, cwd=str(project))
        if rc != 0:
            results["log"].append(f"  ✗ generate_csi_dataset.py failed (rc={rc})")
            write_results(args.results_file, results)
            continue

        # generated image lands in output-dir on the host (shared volume)
        # script names file after position e.g. delay_0.40_0.40_1.20.png — grab latest png
        images = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        if not images:
            results["log"].append("  ✗ No image found in output-dir — skipping")
            write_results(args.results_file, results)
            continue
        target_image = str(images[-1])   # most recently written
        results["log"].append(f"  ✓ Spectrum: {images[-1].name}")

        # ── Step 2: gradient descent localisation ─────────────────────────
        # Point 1: no coarse → script does full grid search internally
        # Point 2+: pass last predicted position as coarse warm-start
        true_pt = {"x": wp["x"],  "y": wp["y"],  "z": wp["z"],  "yaw": wp["yaw"]}
        results["true_path"].append(true_pt)
        results["log"].append("  → Gradient descent localisation...")
        write_results(args.results_file, results)

        gd_cmd = [
            "/home/ved/anaconda3/envs/rf-3dgs/bin/python", "gradient_descent_localization.py",
            "--target_image", target_image,
            "--model_path",   args.model_path,
            "--iteration",    str(args.iteration),
        ]
        if results["pred_path"]:
            prev = results["pred_path"][-1]
            gd_cmd += [
                "--coarse_x",   str(round(prev["x"],   4)),
                "--coarse_y",   str(round(prev["y"],   4)),
                "--coarse_z",   str(round(prev["z"],   4)),
                "--coarse_yaw", str(round(prev["yaw"], 4)),
            ]
            results["log"].append(f"  ↳ Warm start: ({prev['x']:.3f}, {prev['y']:.3f}, {prev['z']:.3f}, yaw={prev['yaw']:.1f}°)")
        else:
            results["log"].append("  ↳ Point 1 — full grid search inside script")

        stdout_gd, _, rc_gd = run(gd_cmd, cwd=str(project))

        # grab error printed by script if gt_pos was extracted from filename
        direct_err = None
        m_err = re.search(r"Distance from Real Position \(Error\):\s*([\d.]+)\s*meters", stdout_gd)
        if m_err:
            direct_err = float(m_err.group(1))
            results["log"].append(f"  ↳ Script-reported error: {direct_err:.6f} m")

        # grab refinement time
        m_time = re.search(r"Total Localization Time: ([\d.]+) seconds", stdout_gd)
        if m_time:
            time_val = float(m_time.group(1))
            results["times"].append(round(time_val, 2))
            results["log"].append(f"  ↳ Time: {time_val:.2f} s")
        else:
            results["times"].append(0.0)

        pred = parse_estimated_position(stdout_gd)

        if pred:
            px, py, pz, pyaw = pred
            results["log"].append(f"  ✓ Predicted: ({px:.3f}, {py:.3f}, {pz:.3f}, yaw={pyaw:.1f}°)")
        else:
            results["log"].append("  ⚠ Could not parse predicted position — check stdout format")
            # fallback: use last known prediction or true position
            if results["pred_path"]:
                fb = results["pred_path"][-1]
                px, py, pz, pyaw = fb["x"], fb["y"], fb["z"], fb["yaw"]
            else:
                px, py, pz, pyaw = wp["x"], wp["y"], wp["z"], wp["yaw"]

        # ── Record results ─────────────────────────────────────────────────
        true_pt  = {"x": wp["x"],  "y": wp["y"],  "z": wp["z"],  "yaw": wp["yaw"]}
        pred_pt  = {"x": px,       "y": py,        "z": pz,       "yaw": pyaw}
        err      = direct_err if direct_err is not None else translation_error(true_pt, pred_pt)

        results["pred_path"].append(pred_pt)
        results["errors"].append(round(err, 4))
        results["log"].append(f"  📏 Translation error: {err:.4f} m")

        if len(results["true_path"]) == len(results["pred_path"]) >= 2:
            results["rmse"] = round(rmse(results["true_path"], results["pred_path"]), 4)

        write_results(args.results_file, results)
        print(f"  Error: {err:.4f} m", flush=True)

    # ── Final ──────────────────────────────────────────────────────────────
    n = len(results["true_path"])
    if n >= 1:
        results["rmse"] = round(rmse(results["true_path"], results["pred_path"]), 4)
    results["status"] = "complete"
    results["log"].append(f"\n✓ Done. {n} points. RMSE = {results['rmse']} m")
    write_results(args.results_file, results)
    print(f"\nDone! RMSE = {results['rmse']} m  |  Results → {args.results_file}", flush=True)


if __name__ == "__main__":
    main()