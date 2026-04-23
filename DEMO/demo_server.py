"""
demo_server.py
--------------
Lightweight Flask server that:
  - Serves the live visualisation at http://localhost:5000
  - Exposes /api/results  → returns current results.json
  - Exposes /api/status   → quick status check

Run:
    pip install flask
    python demo_server.py --results-file results.json --port 5000
"""

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import logging
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

RESULTS_FILE = "results.json"
pipeline_process = None

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).resolve().parent), "demo.html")

@app.route("/api/start", methods=["POST"])
def api_start():
    global pipeline_process
    if pipeline_process and pipeline_process.poll() is None:
        return jsonify({"status": "error", "message": "Pipeline already running"})
    
    cmd = [sys.executable, "demo_pipeline.py", "--waypoints", "waypoints.txt"]
    pipeline_process = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    global pipeline_process
    if pipeline_process and pipeline_process.poll() is None:
        pipeline_process.terminate()
        pipeline_process.wait()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})

@app.route("/api/results")
def api_results():
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"status": "waiting", "error": str(e), "true_path": [],
                        "pred_path": [], "errors": [], "rmse": None, "log": []})

@app.route("/api/status")
def api_status():
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        return jsonify({"status": data.get("status", "unknown"),
                        "points": len(data.get("true_path", []))})
    except Exception:
        return jsonify({"status": "waiting", "points": 0})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-file", default="results.json")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    # Resolve relative to this script so pipeline and server use the same path
    RESULTS_FILE = str(Path(__file__).resolve().parent / args.results_file)
    print(f"Demo server → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
    app.run(port=args.port, debug=False)