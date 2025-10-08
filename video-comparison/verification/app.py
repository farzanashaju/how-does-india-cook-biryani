from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

# Configuration
DATA_FILE = "" # Completed video differentation data
FRAMES_DIR = "" # A dir containing all the extracted frames ( a script in ../scripts/ should have done that / the frames used in stages 1, 2)
LABELS_DIR = "labels"
STATIC_DIR = "static"

# Ensure directories exist
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_data():
    """Load the comparison data from JSON file."""
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def format_frame_name(video_url: str, frame_number: int) -> str:
    """Generate frame name from video URL and frame number."""
    video_filename = Path(video_url).stem
    return f"{video_filename}_frame_{frame_number:06d}.png"


def get_frame_paths(clip_data):
    """Extract all unique frame paths from clip data."""
    frames = set()
    for frame_range in clip_data["retrieval_frames"].values():
        for frame_num in frame_range:
            frame_name = format_frame_name(clip_data["url"], frame_num)
            frames.add(frame_name)
    return sorted(list(frames))


def get_labeled_items():
    """Get all labeled comparison-difference pairs."""
    labeled = set()
    for filename in os.listdir(LABELS_DIR):
        if filename.endswith("_label.json"):
            parts = filename.replace("_label.json", "").split("_")
            if len(parts) >= 2:
                diff_id = parts[-1]
                comp_id = "_".join(parts[:-1])
                labeled.add(f"{comp_id}_{diff_id}")
    return labeled


def get_label_stats():
    """Calculate labeling statistics by winner class."""
    stats = {
        "total": 0,
        "labeled": 0,
        "difference_detected": {"total": 0, "labeled": 0, "correct": 0, "incorrect": 0},
        "no_difference": {"total": 0, "labeled": 0, "correct": 0, "incorrect": 0},
        "error_inconclusive": {"total": 0, "labeled": 0, "correct": 0, "incorrect": 0},
    }

    data = load_data()
    labeled_items = {}

    # Load all labels
    for filename in os.listdir(LABELS_DIR):
        if filename.endswith("_label.json"):
            with open(os.path.join(LABELS_DIR, filename), "r") as f:
                label_data = json.load(f)
                key = f"{label_data['comparison_id']}_{label_data['difference_id']}"
                labeled_items[key] = label_data["label"]

    # Count totals and correctness by grouped class
    for comp_id, comp_data in data.items():
        for diff_id, diff_data in comp_data["proposed_differences"].items():
            winner = diff_data["winner"].upper()
            stats["total"] += 1

            # Group winners into categories
            if winner in ["A", "B"]:
                category = "difference_detected"
            elif winner == "C":
                category = "no_difference"
            else:  # D
                category = "error_inconclusive"

            stats[category]["total"] += 1

            key = f"{comp_id}_{diff_id}"
            if key in labeled_items:
                stats["labeled"] += 1
                stats[category]["labeled"] += 1

                if labeled_items[key] == "correct":
                    stats[category]["correct"] += 1
                else:
                    stats[category]["incorrect"] += 1

    return stats


def get_random_unlabeled_item():
    """Get a random unlabeled comparison-difference pair."""
    data = load_data()
    labeled = get_labeled_items()

    unlabeled = []
    for comp_id, comp_data in data.items():
        for diff_id in comp_data["proposed_differences"].keys():
            key = f"{comp_id}_{diff_id}"
            if key not in labeled:
                unlabeled.append((comp_id, diff_id))

    if not unlabeled:
        return None, None

    return random.choice(unlabeled)


def save_label(comparison_id, difference_id, label, username="anonymous"):
    """Save a label to JSON file."""
    timestamp = datetime.now().isoformat()
    label_data = {
        "comparison_id": comparison_id,
        "difference_id": difference_id,
        "label": label,
        "username": username,
        "timestamp": timestamp,
    }

    filename = f"{comparison_id}_{difference_id}_label.json"
    filepath = os.path.join(LABELS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(label_data, f, indent=2)

    return True


@app.route("/")
def index():
    """Main labeling page - shows random unlabeled item."""
    comp_id, diff_id = get_random_unlabeled_item()

    if not comp_id:
        return render_template("complete.html")

    data = load_data()
    comp_data = data[comp_id]
    diff_data = comp_data["proposed_differences"][diff_id]

    # Get frames for both clips
    clip1_frames = get_frame_paths(comp_data["clips"]["1"])
    clip2_frames = get_frame_paths(comp_data["clips"]["2"])

    context = {
        "comparison_id": comp_id,
        "difference_id": diff_id,
        "action_class": comp_data["action_class"],
        "difference_name": diff_data["name"],
        "query_string": diff_data["query_string"],
        "winner": diff_data["winner"],
        "winner_reason": diff_data["winner_reason"],
        "confidence": diff_data["confidence"],
        "clip1_type": comp_data["clips"]["1"]["type"],
        "clip2_type": comp_data["clips"]["2"]["type"],
        "clip1_frames": clip1_frames,
        "clip2_frames": clip2_frames,
    }

    return render_template("label.html", **context)


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    """Serve frame images."""
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/api/label", methods=["POST"])
def api_label():
    """API endpoint to save labels."""
    data = request.json

    try:
        save_label(
            data["comparison_id"],
            data["difference_id"],
            data["label"],
            data.get("username", "anonymous"),
        )

        return jsonify({"success": True, "message": "Label saved successfully"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Get detailed labeling statistics."""
    return jsonify(get_label_stats())


@app.route("/api/skip")
def api_skip():
    """Get next random unlabeled item."""
    comp_id, diff_id = get_random_unlabeled_item()
    if comp_id:
        return jsonify({"next_url": f"/goto/{comp_id}/{diff_id}"})
    return jsonify({"complete": True})


@app.route("/goto/<comparison_id>/<difference_id>")
def goto_item(comparison_id, difference_id):
    """Navigate to specific item."""
    data = load_data()

    if (
        comparison_id not in data
        or difference_id not in data[comparison_id]["proposed_differences"]
    ):
        return "Item not found", 404

    comp_data = data[comparison_id]
    diff_data = comp_data["proposed_differences"][difference_id]

    clip1_frames = get_frame_paths(comp_data["clips"]["1"])
    clip2_frames = get_frame_paths(comp_data["clips"]["2"])

    context = {
        "comparison_id": comparison_id,
        "difference_id": difference_id,
        "action_class": comp_data["action_class"],
        "difference_name": diff_data["name"],
        "query_string": diff_data["query_string"],
        "winner": diff_data["winner"],
        "winner_reason": diff_data["winner_reason"],
        "confidence": diff_data["confidence"],
        "clip1_type": comp_data["clips"]["1"]["type"],
        "clip2_type": comp_data["clips"]["2"]["type"],
        "clip1_frames": clip1_frames,
        "clip2_frames": clip2_frames,
    }

    return render_template("label.html", **context)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
