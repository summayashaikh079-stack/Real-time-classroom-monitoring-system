"""
app.py
------
Real-Time Classroom Monitoring System - Flask backend.
"""

import os
import threading
import time
from functools import wraps

import cv2
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session

from utils.face_utils import AttendanceSystem
from utils.engagement_utils import EngagementTracker
from utils.phone_utils import PhoneDetector
from utils.db import init_db, create_user, verify_user, get_user_by_id

app = Flask(__name__)
app.secret_key = "change-this-to-something-random-before-deploying"

init_db()

# ---- Shared state ----
lock = threading.Lock()
latest_engagement_faces = []
latest_class_avg = 0
latest_attendance_summary = {}
latest_phone_count = 0
CAMERA_INDEX = 0


# ---------------- Auth helpers ----------------

def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            return render_template("register.html", error="Please fill in all fields.")
        if password != confirm:
            return render_template("register.html", error="Passwords do not match.")
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.")

        ok, message = create_user(username, password, role="teacher")
        if ok:
            return render_template("register.html", success="Account created! You can now sign in.")
        return render_template("register.html", error=message)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = verify_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("index"))

        return render_template("login.html", error="Incorrect username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- Camera / dashboard ----------------

def generate_frames():
    global latest_engagement_faces, latest_class_avg, latest_attendance_summary, latest_phone_count

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    attendance_system = AttendanceSystem()
    engagement_tracker = EngagementTracker()
    phone_detector = PhoneDetector()
    frame_count = 0
    last_phone_count = 0
    hold = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # 1. Attendance (every 3rd frame)
            if frame_count % 3 == 0:
                frame, _ = attendance_system.process_frame(frame)
                with lock:
                    latest_attendance_summary = attendance_system.get_summary()

            # 2. Phone Detection
            frame, current_phones = phone_detector.process_frame(frame)

            if current_phones > 0:
                last_phone_count = current_phones
                hold = 50
            elif hold > 0:
                hold -= 1
            else:
                last_phone_count = 0

            with lock:
                latest_phone_count = last_phone_count

            # 3. Engagement
            frame, face_results, class_avg = engagement_tracker.process_frame(frame)
            with lock:
                latest_engagement_faces = face_results
                latest_class_avg = class_avg

            frame_count += 1

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.01)
    finally:
        cap.release()
        engagement_tracker.close()


@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session.get("username"))


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
@login_required
def api_stats():
    with lock:
        return jsonify({
            "class_avg_attention": latest_class_avg,
            "faces": latest_engagement_faces,
            "attendance": latest_attendance_summary,
            "phone_count": latest_phone_count,
            "timestamp": time.time(),
        })


@app.route("/register_student", methods=["GET", "POST"])
@login_required
def register_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        photo = request.files.get("photo")

        if not name or not photo:
            return render_template("register_student.html", error="Name and photo are required.")

        safe_name = name.replace(" ", "-")
        ext = photo.filename.rsplit(".", 1)[-1].lower()
        if ext not in ["jpg", "jpeg", "png"]:
            return render_template("register_student.html", error="Only JPG, JPEG, PNG allowed.")

        filename = f"{safe_name}.{ext}"
        os.makedirs("known_faces", exist_ok=True)
        save_path = os.path.join("known_faces", filename)
        photo.save(save_path)

        return render_template("register_student.html", success=f"Student '{name}' registered successfully!")

    return render_template("register_student.html")


if __name__ == "__main__":
    print("Starting Classroom Monitoring System...")
    print("Put student photos in /known_faces (filename = student name).")
    print("Open http://127.0.0.1:5000 once the server starts.")
    app.run(debug=False, threaded=True)