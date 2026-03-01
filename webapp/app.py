from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import time
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import quote_plus
from bson import ObjectId

envfile = open("env.txt", "r")
uri = str(envfile.read())   

client = MongoClient(uri, serverSelectionTimeoutMS=8000)

# Test connection
client.admin.command("ping")
print("Connected successfully!")

print(client.list_database_names())

db = client["daniel_hogan"]
print(db.list_collection_names())

print("Collections:", db.list_collection_names())

users = db["users"]                
users.create_index("username", unique=True)


for col_name in db.list_collection_names():
    col = db[col_name]
    print(f"\n--- {col_name} sample ---")
    print(col.find_one())

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

start_time = time.time()
duration = 60   # set your timer duration in seconds

# Demo credentials; replace with real user storage later.
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"

PUBLIC_ENDPOINTS = {"login", "signup", "static"}


@app.before_request
def require_auth_for_protected_routes():
    endpoint = request.endpoint
    if endpoint in PUBLIC_ENDPOINTS:
        return None

    if "user" not in session:
        return redirect(url_for("login"))

    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        if request.is_json:
            return jsonify({"message": "Already logged in", "username": session["user"]}), 200
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        if request.is_json:
            body = request.get_json(silent=True) or {}
            username = str(body.get("username", "")).strip()
            password = str(body.get("password", ""))
        else:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

        user = users.find_one({"username": username})
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = username
            if request.is_json:
                return jsonify({"message": "Login successful", "username": username}), 200
            return redirect(url_for("index"))

        if request.is_json:
            return jsonify({"error": "Invalid username or password"}), 401
        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Basic validation
        if len(username) < 3:
            return "Username must be at least 3 characters.", 400
        if len(password) < 8:
            return "Password must be at least 8 characters.", 400

        doc = {
            "username": username,
            "password_hash": generate_password_hash(password),  # salted hash
            "created_at": time.time(),
        }

        try:
            users.insert_one(doc)
        except DuplicateKeyError:
            return "Username already taken.", 409

        # auto-login after signup (optional)
        session["user"] = username
        return redirect(url_for("index"))

    return """
    <h2>Sign up</h2>
    <form method="post">
      <label>Username</label><br>
      <input name="username" type="text" required><br><br>
      <label>Password</label><br>
      <input name="password" type="password" required><br><br>
      <button type="submit">Create account</button>
    </form>
    <p>Already have an account? <a href="/login">Log in</a></p>
    """

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/_timer")
def timer():
    elapsed = time.time() - start_time
    remaining = max(0, duration - int(elapsed))
    return jsonify({"result": remaining})


@app.route("/start_session", methods=["POST"])
def start_session():
    username = session["user"]
    now_utc = datetime.now(timezone.utc)
    session_id = str(ObjectId())

    study_session = {
        "session_id": session_id,
        "started_at": now_utc,
        "started_at_iso": now_utc.isoformat(),
        "started_at_epoch": int(now_utc.timestamp()),
        "status": "started",
    }

    result = users.update_one(
        {"username": username},
        {"$push": {"study_sessions": study_session}},
    )

    if result.matched_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify(
        {
            "message": "Study session started",
            "username": username,
            "session_id": session_id,
            "started_at": study_session["started_at_iso"],
        }
    ), 201


@app.route("/end_session", methods=["POST"])
def end_session():
    username = session["user"]
    now_utc = datetime.now(timezone.utc)
    now_epoch = int(now_utc.timestamp())

    user = users.find_one({"username": username}, {"study_sessions": 1})
    if not user:
        return jsonify({"error": "User not found"}), 404

    study_sessions = user.get("study_sessions", [])
    active_session = None
    for s in reversed(study_sessions):
        if s.get("status") == "started" and "ended_at_epoch" not in s:
            active_session = s
            break

    if not active_session:
        return jsonify({"error": "No active session found"}), 404

    started_at_epoch = active_session.get("started_at_epoch")
    if started_at_epoch is None:
        return jsonify({"error": "Active session missing start time"}), 400

    total_seconds = max(0, now_epoch - int(started_at_epoch))

    result = users.update_one(
        {"username": username, "study_sessions.session_id": active_session["session_id"]},
        {
            "$set": {
                "study_sessions.$.ended_at": now_utc,
                "study_sessions.$.ended_at_iso": now_utc.isoformat(),
                "study_sessions.$.ended_at_epoch": now_epoch,
                "study_sessions.$.total_session_seconds": total_seconds,
                "study_sessions.$.status": "ended",
            }
        },
    )

    if result.matched_count == 0:
        return jsonify({"error": "Active session no longer available"}), 409

    return jsonify(
        {
            "message": "Study session ended",
            "username": username,
            "session_id": active_session["session_id"],
            "ended_at": now_utc.isoformat(),
            "total_session_seconds": total_seconds,
        }
    ), 200


@app.route("/stats")
def stats():
    username = session["user"]
    user = users.find_one({"username": username}, {"study_sessions": 1, "_id": 0})
    if not user:
        return "User not found", 404

    all_sessions = user.get("study_sessions", [])
    completed_sessions = [
        s
        for s in all_sessions
        if s.get("status") == "ended" and s.get("total_session_seconds") is not None
    ]
    completed_sessions.sort(key=lambda s: s.get("ended_at_epoch", 0), reverse=True)
    latest_10 = completed_sessions[:10]

    total_seconds = sum(int(s.get("total_session_seconds", 0)) for s in latest_10)
    count = len(latest_10)
    avg_seconds = int(total_seconds / count) if count else 0

    def format_seconds(seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    rows = []
    for s in latest_10:
        rows.append(
            {
                "session_id": s.get("session_id", ""),
                "started_at": s.get("started_at_iso", "N/A"),
                "ended_at": s.get("ended_at_iso", "N/A"),
                "duration_seconds": int(s.get("total_session_seconds", 0)),
                "duration_hms": format_seconds(s.get("total_session_seconds", 0)),
            }
        )

    return render_template(
        "stats.html",
        username=username,
        sessions=rows,
        session_count=count,
        total_seconds=total_seconds,
        total_hms=format_seconds(total_seconds),
        avg_seconds=avg_seconds,
        avg_hms=format_seconds(avg_seconds),
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
