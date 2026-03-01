from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import time
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from urllib.parse import quote_plus

username = "pomodorro"
password = ""

uri = f"mongodb+srv://{username}:{quote_plus(password)}@trivia-app.h6q9oau.mongodb.net/?appName=trivia-app"

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
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = users.find_one({"username": username})
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = username
            return redirect(url_for("index"))

        return "Invalid username or password", 401

    return """
    <h2>Login</h2>
    <form method="post">
      <label>Username</label><br>
      <input name="username" type="text" required><br><br>
      <label>Password</label><br>
      <input name="password" type="password" required><br><br>
      <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/signup">Sign up</a></p>
    """


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


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
