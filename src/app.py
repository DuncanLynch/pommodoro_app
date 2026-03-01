from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import time

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

start_time = time.time()
duration = 60   # set your timer duration in seconds

# Demo credentials; replace with real user storage later.
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session["user"] = username
            return redirect(url_for("index"))

        return "Invalid username or password", 401

    return """
    <h2>Login</h2>
    <form method=\"post\">
      <label>Username</label><br>
      <input name=\"username\" type=\"text\" required><br><br>
      <label>Password</label><br>
      <input name=\"password\" type=\"password\" required><br><br>
      <button type=\"submit\">Login</button>
    </form>
    """


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/_timer")
def timer():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    elapsed = time.time() - start_time
    remaining = max(0, duration - int(elapsed))
    return jsonify({"result": remaining})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
