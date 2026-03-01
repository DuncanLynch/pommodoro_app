from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import time

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

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
    session['duration'] = duration
    return render_template("index.html")


@app.route("/start_timer", methods=['POST'])
def start_timer():
    session['timer_running'] = True
    session['start_time'] = time.time()
    return jsonify({"status": "Timer started"})


@app.route("/run_timer", methods=['GET'])
def run_timer():
    if 'timer_running' in session and session['timer_running']:
        elapsed = time.time() - session['start_time']
        remaining = max(0, session['duration'] - int(elapsed))
        
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        remaining = f"{minutes:02d}:{seconds:02d}"

        session['remaining'] = remaining
        return jsonify({"elapsed": remaining})


@app.route("/stop_timer", methods=['POST'])
def stop_timer():
    print("STOP TIMER")
    if 'timer_running' in session and session['timer_running']:
        session['timer_running'] = False

        # If you want the timer to continue instead of resetting, uncomment this:
        # session['duration'] = session['remaining']

        minutes = int(session['duration'] // 60)
        seconds = int(session['duration'] % 60)
        duration_formatted = f"{minutes:02d}:{seconds:02d}"
        return jsonify({"status": "Timer stopped", "duration": duration_formatted})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)