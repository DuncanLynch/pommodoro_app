from flask import Flask, render_template, jsonify
import time

app = Flask(__name__)
start_time = time.time()
duration = 60   # set your timer duration in seconds

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/_timer")
def timer():
    elapsed = time.time() - start_time
    remaining = max(0, duration - int(elapsed))
    return jsonify({"result": remaining})