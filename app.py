import serial
import threading
import time
import re
from flask import Flask, render_template_string

app = Flask(__name__)

# Configuration
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
WINNING_SCORE = 100
RESET_DELAY = 15  # seconds

# Score state
scores = {
    "DINO1": 0,
    "DINO2": 0,
    "DINO3": 0,
    "DINO4": 0
}
winner = ""
reset_timer = None
reset_time_left = 0
lock = threading.Lock()

# Regex pattern for parsing Arduino lines
score_pattern = re.compile(
    r'DINO1 SCORE-(\d+)\s*, DINO2 SCORE-(\d+)\s*, DINO3 SCORE-(\d+)\s*, DINO4 SCORE-(\d+)'
)

def reset_scores():
    global scores, winner, reset_timer, reset_time_left
    with lock:
        print("[RESET] Resetting scores and winner")
        scores = {k: 0 for k in scores}
        winner = ""
        reset_timer = None
        reset_time_left = 0

def countdown_reset():
    global reset_time_left
    reset_time_left = RESET_DELAY
    while reset_time_left > 0:
        time.sleep(1)
        reset_time_left -= 1
    reset_scores()

def handle_line(line):
    global scores, winner, reset_timer
    match = score_pattern.search(line)
    if match:
        with lock:
            d1, d2, d3, d4 = map(int, match.groups())
            updated_scores = {
                "DINO1": d1,
                "DINO2": d2,
                "DINO3": d3,
                "DINO4": d4
            }
            # Only update if scores increase by 1 step (no skipping)
            for key in scores:
                if updated_scores[key] == scores[key] + 1:
                    scores[key] = updated_scores[key]

            # Check for winner
            if not winner:
                for key in scores:
                    if scores[key] >= WINNING_SCORE:
                        winner = key
                        print(f"[WINNER] {winner} reached {WINNING_SCORE}")
                        reset_timer = threading.Thread(target=countdown_reset)
                        reset_timer.start()

def read_serial():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[SERIAL] Received: {line}")
                    handle_line(line)
    except Exception as e:
        print(f"[Serial Error] {e}")

# Start serial reading thread
threading.Thread(target=read_serial, daemon=True).start()

# Overlay template with live update JS
overlay_template = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
        body {
            margin: 0;
            background: transparent;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            user-select: none;
        }
    </style>
    <script>
        // Refresh the body content every 500ms without reloading the page
        setInterval(() => {
            fetch(window.location.href)
                .then(res => res.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const newContent = doc.querySelector('body').innerHTML;
                    document.body.innerHTML = newContent;
                });
        }, 500);
    </script>
</head>
<body>
    {{ content }}
</body>
</html>
"""

@app.route("/overlay/<dino_id>")
def overlay_dino(dino_id):
    with lock:
        score = scores.get(dino_id.upper(), 0)
        return render_template_string(
            overlay_template,
            title=dino_id,
            content=f"{dino_id.upper()} SCORE - {score}"
        )

@app.route("/overlay/winner")
def overlay_winner():
    with lock:
        if winner:
            return render_template_string(
                overlay_template,
                title="WINNER",
                content=f"WINNER - {winner} , NEW GAME IN {reset_time_left} SECONDS"
            )
        else:
            return render_template_string(
                overlay_template,
                title="WINNER",
                content=""
            )

@app.route("/overlay/full")
def overlay_full():
    with lock:
        scores_line = ' , '.join(f"{k} SCORE-{v}" for k, v in scores.items())
        content = scores_line
        if winner:
            content += f"<br>WINNER - {winner} , NEW GAME IN {reset_time_left} SECONDS"
        return render_template_string(
            overlay_template,
            title="FULL",
            content=content
        )

if __name__ == "__main__":
    print("[Flask] Starting server on http://localhost:9152")
    app.run(host="0.0.0.0", port=9152)
