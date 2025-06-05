import time
import serial
import threading
import re

# Global serial object
ser = None

# Store last execution time per command
last_command_time = {}

# Cooldown settings (in seconds)
command_cooldowns = {
    "s1go": 0.7,
    "s2go": 0.7,
    "s3go": 0.7,
    "s4go": 0.7,
    "s6": 60.0
}

# Latest scores and winner
latest_scores = {"DINO1": 0, "DINO2": 0, "DINO3": 0, "DINO4": 0}
latest_winner = None

def init(port, baudrate=9600):
    """
    Initialize the serial port.
    """
    global ser
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[SERIAL] Connected to {port} at {baudrate} baud")

        # Start background reader thread
        threading.Thread(target=read_serial_loop, daemon=True).start()

    except Exception as e:
        print(f"[SERIAL] Failed to connect: {e}")
        ser = None

def handleCommand(cmd, key_position=None, command_price=0):
    """
    Process incoming commands with cooldowns.
    """
    global last_command_time

    cmd = cmd.strip().lower()
    now = time.time()
    cooldown = command_cooldowns.get(cmd, 0.5)
    last_time = last_command_time.get(cmd, 0)

    if now - last_time < cooldown:
        print(f"[SERIAL] Ignored '{cmd}' (cooldown {cooldown}s, waited {now - last_time:.2f}s)")
        return

    last_command_time[cmd] = now

    if ser:
        try:
            ser.write((cmd + '\n').encode())
            print(f"[SERIAL] Sent: {cmd}")
        except Exception as e:
            print(f"[SERIAL] Error sending '{cmd}': {e}")
    else:
        print(f"[SERIAL] No serial connection. Could not send: {cmd}")

def read_serial_loop():
    """
    Background loop to read lines from serial and parse scores and winner.
    """
    global latest_scores, latest_winner
    while True:
        try:
            if ser and ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue

                print(f"[SERIAL] Received: {line}")

                # Parse scores
                if line.startswith("DINO1 SCORE"):
                    match = re.findall(r"(DINO\d) SCORE-(\d+)", line)
                    if match:
                        for dino, score in match:
                            latest_scores[dino] = int(score)
                        print(f"[SCORES] {latest_scores}")

                # Parse winner
                elif line.startswith("WINNER"):
                    match = re.search(r"WINNER - (DINO\d)", line)
                    if match:
                        latest_winner = match.group(1)
                        print(f"[WINNER] {latest_winner}")

                elif line == "RESET":
                    latest_scores = {"DINO1": 0, "DINO2": 0, "DINO3": 0, "DINO4": 0}
                    latest_winner = None
                    print("[RESET] Scores reset")

        except Exception as e:
            print(f"[SERIAL] Error reading serial: {e}")
        time.sleep(0.1)

def get_scores():
    return latest_scores.copy()

def get_winner():
    return latest_winner

def close():
    global ser
    if ser and ser.is_open:
        ser.close()
        print("[SERIAL] Serial connection closed")
