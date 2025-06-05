import time
import serial
import threading
import re

# Global serial object
ser = None

# Cooldown settings (in seconds)
command_cooldowns = {
    "s1go": 0.7,
    "s2go": 0.7,
    "s3go": 0.7,
    "s4go": 0.7,
    "s6": 60.0
}

# Store last execution time per command
last_command_time = {}

# Shared data for overlay
current_scores = {
    "DINO1": 0,
    "DINO2": 0,
    "DINO3": 0,
    "DINO4": 0
}
current_winner = None
winner_timestamp = 0

def init(port, baudrate=9600):
    """
    Initialize the serial port and start the reader thread.
    """
    global ser
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[SERIAL] Connected to {port} at {baudrate} baud")
        threading.Thread(target=read_serial_loop, daemon=True).start()
    except Exception as e:
        print(f"[SERIAL] Failed to connect: {e}")
        ser = None

def handleCommand(cmd, key_position=None, command_price=0):
    """
    Send servo command if cooldown allows.
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
    Read from serial and parse score/winner messages.
    """
    global current_scores, current_winner, winner_timestamp

    while True:
        try:
            if ser and ser.in_waiting:
                line = ser.readline().decode().strip()
                if not line:
                    continue

                print(f"[SERIAL] Received: {line}")

                if "DINO1 SCORE" in line:
                    match = re.findall(r"DINO\d SCORE-(\d+)", line)
                    if len(match) == 4:
                        current_scores = {
                            "DINO1": int(match[0]),
                            "DINO2": int(match[1]),
                            "DINO3": int(match[2]),
                            "DINO4": int(match[3])
                        }

                elif "WINNER -" in line:
                    winner_match = re.search(r"WINNER - (DINO\d)", line)
                    if winner_match:
                        current_winner = winner_match.group(1)
                        winner_timestamp = time.time()

            # Reset winner after 15 seconds
            if current_winner and (time.time() - winner_timestamp > 15):
                current_winner = None
                current_scores = {k: 0 for k in current_scores}

            time.sleep(0.1)
        except Exception as e:
            print(f"[SERIAL] Read error: {e}")
            time.sleep(1)

def get_scores():
    return current_scores.copy()

def get_winner():
    return current_winner
