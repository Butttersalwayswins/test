import sys
import asyncio
import json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit, QLabel
)
from PyQt6.QtCore import pyqtSignal
import qasync
import websockets

API_HOST = "https://api.robotstreamer.com"  # adjust if needed

# Replace this function with your actual way to get chat/control websocket endpoints
async def get_ws_endpoints(robot_id):
    # For demo, just return hardcoded URLs - replace with your api calls
    return {
        "chat": f"wss://208-113-134-124.robotstreamer.com:8765/echo",
        "control": f"wss://208-113-134-124.robotstreamer.com:8865/echo"
    }

class LoginWindow(QWidget):
    logged_in = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.resize(300, 150)
        layout = QVBoxLayout()
        self.robot_id = QLineEdit()
        self.robot_id.setPlaceholderText("Robot ID")
        self.stream_key = QLineEdit()
        self.stream_key.setPlaceholderText("Stream Key")
        self.login_button = QPushButton("Login")
        self.status_label = QLabel("")
        layout.addWidget(self.robot_id)
        layout.addWidget(self.stream_key)
        layout.addWidget(self.login_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.login_button.clicked.connect(self.try_login)
    def try_login(self):
        rid = self.robot_id.text().strip()
        key = self.stream_key.text().strip()
        if rid and key:
            self.logged_in.emit(rid, key)
        else:
            self.status_label.setText("Please enter both Robot ID and Stream Key.")

class MainWindow(QWidget):
    def __init__(self, robot_id, stream_key):
        super().__init__()
        self.setWindowTitle(f"RobotStreamer Chat - Robot {robot_id}")
        self.resize(600, 400)

        self.robot_id = robot_id
        self.stream_key = stream_key

        layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.send_button = QPushButton("Send")

        layout.addWidget(self.chat_display)
        layout.addWidget(self.chat_input)
        layout.addWidget(self.send_button)
        self.setLayout(layout)

        self.send_button.clicked.connect(self.send_chat_message)
        self.chat_input.returnPressed.connect(self.send_chat_message)

        self.ws_chat = None
        self.ws_control = None

    async def connect_websockets(self):
        endpoints = await get_ws_endpoints(self.robot_id)
        chat_url = endpoints["chat"]
        control_url = endpoints["control"]

        # Connect to chat websocket
        self.ws_chat = await websockets.connect(chat_url)
        await self.ws_chat.send(json.dumps({
            "type": "connect",
            "stream_key": self.stream_key,
            "robot_id": self.robot_id
        }))
        self.chat_display.append(f"Connected to chat websocket: {chat_url}")

        # Connect to control websocket
        self.ws_control = await websockets.connect(control_url)
        await self.ws_control.send(json.dumps({
            "type": "robot_connect",
            "robot_id": self.robot_id,
            "stream_key": self.stream_key
        }))
        self.chat_display.append(f"Connected to control websocket: {control_url}")

        # Start listening loops concurrently
        asyncio.create_task(self.receive_chat_messages())
        asyncio.create_task(self.receive_control_messages())

    async def receive_chat_messages(self):
        try:
            async for message in self.ws_chat:
                data = json.loads(message)

                # Filter out RS_PING keepalive messages
                if data.get("type") == "RS_PING" or data.get("command") == "RS_PING":
                    continue

                # Try different places to get username
                username = data.get("user") or data.get("username") or "Unknown"

                # Sometimes username might be nested inside "user" dict
                if isinstance(username, dict):
                    username = username.get("name") or username.get("username") or "Unknown"

                if "message" in data:
                    self.chat_display.append(f"{username}: {data['message']}")
        except Exception as e:
            self.chat_display.append(f"Chat websocket error: {e}")

    async def receive_control_messages(self):
        try:
            async for message in self.ws_control:
                data = json.loads(message)

                # Filter out RS_PING keepalive messages from control too
                if data.get("type") == "RS_PING" or data.get("command") == "RS_PING":
                    continue

                self.chat_display.append(f"Control: {data}")
        except Exception as e:
            self.chat_display.append(f"Control websocket error: {e}")

    def send_chat_message(self):
        msg = self.chat_input.text().strip()
        if not msg or self.ws_chat is None:
            return
        message_obj = {
            "type": "message",
            "message": msg,
            "stream_key": self.stream_key,
            "robot_id": self.robot_id
        }
        asyncio.create_task(self.ws_chat.send(json.dumps(message_obj)))
        self.chat_display.append(f"You: {msg}")
        self.chat_input.clear()

async def main():
    app = QApplication(sys.argv)

    login = LoginWindow()
    login.show()

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    future = asyncio.Future()
    def on_logged_in(rid, key):
        future.set_result((rid, key))
    login.logged_in.connect(on_logged_in)

    robot_id, stream_key = await future
    login.close()

    main_window = MainWindow(robot_id, stream_key)
    main_window.show()

    await main_window.connect_websockets()

    with loop:
        await asyncio.Future()  # Keep app running forever

if __name__ == "__main__":
    import qasync
    qasync.run(main())
