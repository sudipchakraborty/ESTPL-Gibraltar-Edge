import socketio
import threading
from collections import deque

class SocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.sio = socketio.Client(reconnection=True,reconnection_attempts=0,reconnection_delay=2)
        self._stop_event = threading.Event()
        self._pending_inspections = deque(maxlen=500)
        self._pending_lock = threading.Lock()
        self._register_events()
    #########################
    def _register_events(self):
        @self.sio.event
        def connect():
            print("Socket.IO connected " "to Node.js backend" )
            self._flush_pending_inspections()
        @self.sio.event
        def disconnect():
            print( "Socket.IO disconnected " "from Node.js backend" )
        @self.sio.event
        def connect_error(error):
            print("Socket.IO connection error:", error)
    ############################################
    def connect(self):
        self._stop_event.clear()
        while not self._stop_event.is_set():
            if self.sio.connected:
                self._stop_event.wait(1.0)
                continue
            try:
                print("Connecting Socket.IO to:",self.server_url)
                self.sio.connect(
                    self.server_url,
                    transports=["websocket","polling"],
                    wait_timeout=5,
                )
            except Exception as error:
                if not self._stop_event.is_set():
                    print("Unable to connect Socket.IO:", error)
                    self._stop_event.wait(2.0)

    def _flush_pending_inspections(self):
        while self.sio.connected:
            with self._pending_lock:
                if not self._pending_inspections:
                    return
                data = self._pending_inspections.popleft()
            try:
                self.sio.emit("inspection_status", data)
                print(
                    "Queued inspection sent:",
                    data.get("event_id", "unknown"),
                )
            except Exception as error:
                with self._pending_lock:
                    self._pending_inspections.appendleft(data)
                print("Unable to flush queued inspection:", error)
                return
    ###########################################
    def send_inspection_status(self,data):
        if not self.sio.connected:
            with self._pending_lock:
                self._pending_inspections.append(data)
            print("Inspection queued: Socket.IO is disconnected" )
            return False
        try:
            self.sio.emit("inspection_status", data)
            return True
        except Exception as error:
            with self._pending_lock:
                self._pending_inspections.append(data)
            print("Inspection queued after send error:", error)
            return False
    ###########################################
    def send_log(self, data):
        if not self.sio.connected:
            print("Log not sent: " "Socket.IO is disconnected" )
            return False
        self.sio.emit("edge_log",data)
        return True
    ###########################################
    def disconnect(self):
        self._stop_event.set()
        if self.sio.connected:
            self.sio.disconnect()
    ###########################################
