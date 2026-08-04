import socketio

class SocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.sio = socketio.Client(reconnection=True,reconnection_attempts=0,reconnection_delay=2)
        self._register_events()
    #########################
    def _register_events(self):
        @self.sio.event
        def connect():
            print("Socket.IO connected " "to Node.js backend" )
        @self.sio.event
        def disconnect():
            print( "Socket.IO disconnected " "from Node.js backend" )
        @self.sio.event
        def connect_error(error):
            print("Socket.IO connection error:", error)
    ############################################
    def connect(self):
        try:
            print("Connecting Socket.IO to:",self.server_url)
            self.sio.connect(self.server_url,transports=["websocket","polling"])
        except Exception as error:
            print("Unable to connect " "Socket.IO:", error )
    ###########################################
    def send_inspection_status(self,data):
        if not self.sio.connected:
            print("Inspection status not sent: " "Socket.IO is disconnected" )
            return False
        self.sio.emit("inspection_status", data)
        return True
    ###########################################
    def send_log(self, data):
        if not self.sio.connected:
            print("Log not sent: " "Socket.IO is disconnected" )
            return False
        self.sio.emit("edge_log",data)
        return True
    ###########################################
    def disconnect(self):
        if self.sio.connected:
            self.sio.disconnect()
    ###########################################