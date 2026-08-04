"use strict";

const http = require("node:http");
const { Server } = require("socket.io");

const port = Number(
  process.env.PORT ||
  process.env.SOCKETIO_PORT ||
  3000,
);
const host = process.env.HOST || "127.0.0.1";
const connectionToken =
  process.env.SOCKET_IO_TOKEN ||
  process.env.SOCKET_AUTH_TOKEN ||
  process.env.SOCKETIO_TOKEN ||
  process.env.SOCKETIO_AUTH_TOKEN ||
  process.env.CONNECTION_TOKEN ||
  process.env.AUTH_TOKEN;
const allowedOrigins = (
  process.env.ALLOWED_ORIGINS ||
  process.env.ALLOWED_ORIGIN ||
  process.env.CORS_ORIGIN ||
  process.env.SOCKETIO_CORS_ORIGIN ||
  "https://www.sukalyanai.com"
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

if (!connectionToken) {
  throw new Error("SOCKET_IO_TOKEN is required");
}

let io;
const server = http.createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({
      status: "ok",
      service: "socketio-alert-relay",
      clients: io?.engine?.clientsCount || 0,
      timestamp: new Date().toISOString(),
    }));
    return;
  }

  response.writeHead(404, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ error: "Not Found" }));
});
io = new Server(server, {
  cors: {
    origin: [...new Set([...allowedOrigins, "https://sukalyanai.com"])],
    methods: ["GET", "POST"],
  },
  maxHttpBufferSize: 1_000_000,
});

io.use((socket, next) => {
  if (socket.handshake.auth?.token !== connectionToken) {
    next(new Error("unauthorized"));
    return;
  }
  next();
});

io.on("connection", (socket) => {
  socket.emit("server:ready", {
    socketId: socket.id,
    message: "Connected to the Socket.IO alert relay",
    timestamp: new Date().toISOString(),
  });

  const relayAlert = (payload, acknowledge) => {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      acknowledge?.({ received: false, error: "Alert must be an object" });
      return;
    }

    const receivedAt = new Date().toISOString();
    const envelope = {
      received_at: receivedAt,
      alert: payload,
    };
    io.emit("alert_received", envelope);
    acknowledge?.({ received: true, received_at: receivedAt });
  };

  socket.on("inspection_status", relayAlert);
  socket.on("alert:publish", relayAlert);
});

server.listen(port, host, () => {
  console.log(`Socket.IO alert relay listening on ${host}:${port}`);
});
