#!/usr/bin/env bash
#
# Standalone Socket.IO alert-relay installer for a Debian/Ubuntu Contabo VPS.
# Safe to rerun: existing configuration is backed up and the authentication
# token is preserved unless --rotate-token is supplied.

set -Eeuo pipefail
IFS=$'\n\t'

APP_USER="socketio"
APP_GROUP="socketio"
APP_DIR="/opt/socketio-server"
ENV_FILE="/etc/socketio-server.env"
SERVICE_FILE="/etc/systemd/system/socketio-server.service"
NGINX_SITE="/etc/nginx/sites-available/socketio-server"
NGINX_LINK="/etc/nginx/sites-enabled/socketio-server"
PORT="3000"
DOMAIN=""
EMAIL=""
ORIGINS="https://www.sukalyanai.com,https://sukalyanai.com"
ROTATE_TOKEN="false"
SKIP_CERTBOT="false"

usage() {
  cat <<'USAGE'
Usage:
  sudo bash install_socketio_contabo.sh \
    --domain socket.example.com \
    --email admin@example.com \
    --origins https://www.example.com,https://example.com

Options:
  --domain NAME       Public Socket.IO hostname (required)
  --email ADDRESS     Email used by Let's Encrypt (required unless --skip-certbot)
  --origins CSV       Allowed browser origins
  --port NUMBER       Local relay port (default: 3000)
  --rotate-token      Replace the existing SOCKET_AUTH_TOKEN
  --skip-certbot      Configure HTTP only; useful before DNS is ready
  -h, --help          Show this help

The relay port stays private on 127.0.0.1. Browsers connect through Nginx on
HTTPS port 443. The generated token is printed at the end; keep it secret.
USAGE
}

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
valid_port() { [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" >= 1 && "$1" <= 65535 )); }
backup_if_present() {
  local target="$1"
  local stamp="$2"
  if [[ -e "$target" || -L "$target" ]]; then
    cp -a -- "$target" "${target}.backup-${stamp}"
  fi
}

while (($#)); do
  case "$1" in
    --domain) (($# >= 2)) || die "--domain requires a value"; DOMAIN="$2"; shift 2 ;;
    --email) (($# >= 2)) || die "--email requires a value"; EMAIL="$2"; shift 2 ;;
    --origins) (($# >= 2)) || die "--origins requires a value"; ORIGINS="$2"; shift 2 ;;
    --port) (($# >= 2)) || die "--port requires a value"; PORT="$2"; shift 2 ;;
    --rotate-token) ROTATE_TOKEN="true"; shift ;;
    --skip-certbot) SKIP_CERTBOT="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || die "Run this installer with sudo."
[[ -n "$DOMAIN" ]] || die "--domain is required."
[[ "$DOMAIN" =~ ^[A-Za-z0-9.-]+$ ]] || die "Invalid domain name: $DOMAIN"
valid_port "$PORT" || die "Invalid port: $PORT"
[[ -n "$ORIGINS" ]] || die "--origins cannot be empty."
if [[ "$SKIP_CERTBOT" != "true" ]]; then
  [[ "$EMAIL" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]] ||
    die "A valid --email is required for Let's Encrypt."
fi

[[ -r /etc/os-release ]] || die "This installer supports Debian and Ubuntu."
# shellcheck disable=SC1091
. /etc/os-release
case "${ID:-}" in
  debian|ubuntu) ;;
  *) die "Unsupported operating system: ${ID:-unknown}. Use Debian or Ubuntu." ;;
esac

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
log "Installing operating-system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl nginx nodejs npm openssl ufw
if [[ "$SKIP_CERTBOT" != "true" ]]; then
  apt-get install -y --no-install-recommends certbot python3-certbot-nginx
fi

NODE_BIN="$(command -v node || true)"
NPM_BIN="$(command -v npm || true)"
[[ -x "$NODE_BIN" ]] || die "Node.js was not installed."
[[ -x "$NPM_BIN" ]] || die "npm was not installed."
NODE_MAJOR="$("$NODE_BIN" -p 'Number(process.versions.node.split(".")[0])')"
(( NODE_MAJOR >= 18 )) ||
  die "Node.js 18 or newer is required; found $("$NODE_BIN" --version)."

log "Creating the service account and application directory"
getent group "$APP_GROUP" >/dev/null || groupadd --system "$APP_GROUP"
if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --system --gid "$APP_GROUP" --home-dir "$APP_DIR" \
    --shell /usr/sbin/nologin "$APP_USER"
fi
install -d -o "$APP_USER" -g "$APP_GROUP" -m 0750 "$APP_DIR"

backup_if_present "$APP_DIR/server.js" "$STAMP"
backup_if_present "$APP_DIR/package.json" "$STAMP"
backup_if_present "$ENV_FILE" "$STAMP"
backup_if_present "$SERVICE_FILE" "$STAMP"
backup_if_present "$NGINX_SITE" "$STAMP"

log "Installing the Socket.IO relay"
cat >"$APP_DIR/package.json" <<'PACKAGE_JSON'
{
  "name": "sukalyanai-socketio-alert-relay",
  "version": "1.0.0",
  "private": true,
  "description": "Authenticated Socket.IO relay for VisualAI alerts",
  "main": "server.js",
  "engines": {"node": ">=18"},
  "dependencies": {"socket.io": "^4.8.1"}
}
PACKAGE_JSON

cat >"$APP_DIR/server.js" <<'SERVER_JS'
"use strict";

const http = require("node:http");
const { Server } = require("socket.io");

const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const connectionToken =
  process.env.SOCKET_AUTH_TOKEN || process.env.SOCKET_IO_TOKEN;
const allowedOrigins = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

if (!connectionToken) throw new Error("SOCKET_AUTH_TOKEN is required");

let io;
const server = http.createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    response.writeHead(200, {"Content-Type": "application/json"});
    response.end(JSON.stringify({
      status: "ok",
      service: "socketio-alert-relay",
      clients: io?.engine?.clientsCount || 0,
      timestamp: new Date().toISOString(),
    }));
    return;
  }
  response.writeHead(404, {"Content-Type": "application/json"});
  response.end(JSON.stringify({error: "Not Found"}));
});

io = new Server(server, {
  cors: {origin: allowedOrigins, methods: ["GET", "POST"]},
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
  console.log(`client connected: ${socket.id}`);
  socket.emit("server:ready", {
    socketId: socket.id,
    message: "Connected to the Socket.IO alert relay",
    timestamp: new Date().toISOString(),
  });

  const relayAlert = (payload, acknowledge) => {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      acknowledge?.({received: false, error: "Alert must be an object"});
      return;
    }
    const receivedAt = new Date().toISOString();
    io.emit("alert_received", {received_at: receivedAt, alert: payload});
    acknowledge?.({received: true, received_at: receivedAt});
  };

  socket.on("inspection_status", relayAlert);
  socket.on("alert:publish", relayAlert);
  socket.on("disconnect", (reason) => {
    console.log(`client disconnected: ${socket.id}; reason: ${reason}`);
  });
});

server.listen(port, host, () => {
  console.log(`Socket.IO alert relay listening on ${host}:${port}`);
});

function shutdown(signal) {
  console.log(`${signal} received; closing the server`);
  io.close(() => server.close(() => process.exit(0)));
  setTimeout(() => process.exit(1), 10_000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
SERVER_JS

chown "$APP_USER:$APP_GROUP" "$APP_DIR/package.json" "$APP_DIR/server.js"
chmod 0640 "$APP_DIR/package.json" "$APP_DIR/server.js"
(
  cd "$APP_DIR"
  runuser -u "$APP_USER" -- "$NPM_BIN" install --omit=dev --no-audit --no-fund
)

TOKEN=""
if [[ "$ROTATE_TOKEN" != "true" && -r "$ENV_FILE" ]]; then
  TOKEN="$(sed -n 's/^SOCKET_AUTH_TOKEN=//p' "$ENV_FILE" | tail -n 1)"
fi
[[ -n "$TOKEN" ]] || TOKEN="$(openssl rand -hex 32)"

cat >"$ENV_FILE" <<ENVIRONMENT
NODE_ENV=production
HOST=127.0.0.1
PORT=$PORT
ALLOWED_ORIGINS=$ORIGINS
SOCKET_AUTH_TOKEN=$TOKEN
ENVIRONMENT
chown root:"$APP_GROUP" "$ENV_FILE"
chmod 0640 "$ENV_FILE"

cat >"$SERVICE_FILE" <<SYSTEMD
[Unit]
Description=Socket.IO Alert Relay
Documentation=https://socket.io/docs/v4/
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$NODE_BIN $APP_DIR/server.js
Restart=on-failure
RestartSec=3
TimeoutStopSec=15
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
CapabilityBoundingSet=
AmbientCapabilities=

[Install]
WantedBy=multi-user.target
SYSTEMD

log "Configuring Nginx"
cat >"$NGINX_SITE" <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    location = /health {
        proxy_pass http://127.0.0.1:$PORT/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /socket.io/ {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 65s;
        proxy_send_timeout 65s;
        proxy_buffering off;
    }
}
NGINX
ln -sfn "$NGINX_SITE" "$NGINX_LINK"
rm -f /etc/nginx/sites-enabled/default

"$NODE_BIN" --check "$APP_DIR/server.js"
nginx -t
systemctl daemon-reload
systemctl enable --now socketio-server
systemctl enable --now nginx
systemctl restart socketio-server
systemctl reload nginx

log "Configuring the firewall"
SSH_PORT="$(sshd -T 2>/dev/null | awk '$1 == "port" {print $2; exit}')"
SSH_PORT="${SSH_PORT:-22}"
ufw allow "${SSH_PORT}/tcp" comment "SSH"
ufw allow 80/tcp comment "HTTP"
ufw allow 443/tcp comment "HTTPS"
ufw --force enable

if [[ "$SKIP_CERTBOT" != "true" ]]; then
  log "Requesting and installing the TLS certificate"
  certbot --nginx --non-interactive --agree-tos --redirect \
    --email "$EMAIL" -d "$DOMAIN"
  systemctl reload nginx
fi

log "Running health checks"
for _ in {1..15}; do
  curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null ||
  die "Local health check failed. Run: journalctl -u socketio-server -n 100"

PUBLIC_SCHEME="http"
[[ "$SKIP_CERTBOT" == "true" ]] || PUBLIC_SCHEME="https"

cat <<RESULT

Installation complete.

Public URL:        $PUBLIC_SCHEME://$DOMAIN
Health check:      $PUBLIC_SCHEME://$DOMAIN/health
Allowed origins:  $ORIGINS
Publish events:    inspection_status or alert:publish
Browser event:     alert_received
Authentication:    auth.token

SOCKET_AUTH_TOKEN=$TOKEN

Copy the token into the VisualAI Edge .env and your protected frontend CI/CD
secret. Do not commit it to Git. It is also stored in $ENV_FILE.

Useful commands:
  systemctl status socketio-server --no-pager
  journalctl -u socketio-server -f
  nginx -t
  certbot renew --dry-run
RESULT
