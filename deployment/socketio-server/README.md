# Contabo Socket.IO alert relay

For a fresh Debian/Ubuntu Contabo VPS, copy only
`install_socketio_contabo.sh` to the server and run:

```bash
chmod +x install_socketio_contabo.sh
sudo ./install_socketio_contabo.sh \
  --domain socket.sukalyanai.com \
  --email YOUR_LETS_ENCRYPT_EMAIL \
  --origins https://www.sukalyanai.com,https://sukalyanai.com
```

The installer creates the service, generates an authentication token, configures
Nginx/TLS and UFW, and checks the relay. It is safe to rerun; the current token
is retained unless `--rotate-token` is supplied.

Runtime files:

```text
/opt/socketio-server/server.js
/etc/socketio-server.env
/etc/systemd/system/socketio-server.service
/etc/nginx/sites-available/socketio-server
```

The relay accepts `inspection_status` and `alert:publish`, then broadcasts
`alert_received`. Clients authenticate with Socket.IO `auth.token`.
