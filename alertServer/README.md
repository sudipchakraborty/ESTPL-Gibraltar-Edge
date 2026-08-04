# AlertServer

Independent Socket.IO relay for VisualAI JSON alerts.

## Events

- Edge sender emits: `inspection_status`
- Server broadcasts: `alert_received`
- Health endpoint: `GET /health`

## Run independently

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m alertServer --host 0.0.0.0 --port 5000
```

On another machine, copy the `alertServer` folder and install:

```powershell
pip install -r alertServer\requirements.txt
python -m alertServer --host 0.0.0.0 --port 5000
```

Set the edge device configuration to the reachable server address:

```env
SOCKET_SERVER_URL=http://SERVER_IP:5000
```

For public production deployments, place the service behind TLS, restrict
allowed origins, and add authentication before exposing it to the internet.
