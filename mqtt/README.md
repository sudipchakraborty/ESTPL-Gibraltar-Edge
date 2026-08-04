# MQTT alert publisher

Reusable, non-blocking JSON alert transport for VisualAI Edge.

Credentials are read from environment variables and must not be placed in
`config.json`.

```env
MQTT_USERNAME=your-user
MQTT_PASSWORD=your-password
```

The browser must use a Mosquitto WebSocket listener. Native MQTT port `1883`
cannot be used by a web browser.
