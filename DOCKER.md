# Docker

Docker runs the detector headlessly for log and health monitoring. Set `runtime.debug_view: false` in `config.yaml` before using Docker because OpenCV preview windows cannot open inside the container.

## Build and run

```powershell
docker compose up -d --build security-ai
```

Watch logs:

```powershell
docker compose logs -f security-ai
```

Health endpoint:

```text
http://localhost:8765/health
```

## Telegram reply bot

Run the optional Telegram reply bot when you want `/chatid` replies:

```powershell
docker compose --profile telegram-reply up -d telegram-reply-bot
```

## Required local files

These stay outside Git and are mounted into the container:

- `.env`
- `config.yaml`
- YOLO weights such as `yolo11m.pt`
- `alerts/`
- `logs/`

If `config.yaml` points at another root-level weight file, replace the `./yolo11m.pt:/app/yolo11m.pt:ro` bind mount in `docker-compose.yml`.
