#!/usr/bin/env bash
# Pull the latest code from GitHub and rebuild/restart the container.
# Run on the VPS from the project folder:  bash deploy/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "No .env file found. Run: cp .env.example .env  and fill in APP_DOMAIN etc." >&2
  exit 1
fi

COMPOSE_FILE="docker-compose.yml"
if [ "${1:-}" = "--caddy" ]; then
  COMPOSE_FILE="docker-compose.caddy.yml"
fi

echo "==> Pulling latest code"
git pull --ff-only

echo "==> Building and starting ($COMPOSE_FILE)"
docker compose -f "$COMPOSE_FILE" up -d --build

echo "==> Removing old unused images"
docker image prune -f >/dev/null

echo "==> Status"
docker compose -f "$COMPOSE_FILE" ps
DOMAIN=$(grep -E '^APP_DOMAIN=' .env | cut -d= -f2)
echo "Done. Open https://${DOMAIN} (first start can take ~30 seconds)."
