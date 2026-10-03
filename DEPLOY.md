# Deploy to your own domain with Docker

This puts the SellerPolicy Assistant online at `https://rag.your-domain.com`, with a free HTTPS certificate, on a Linux VPS (for example Hostinger) that already runs Docker and Traefik.

```
Browser --HTTPS--> your domain --DNS--> VPS IP --> Traefik (ports 80/443) --> sellerpolicy container (port 8501)
```

## What you need

- A VPS with Docker installed. Hostinger's Docker and n8n templates include Docker and Traefik.
- A domain you control (from Hostinger, GoDaddy, Namecheap, ...).
- About 3 GB of free disk space. The image is roughly 2 GB because of CPU PyTorch and the embedding model.

## 1. Point a subdomain at the VPS

In your domain's DNS settings, add one record:

| Type | Name | Value | TTL |
| --- | --- | --- | --- |
| A | `rag` | your VPS IP address | 300 (or default) |

This makes `rag.your-domain.com` point to the VPS. Check it from any computer (it can take 5–30 minutes):

```bash
nslookup rag.your-domain.com
```

It should return your VPS IP. Don't continue until it does: the HTTPS certificate can only be issued once DNS points to the server.

## 2. Open a terminal on the VPS

Use Hostinger's **Browser terminal** (VPS > Overview), or `ssh root@YOUR_VPS_IP`.

## 3. Find your Traefik settings

The app has to join the same Docker network as Traefik and use its certificate resolver.

```bash
# Name of the Traefik container
docker ps --format '{{.Names}}\t{{.Image}}' | grep -i traefik

# Its network (use the container name from above, e.g. root-traefik-1)
docker inspect root-traefik-1 --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}'

# Its certificate resolver and entrypoint names
docker inspect root-traefik-1 --format '{{join .Config.Cmd "\n"}}' | grep -E 'certificatesresolvers|entrypoints' | head
```

On Hostinger's n8n/Traefik template the answers are usually:

| Setting | Typical value |
| --- | --- |
| Network | `root_default` |
| Cert resolver | `mytlschallenge` |
| HTTPS entrypoint | `websecure` |

If `docker ps` shows no Traefik at all, skip to [Option B: no Traefik](#option-b-no-traefik-on-the-server).

## 4. Download the code and configure it

```bash
cd /opt
git clone https://github.com/Hariharan17194/hybrid-rag-seller-policy.git
cd hybrid-rag-seller-policy
cp .env.example .env
nano .env
```

Set these lines, then save with `Ctrl+O`, `Enter`, `Ctrl+X`:

```dotenv
APP_DOMAIN=rag.your-domain.com
TRAEFIK_NETWORK=root_default
TRAEFIK_CERTRESOLVER=mytlschallenge
TRAEFIK_ENTRYPOINT=websecure

# Optional - leave empty for a free public demo (extractive answers, no API cost)
OPENAI_API_KEY=
```

> **Cost warning:** if you put your OpenAI key here, anyone who opens the public site can spend it. For a portfolio demo, leave it empty, or set a monthly spending limit in the OpenAI dashboard.

## 5. Build and start

```bash
docker compose up -d --build
```

The first build takes about 5–15 minutes: it downloads PyTorch and the embedding model, and builds the index. Watch progress with:

```bash
docker compose logs -f        # Ctrl+C to stop watching (the app keeps running)
docker compose ps             # STATUS should become "Up ... (healthy)"
```

Open **https://rag.your-domain.com**. The certificate is issued on the first visit, which can take up to a minute.

## 6. Update after you push new code to GitHub

```bash
cd /opt/hybrid-rag-seller-policy
bash deploy/deploy.sh
```

This pulls the latest code, rebuilds, restarts and removes old images.

## If you deployed an earlier version on port 8501

Stop it, so the old app isn't still reachable at `http://IP:8501`:

```bash
docker ps                      # find the old container (port 0.0.0.0:8501)
docker stop <old-container-name> && docker rm <old-container-name>
```

The new setup does not publish port 8501. It is reachable only through HTTPS on your domain.

## Option B: no Traefik on the server

If ports 80 and 443 are free, Caddy can handle HTTPS instead. Set `APP_DOMAIN` in `.env` (the `TRAEFIK_*` lines are ignored), then:

```bash
docker compose -f docker-compose.caddy.yml up -d --build
# updates later:  bash deploy/deploy.sh --caddy
```

## Troubleshooting

| What you see | Likely cause | Fix |
| --- | --- | --- |
| `404 page not found` (plain text) | Traefik can't see the container | Check `TRAEFIK_NETWORK` matches step 3, then `docker compose up -d` |
| Browser warns "certificate not valid" / `TRAEFIK DEFAULT CERT` | DNS not pointing to the VPS yet, or wrong resolver name | Check `nslookup`, then `TRAEFIK_CERTRESOLVER`; check `docker logs root-traefik-1 \| grep -i acme` |
| `Bad Gateway` | App still starting, or it crashed | `docker compose logs --tail 50` |
| `network root_default declared as external, but could not be found` | Wrong network name | Use the exact name from step 3 |
| Build stops with `Killed` | VPS ran out of memory during the build | Add swap: `fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`, then build again |
| `no space left on device` | Old images filling the disk | `docker system prune -a` (removes unused images) |
| Page loads but stays on "Please wait..." | Websocket blocked by another proxy (e.g. Cloudflare settings) | Turn on WebSockets in Cloudflare, or set the DNS record to "DNS only" |

## Useful commands

```bash
docker compose ps                 # status + health
docker compose logs -f            # live logs
docker compose restart            # restart the app
docker compose down               # stop and remove the container
docker stats sellerpolicy         # CPU / memory use
```
