# EC2 Deployment Guide — api.gmr.zendot.in

Backend: **https://api.gmr.zendot.in** · Frontend: **https://gmr.zendot.in** (deployed separately)

## 1. Launch the EC2 instance

| Setting | Recommendation |
| --- | --- |
| AMI | Ubuntu Server 22.04 LTS (64-bit x86) |
| Instance type | `t3.large` minimum (2 vCPU / 8 GB) for API-only; `c5.2xlarge`+ or a GPU instance (`g4dn.xlarge`) if camera AI workers run on this box |
| Storage | 50 GB+ gp3 (snapshots/crops grow over time) |
| Key pair | Create/download a `.pem` key |

**Security Group (inbound):**

| Port | Source | Purpose |
| --- | --- | --- |
| 22 | Your IP only | SSH |
| 80 | 0.0.0.0/0 | HTTP (certbot + redirect) |
| 443 | 0.0.0.0/0 | HTTPS |

Do **NOT** open 8000 or 5432 — nginx proxies the app locally, Postgres stays internal.

Allocate an **Elastic IP** and associate it with the instance (so the IP never changes).

## 2. DNS

In your DNS provider for `zendot.in`, add:

```
A    api.gmr    ->  <ELASTIC_IP>      TTL 300
```

(`gmr.zendot.in` points wherever the frontend is hosted — Vercel/S3+CloudFront/another box.)

Verify before continuing: `dig +short api.gmr.zendot.in` returns the Elastic IP.

## 3. Server setup

```bash
ssh -i your-key.pem ubuntu@<ELASTIC_IP>

# System updates + tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx certbot python3-certbot-nginx

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker          # or log out / back in
```

## 4. Deploy the application

```bash
cd /opt
sudo mkdir retail-ai-platform && sudo chown ubuntu:ubuntu retail-ai-platform

# Get the code (git clone, or scp from your machine):
#   Option A - git:
git clone <YOUR_REPO_URL> retail-ai-platform
#   Option B - from your laptop:
#   scp -i your-key.pem -r retail-ai-platform ubuntu@<ELASTIC_IP>:/opt/

cd /opt/retail-ai-platform

# Production environment
cp deployment/.env.production .env
nano .env
#   - SECRET_KEY:        paste output of `openssl rand -hex 32`
#   - POSTGRES_PASSWORD: strong password (also inside both DATABASE_* URLs)
#   - CORS_ORIGINS:      https://gmr.zendot.in   (already set)

# Model weights (optional now; stubs run without them)
# yolov8n.pt downloads automatically on first run, or:
#   wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
# OSNet weights -> models/osnet_x1_0.pth

# Build & start (app runs migrations automatically on boot)
docker compose -f docker-compose.prod.yml up -d --build

# Verify
docker compose -f docker-compose.prod.yml ps
curl http://127.0.0.1:8000/health        # -> {"status":"ok"}
```

## 5. Nginx + HTTPS

```bash
sudo cp deployment/nginx-api.gmr.zendot.in.conf /etc/nginx/sites-available/api.gmr.zendot.in
sudo ln -s /etc/nginx/sites-available/api.gmr.zendot.in /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# TLS certificate (auto-renews via systemd timer)
sudo certbot --nginx -d api.gmr.zendot.in --redirect -m you@zendot.in --agree-tos -n
```

## 6. Verify from anywhere

```bash
curl https://api.gmr.zendot.in/health
# {"status":"ok"}
```

- Swagger UI: https://api.gmr.zendot.in/docs
- Create the first admin: `POST https://api.gmr.zendot.in/api/auth/signup`

The frontend at `gmr.zendot.in` calls the API with base URL `https://api.gmr.zendot.in` —
CORS is already restricted to that origin via `CORS_ORIGINS`.

## 7. Operations

```bash
cd /opt/retail-ai-platform

docker compose -f docker-compose.prod.yml logs -f app      # logs
docker compose -f docker-compose.prod.yml restart app      # restart API
docker compose -f docker-compose.prod.yml up -d --build    # deploy new code (after git pull)
docker compose -f docker-compose.prod.yml down             # stop (data persists in volumes)

# DB backup (cron this daily)
docker exec retail_ai_postgres pg_dump -U retail_user retail_ai_db | gzip > /opt/backups/db_$(date +%F).sql.gz
```

## 8. Production checklist

- [ ] `SECRET_KEY` and `POSTGRES_PASSWORD` changed from defaults
- [ ] Ports 8000/5432 NOT in the security group
- [ ] SSH restricted to your IP
- [ ] HTTPS works and HTTP redirects (`--redirect` flag handled it)
- [ ] Elastic IP attached (IP survives instance stop/start)
- [ ] DB backups scheduled
- [ ] CloudWatch alarm on instance CPU/disk (optional but recommended)

## Notes for the AI runtime

RTSP cameras must be reachable **from the EC2 instance**. For an on-prem pharmacy this
usually means a site-to-site VPN (AWS Site-to-Site VPN or WireGuard) between the store
network and the VPC — or run this same stack on an on-prem box instead and use EC2 only
for the dashboard/API. The application supports both topologies unchanged.