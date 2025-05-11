#!/bin/bash

# Script to attempt Let's Encrypt certificate renewal and reload Nginx if needed.
# This script is intended to be run periodically (e.g., via cron or Task Scheduler).

# --- Configuration ---
# !!! IMPORTANT: Replace with your actual email address for Let's Encrypt notices.
YOUR_EMAIL="weioon0509@gmail.com"

# Domain(s) for the certificate. If multiple, use comma-separated or multiple -d flags.
# This should match the domain you used for initial certificate issuance.
DOMAIN_NAME="emotionwave.hon2838.name.my"

# Docker Compose project name (optional, usually derived from the directory name)
# If your docker-compose.yml is in the same directory as this script, you might not need this.
# Check 'docker compose ps' for your project name if unsure.
# DOCKER_COMPOSE_PROJECT_NAME="fyp-face-emotion-recognition-system-v3"

# Name of your Nginx service in docker-compose.yml
NGINX_SERVICE_NAME="frontend"

# Host paths for Certbot data and ACME challenges.
# These paths must be mounted as volumes in your Nginx and Certbot (when run) containers.
# Assumes docker-compose.yml is in the same directory or one level up from where these are created.
CERTBOT_CONF_DIR="./certbot/conf"
CERTBOT_WWW_DIR="./certbot/www"

# --- End Configuration ---

echo "------------------------------------"
echo "Starting Let's Encrypt renewal process at $(date)"
echo "------------------------------------"

# Ensure host directories for Certbot exist
mkdir -p "${CERTBOT_CONF_DIR}"
mkdir -p "${CERTBOT_WWW_DIR}"
echo "Host directories for Certbot data ensured at ${CERTBOT_CONF_DIR} and ${CERTBOT_WWW_DIR}"

# Attempt to renew the certificate using the certbot/certbot Docker image
# This command uses the webroot plugin. Your Nginx must be configured to serve
# challenges from CERTBOT_WWW_DIR (e.g., mounted to /var/www/certbot in Nginx).
echo "Attempting certificate renewal for ${DOMAIN_NAME}..."
docker run --rm \
  -v "${PWD}/${CERTBOT_CONF_DIR}:/etc/letsencrypt" \
  -v "${PWD}/${CERTBOT_WWW_DIR}:/var/www/certbot" \
  certbot/certbot:latest renew \
  --webroot -w /var/www/certbot \
  --email "${YOUR_EMAIL}" \
  --agree-tos --no-eff-email \
  --quiet # Suppresses most output, only shows errors or if renewal happens

RENEWAL_STATUS=$?

if [ ${RENEWAL_STATUS} -eq 0 ]; then
  echo "Certbot renew command executed successfully (this doesn't necessarily mean a renewal occurred, just that the command ran without error)."
  # Certbot's renew command is idempotent. It will only renew if certificates are due.
  # We reload Nginx to ensure it picks up renewed certificates if any.
  # A more advanced setup might use Certbot's --deploy-hook.
else
  echo "Certbot renew command failed with status ${RENEWAL_STATUS}. Check Certbot logs if this persists."
  # You might want to add notification mechanisms here for failures.
fi

# Reload Nginx to pick up any renewed certificates.
# This is generally safe to do even if no renewal occurred.
echo "Reloading Nginx configuration for service '${NGINX_SERVICE_NAME}'..."
if [ -n "${DOCKER_COMPOSE_PROJECT_NAME}" ]; then
  docker compose -p "${DOCKER_COMPOSE_PROJECT_NAME}" exec "${NGINX_SERVICE_NAME}" nginx -s reload
else
  # Assumes docker-compose.yml is in the current directory or discoverable by docker compose
  docker compose exec "${NGINX_SERVICE_NAME}" nginx -s reload
fi

NGINX_RELOAD_STATUS=$?

if [ ${NGINX_RELOAD_STATUS} -eq 0 ]; then
  echo "Nginx reloaded successfully."
else
  echo "Error reloading Nginx (Status: ${NGINX_RELOAD_STATUS}). Check Nginx container logs."
fi

echo "------------------------------------"
echo "Let's Encrypt renewal process finished at $(date)"
echo "------------------------------------"
echo ""

# Note: For this script to work, your Nginx container (service name: frontend)
# must have a volume mount for the ACME challenge like:
# volumes:
#   - ./certbot/www:/var/www/certbot:ro # Nginx reads challenges
#
# And your nginx.conf should serve this path:
# location /.well-known/acme-challenge/ {
#     root /var/www/certbot;
# }
#
# The Certbot container run by this script also mounts these host paths:
# - ./certbot/conf:/etc/letsencrypt (for storing certificates)
# - ./certbot/www:/var/www/certbot (for placing challenge files)
#
# Initial certificate issuance is not covered by this script. You would typically run:
# docker run --rm -p 80:80 \
#   -v "${PWD}/certbot/conf:/etc/letsencrypt" \
#   -v "${PWD}/certbot/www:/var/www/certbot" \
#   certbot/certbot:latest certonly \
#   --webroot -w /var/www/certbot \
#   --email YOUR_EMAIL --agree-tos --no-eff-email \
#   -d YOUR_DOMAIN_NAME
# (Ensure Nginx is temporarily stopped or not using port 80, or use Nginx to serve challenges during initial setup)
