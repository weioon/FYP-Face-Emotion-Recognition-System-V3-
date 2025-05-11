#!/bin/bash

# Script to attempt Let's Encrypt certificate renewal and reload Nginx if needed.
# This script is intended to be run periodically by the cron scheduler in the certbot-scheduler container.

# --- Configuration ---
# Your email for Let's Encrypt notices
YOUR_EMAIL="weioon0509@gmail.com"

# Domain for the certificate
DOMAIN_NAME="emotionwave.hon2838.name.my"

# Name of your Nginx service in docker-compose.yml
NGINX_SERVICE_NAME="frontend"

# Paths for Certbot data and ACME challenges
CERTBOT_CONF_DIR="/certbot/conf"
CERTBOT_WWW_DIR="/certbot/www"

# --- End Configuration ---

echo "------------------------------------"
echo "Starting Let's Encrypt renewal process at $(date)"
echo "------------------------------------"

# Check if we need to do initial certificate issuance (if certificates don't exist yet)
if [ ! -d "${CERTBOT_CONF_DIR}/live/${DOMAIN_NAME}" ]; then
  echo "No certificate found for ${DOMAIN_NAME}. Attempting initial issuance..."
  
  # Initial certificate issuance using the webroot method
  certbot certonly --webroot \
    -w ${CERTBOT_WWW_DIR} \
    --email ${YOUR_EMAIL} \
    --agree-tos --no-eff-email \
    -d ${DOMAIN_NAME} \
    --cert-name ${DOMAIN_NAME} \
    --config-dir ${CERTBOT_CONF_DIR} \
    --work-dir /tmp \
    --logs-dir /var/log/letsencrypt
  
  CERTBOT_EXIT=$?
  if [ ${CERTBOT_EXIT} -ne 0 ]; then
    echo "Failed to obtain initial certificate (exit code: ${CERTBOT_EXIT})"
    exit 1
  else
    echo "Successfully obtained initial certificate for ${DOMAIN_NAME}"
  fi
else
  # Attempt to renew existing certificates
  echo "Attempting certificate renewal for ${DOMAIN_NAME}..."
  
  certbot renew \
    --webroot -w ${CERTBOT_WWW_DIR} \
    --email ${YOUR_EMAIL} \
    --agree-tos --no-eff-email \
    --config-dir ${CERTBOT_CONF_DIR} \
    --work-dir /tmp \
    --logs-dir /var/log/letsencrypt \
    --cert-name ${DOMAIN_NAME}
  
  RENEWAL_STATUS=$?
  if [ ${RENEWAL_STATUS} -eq 0 ]; then
    echo "Certbot renew command executed successfully."
  else
    echo "Certbot renew command failed with status ${RENEWAL_STATUS}."
    exit 1
  fi
fi

# Reload Nginx configuration using Docker commands
echo "Reloading Nginx configuration for service '${NGINX_SERVICE_NAME}'..."
docker compose exec ${NGINX_SERVICE_NAME} nginx -s reload

NGINX_RELOAD_STATUS=$?
if [ ${NGINX_RELOAD_STATUS} -eq 0 ]; then
  echo "Nginx reloaded successfully."
else
  echo "Error reloading Nginx (Status: ${NGINX_RELOAD_STATUS})."
  exit 1
fi

echo "------------------------------------"
echo "Let's Encrypt renewal process finished at $(date)"
echo "------------------------------------"
echo ""
