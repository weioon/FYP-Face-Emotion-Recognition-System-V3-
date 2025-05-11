#!/bin/bash

# Initial certificate acquisition script
# This runs when the container starts, checks if certs exist, and if not, obtains them

# Configuration 
YOUR_EMAIL="weioon0509@gmail.com"
DOMAIN_NAME="emotionwave.hon2838.name.my"
CERTBOT_CONF_DIR="/certbot/conf"
CERTBOT_WWW_DIR="/certbot/www"
NGINX_SERVICE_NAME="frontend"

echo "------------------------------------"
echo "Checking for existing certificates at $(date)"
echo "------------------------------------"

# Check if certificates already exist
if [ -d "${CERTBOT_CONF_DIR}/live/${DOMAIN_NAME}" ]; then
    echo "Certificates for ${DOMAIN_NAME} already exist. Skipping initial issuance."
else
    echo "No certificates found for ${DOMAIN_NAME}. Starting initial issuance..."
    
    # Attempt to get certificates using the webroot plugin
    # First, make sure Nginx is up and can serve the ACME challenge
    echo "Waiting for Nginx to be ready to serve ACME challenges..."
    sleep 10

    # Copy dummy certs to frontend container
    echo "Copying dummy certificates to frontend container..."
    docker cp /etc/nginx/conf.d/dummy.key "${NGINX_SERVICE_NAME}:/etc/nginx/conf.d/"
    docker cp /etc/nginx/conf.d/dummy.crt "${NGINX_SERVICE_NAME}:/etc/nginx/conf.d/"
    
    echo "Attempting to obtain certificates using Certbot..."
    certbot certonly --webroot \
      -w ${CERTBOT_WWW_DIR} \
      --email ${YOUR_EMAIL} \
      --agree-tos --no-eff-email \
      -d ${DOMAIN_NAME} \
      --cert-name ${DOMAIN_NAME} \
      --config-dir ${CERTBOT_CONF_DIR} \
      --work-dir /tmp \
      --logs-dir /var/log/letsencrypt \
      --non-interactive
    
    CERTBOT_STATUS=$?
    
    if [ ${CERTBOT_STATUS} -eq 0 ]; then
        echo "Successfully obtained initial certificates for ${DOMAIN_NAME}"
        
        # Reload Nginx to use the new certificates
        echo "Reloading Nginx configuration..."
        docker compose exec ${NGINX_SERVICE_NAME} nginx -s reload
        
        NGINX_RELOAD_STATUS=$?
        if [ ${NGINX_RELOAD_STATUS} -eq 0 ]; then
            echo "Nginx reloaded successfully."
        else
            echo "Error reloading Nginx (Status: ${NGINX_RELOAD_STATUS}). Check Nginx container logs."
        fi
    else
        echo "Failed to obtain initial certificates (Status: ${CERTBOT_STATUS}). Check Certbot logs."
    fi
fi

echo "Initial certificate check/acquisition complete at $(date)"
echo "Starting regular certificate renewal service..."
echo ""
