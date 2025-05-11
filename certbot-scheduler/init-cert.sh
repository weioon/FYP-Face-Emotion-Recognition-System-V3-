#!/bin/bash
# filepath: c:\Users\weioo\OneDrive - UNIVERSITY UTARA MALAYSIA\Desktop\FYP-Face-Emotion-Recognition-System-V3-\certbot-scheduler\init-cert.sh

# Initial certificate issuance script
# This script is specifically designed to obtain the first certificate

# Configuration
DOMAIN="emotionwave.hon2838.name.my"
EMAIL="weioon0509@gmail.com"
CERTBOT_DIR="/certbot"

echo "======================================"
echo "Starting initial certificate issuance"
echo "======================================"

# Wait a bit to make sure Nginx is up and running
echo "Waiting for Nginx to be ready..."
sleep 15

# First, test if the domain is accessible and properly configured for ACME challenge
echo "Testing domain $DOMAIN for ACME challenge access..."
curl -sI "http://$DOMAIN/.well-known/acme-challenge/test-file" | head -n 1

# Create a test file to verify Nginx is serving from the correct location
echo "Creating test file in webroot..."
echo "This is a test file for ACME challenge" > "$CERTBOT_DIR/www/test-file"
chmod 644 "$CERTBOT_DIR/www/test-file"

# Check if test file is accessible
echo "Checking if test file is accessible..."
curl -s "http://$DOMAIN/.well-known/acme-challenge/test-file" || echo "Test file not accessible!"

# Try to obtain the certificate (staging mode first for testing)
echo "Attempting to obtain certificate from Let's Encrypt staging server..."
certbot certonly --webroot \
  --webroot-path="$CERTBOT_DIR/www" \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email \
  --staging \
  -d "$DOMAIN" \
  --config-dir="$CERTBOT_DIR/conf" \
  --work-dir="/tmp" \
  --logs-dir="/var/log/letsencrypt" \
  --verbose

# If staging was successful, try production
if [ $? -eq 0 ]; then
  echo "Staging certificate obtained successfully. Now trying production server..."
  # Remove the staging certificate
  rm -rf "$CERTBOT_DIR/conf/live/$DOMAIN"
  rm -rf "$CERTBOT_DIR/conf/archive/$DOMAIN"
  rm -rf "$CERTBOT_DIR/conf/renewal/$DOMAIN.conf"
  
  # Obtain a production certificate
  certbot certonly --webroot \
    --webroot-path="$CERTBOT_DIR/www" \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" \
    --config-dir="$CERTBOT_DIR/conf" \
    --work-dir="/tmp" \
    --logs-dir="/var/log/letsencrypt" \
    --verbose
    
  if [ $? -eq 0 ]; then
    echo "Production certificate obtained successfully!"
    # Reload Nginx to use the new certificate
    echo "Reloading Nginx..."
    docker compose exec frontend nginx -s reload
    echo "Done!"
  else
    echo "Failed to obtain production certificate."
  fi
else
  echo "Failed to obtain staging certificate. Check the logs for details."
fi

echo "======================================"
echo "Initial certificate issuance completed"
echo "======================================"