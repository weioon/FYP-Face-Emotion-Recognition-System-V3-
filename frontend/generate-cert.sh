#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

# Create a directory for the certificates if it doesn't exist
mkdir -p /etc/nginx/ssl

# Generate a self-signed certificate and private key
# Using localhost as CN, adjust if needed for a specific domain during development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/nginx-selfsigned.key \
    -out /etc/nginx/ssl/nginx-selfsigned.crt \
    -subj "/C=MY/ST=Kedah/L=Sintok/O=UUM/OU=IT/CN=localhost"

echo "Self-signed certificate generated."