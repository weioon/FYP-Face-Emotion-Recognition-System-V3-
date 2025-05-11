#!/bin/bash
# filepath: c:\Users\weioo\OneDrive - UNIVERSITY UTARA MALAYSIA\Desktop\FYP-Face-Emotion-Recognition-System-V3-\frontend\entrypoint.sh

# Make sure the certbot webroot directory exists
mkdir -p /var/www/certbot

# Wait for Nginx to be available
echo "Waiting for Nginx to be available..."
sleep 5

# Check if we need the dummy SSL certificate
if [ ! -f "/etc/letsencrypt/live/emotionwave.hon2838.name.my/fullchain.pem" ]; then
    echo "No SSL certificate found, using self-signed certificate temporarily"
    # Generate a self-signed certificate for initial server startup
    mkdir -p /etc/nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/nginx-selfsigned.key \
        -out /etc/nginx/ssl/nginx-selfsigned.crt \
        -subj "/CN=localhost"
    
    # Update Nginx config to use self-signed certificate
    cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name emotionwave.hon2838.name.my localhost;

    # This is crucial for Let's Encrypt verification
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
        allow all;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name emotionwave.hon2838.name.my localhost;

    # Temporary self-signed certificate
    ssl_certificate /etc/nginx/ssl/nginx-selfsigned.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx-selfsigned.key;

    root /usr/share/nginx/html;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Optional: Add cache control headers for static assets
    location ~* \.(?:ico|css|js|gif|jpe?g|png)$ {
        expires 1M;
        add_header Cache-Control "public";
    }
}
EOF
else
    echo "SSL certificate found, using Let's Encrypt certificate"
    # Use the Let's Encrypt certificates
    cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name emotionwave.hon2838.name.my localhost;

    # This is crucial for Let's Encrypt verification
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
        allow all;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name emotionwave.hon2838.name.my localhost;

    # Let's Encrypt certificates
    ssl_certificate /etc/letsencrypt/live/emotionwave.hon2838.name.my/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/emotionwave.hon2838.name.my/privkey.pem;

    # Recommended SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    root /usr/share/nginx/html;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Optional: Add cache control headers for static assets
    location ~* \.(?:ico|css|js|gif|jpe?g|png)$ {
        expires 1M;
        add_header Cache-Control "public";
    }
}
EOF
fi

# Start Nginx
echo "Starting Nginx..."
exec nginx -g 'daemon off;'