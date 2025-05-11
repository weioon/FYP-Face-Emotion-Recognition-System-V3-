FROM alpine:3.18

# Install necessary tools
RUN apk add --no-cache \
    bash \
    docker-cli \
    docker-compose \
    curl \
    tzdata \
    openssl \
    certbot

# Set up cron
RUN apk add --no-cache dcron

# Create the log file to be able to run tail
RUN touch /var/log/crontab.log

# Create dummy SSL certificates for Nginx to start with
RUN mkdir -p /etc/nginx/conf.d/ && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/conf.d/dummy.key \
    -out /etc/nginx/conf.d/dummy.crt \
    -subj "/CN=localhost"

# Copy our script into the container
COPY ./renew_certs.sh /usr/local/bin/renew_certs.sh
RUN chmod +x /usr/local/bin/renew_certs.sh

# Copy crontab file
COPY ./crontab.txt /etc/crontabs/root

# Create necessary directories for Certbot
RUN mkdir -p /certbot/conf /certbot/www

# Initial certificate script
COPY ./init-cert.sh /usr/local/bin/init-cert.sh
RUN chmod +x /usr/local/bin/init-cert.sh

# Set working directory
WORKDIR /

# Run initial certificate acquisition, then start cron
CMD ["/bin/sh", "-c", "/usr/local/bin/init-cert.sh && crond -f -d 8 & tail -f /var/log/crontab.log"]
