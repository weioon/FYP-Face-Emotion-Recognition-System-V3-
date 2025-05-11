FROM alpine:3.18

# Install necessary tools
RUN apk add --no-cache \
    bash \
    docker-cli \
    docker-compose \
    curl \
    tzdata

# Set up cron
RUN apk add --no-cache dcron

# Create the log file to be able to run tail
RUN touch /var/log/crontab.log

# Copy our script into the container
COPY ./renew_certs.sh /usr/local/bin/renew_certs.sh
RUN chmod +x /usr/local/bin/renew_certs.sh

# Copy crontab file
COPY ./crontab.txt /etc/crontabs/root

# Create necessary directories for Certbot
RUN mkdir -p /certbot/conf /certbot/www

# Set working directory
WORKDIR /

# Run crond in the foreground and tail the logs
CMD ["sh", "-c", "crond -f -d 8 & tail -f /var/log/crontab.log"]
