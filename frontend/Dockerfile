# filepath: frontend/Dockerfile
# Stage 1: Build the React app
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# Set the API URL build argument (can be overridden in docker-compose)
ARG REACT_APP_API_URL=""
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# Stage 2: Serve the static files with Nginx
FROM nginx:1.25-alpine

# Install openssl
RUN apk add --no-cache openssl

# Copy the certificate generation script and run it
COPY generate-cert.sh /generate-cert.sh
RUN chmod +x /generate-cert.sh \
    && /generate-cert.sh \
    && if [ ! -f /etc/nginx/ssl/nginx-selfsigned.crt ] || [ ! -f /etc/nginx/ssl/nginx-selfsigned.key ]; then \
        echo "Error: Certificate files not found in /etc/nginx/ssl/ after running generate-cert.sh" >&2; \
        exit 1; \
    else \
        echo "Successfully generated certificate files in /etc/nginx/ssl/"; \
    fi

COPY --from=builder /app/build /usr/share/nginx/html
# Copy a custom nginx config if needed, otherwise default works for SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
EXPOSE 443
CMD ["nginx", "-g", "daemon off;"]