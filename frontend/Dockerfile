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
COPY --from=builder /app/build /usr/share/nginx/html
# Copy a custom nginx config if needed, otherwise default works for SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create dummy SSL certificates for Nginx initial startup
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/conf.d/ && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/conf.d/dummy.key \
    -out /etc/nginx/conf.d/dummy.crt \
    -subj "/CN=localhost"

EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]