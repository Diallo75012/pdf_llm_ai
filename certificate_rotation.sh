#!/bin/bash

# Generate a new self-signed SSL certificate
openssl req -x509 -nodes -days 367 -newkey rsa:2048 \
-keyout /etc/ssl/private/nginx-selfsigned.key \
-out /etc/ssl/certs/nginx-selfsigned.crt -subj "/C=US/ST=State/L=City/O=Organization/CN=creditizens.local"

# Reload Nginx to apply the changes
systemctl reload nginx
