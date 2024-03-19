#!/usr/bin/python3

import multiprocessing

"""Gunicorn *development* config file"""

# Django WSGI application path in pattern MODULE_NAME:VARIABLE_NAME
wsgi_app = "pdf_llm.wsgi:application"
# The number of worker processes for handling requests
workers = multiprocessing.cpu_count() * 2 + 1
# The granularity of Error log outputs
loglevel = "debug"
# The socket to bind normal for django
# bind = "0.0.0.0:8000"
# the socket to bind for nginx if reverse proxy is setup
bind = "unix:/opt/creditizens-local/run/gunicorn.sock"
# Restart workers when code changes (development only!)
reload = True
# Write access and error info to /var/log
accesslog = errorlog = "/var/log/gunicorn/gunicorn.log"
# Redirect stdout/stderr to log file
capture_output = True
# PID file so you can easily fetch process ID
pidfile = "/var/run/gunicorn/gunicorn.pid"
# Daemonize the Gunicorn process (detach & enter background)
daemon = False
