# use virtual environment of crewAI then do required installation and then create a new requirements.txt file

# set your local domain name for nginx config using /etc/hosts 
```<your_server_ip> <domain_name>```

# set up ssl for https (self signed certificate)
openssl req -x509 -nodes -days 367 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt -subj "/C=US/ST=State/L=City/O=Organization/CN=<domain_name>"

# can also run the cron job to rotate keys every year

# go in virtualenv > langchain community librairy > llms > ollama.py > change the model name of the one you want to run : mistral:7b for example at line 35

# for nginx: uncomment 'server_names_hash_bucket_size 64;' in /etc/nginx/nginx.conf file to activate your domain name site preventing errors of having several domains served (hash bucket memory problem due to several server names)
