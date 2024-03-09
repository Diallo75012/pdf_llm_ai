# use virtual environment of crewAI then do required installation and then create a new requirements.txt file

# set your local domain name for nginx config using /etc/hosts 
```<your_server_ip> <domain_name>```

# set up ssl for https (self signed certificate)
sudo openssl req -x509 -nodes -days 367 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt -subj "/C=US/ST=State/L=City/O=Organization/CN=<domain_name>"

# can also run the cron job to rotate keys every year

# go in virtualenv > langchain community librairy > llms > ollama.py > change the model name of the one you want to run : mistral:7b for example at line 35

# go in virtualenv > langchain_community library > embeddings > ollama.py > change the model llama2 to mistral:7b or the model that you have available OR just set in your code:  
```embeddings = OllamaEmbeddings(model="mistral:7b", temperature=0)```

# for nginx: uncomment 'server_names_hash_bucket_size 64;' in /etc/nginx/nginx.conf file to activate your domain name site preventing errors of having several domains served (hash bucket memory problem due to several server names)

# create a .streamlit/secrets.toml folder and file in order to use secrets in your streamlit app
```
# then just put this in your toml file and fill in the fields. i put dialect=postgresql for mine
[connections.my_database] # my_database will be used as name when initalizing database (conn = st.connection("my_database", type="sql"))
    type="sql"
    dialect="mysql"
    username="xxx"
    password="xxx"
    host="example.com" # IP or URL
    port=3306 # Port number
    database="mydb" # Database name
```
