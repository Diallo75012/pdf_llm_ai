#/bin/bash


# make the scritp executable
chmod +x /<path_to_cert>/certificate_rotation.sh

# run certificates rotation script every 1st of january (365days)
sudo /bin/bash -c 'echo "0 0 1 1 * root /<path_to_cert>/certificate_rotation.sh" >> /etc/crontab'

