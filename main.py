import paramiko
import pyarrow.parquet as pq
import os
 
keyfile = os.path.expanduser('/Users/Levis/.ssh/itu.pem')
key = paramiko.RSAKey.from_private_key_file(keyfile, password='/')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('orkneycloud.itu.dk', username='liwb', password='ch4ng3m3', pkey = key) 

ftp_client= ssh.open_sftp()
remote_file = ftp_client.open('/home/sebastian/energinet/ITU_DATA/masterdatawind.parquet')

windmill = pq.read_pandas(remote_file).to_pandas()

ftp_client.close()
ssh.close()

# import pysftp
# sftp = pysftp.Connection(host = 'orkneycloud.itu.dk', username="liwb", private_key = '/Users/Levis/.ssh/itu.pem')