# import paramiko
# import pyarrow.parquet as pq
# import pandas as pd
# import os
 
# keyfile = os.path.expanduser('/Users/Levis/.ssh/itu.pem')
# key = paramiko.RSAKey.from_private_key_file(keyfile, password='/')

# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect('orkneycloud.itu.dk', username='liwb', password='ch4ng3m3', pkey = key) 

# ftp_client= ssh.open_sftp()
# remote_file = ftp_client.open('/home/sebastian/energinet/ITU_DATA/masterdatawind.parquet')

# windmill = pq.read_table(remote_file).to_pandas()
# windmill.to_csv('data/windmills.csv',index=False)

# ftp_client.close()
# ssh.close()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

num_rows = 20
years = list(range(1990, 1990 + num_rows))
data_preproc = pd.DataFrame({
    'Year': years, 
    'A': np.random.randn(num_rows).cumsum(),
    'B': np.random.randn(num_rows).cumsum(),
    'C': np.random.randn(num_rows).cumsum(),
    'D': np.random.randn(num_rows).cumsum()})
# A single plot with four lines, one per measurement type, is obtained with

sns.lineplot(x='Year', y='value', hue='variable', 
             data=pd.melt(data_preproc, ['Year']))
plt.show()
