# bin/bash
spark-submit \
  --master yarn \
  --deploy-mode client\
  prediction.py 