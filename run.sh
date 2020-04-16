# bin/bash
spark-submit \
  --name WindTurbine_ws \
  --master local[*] \
  --deploy-mode client \
  --conf spark.executor.instances=10 \
  prediction.py 