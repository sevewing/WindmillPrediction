import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sparktorch import serialize_torch_obj, SparkTorch, PysparkPipelineWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler

import constant
import tools

def _load_parquet(path, schema:StructType = None):
    """
    load apache parquet file
    """
    return sqlContext.read.schema(schema).parquet(path) if schema is not None else sqlContext.read.parquet(path)

def _load_weather_toPandas(path, schema:StructType = None):
    """
    load apache parquet file
    """
    weather = _load_parquet(path) \
                    .withColumnRenamed("__index_level_0__", "TIME") \
                    .dropna() \
                    .withColumn("id", monotonically_increasing_id())
    weather.createOrReplaceTempView("weather_temp")
    weather_dic = spark.sql("select * from weather_temp where id in (select max(id) as id from weather_temp group by TIME)").toPandas()
    return weather_dic

def _load_csv(path, schema:StructType = None):
    """
    load csv file
    """
    return sqlContext.read.csv(path, sep=",", header=True, schema=schema)


def load_files():
# to read parquet file
    settlement = _load_parquet(constant.settlement_path, constant.settlement_schema)
    settlement = settlement.dropna(subset =["VAERDI"]) \
                .withColumn("VAERDI", settlement["VAERDI"].cast("float"))
                # .where("TIME_CET like '%:00:%'")
    
    windmill = _load_csv(constant.windmill_path, constant.windmills_schema)
    windmill = windmill.where("grid != 0").fillna(0.1)
    ws10_dic = _load_weather_toPandas(constant.ws10_path)
    ws100_dic = _load_weather_toPandas(constant.ws10_path)
    wd10_dic = _load_weather_toPandas(constant.wd10_path)
    wd100_dic = _load_weather_toPandas(constant.wd100_path)
    tools.get_udf(ws10_dic, ws100_dic, wd10_dic, wd100_dic)

    return settlement, windmill


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = F.tanh(self.fc1(x))
    x = F.tanh(self.fc2(x))
    x = self.fc3(x)

    return x


if __name__ == "__main__":
    # initialise sparkContext\
    spark = SparkSession.builder \
            .appName("WindTurbine_ws") \
            .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    settlement, windmill = load_files()
    settlement.persist()
    windmill.persist()

    # upscaling
    feat_cols = ["GSRN", "Turbine_type", "Slope", "roughness", "grid"]
    windmill_W = windmill.select(feat_cols).where("Turbine_type = 'W'")

    train = settlement.where("TIME_CET not like '2019-03-31 02%'").where("TIME_CET not like '2019-12-30%'") \
        .sample(fraction=0.00001, seed=5)
    train = tools.upscaling(train, windmill_W)

    test = settlement.where("TIME_CET like '2019-12-30%'")
    test = tools.upscaling(test, windmill_W)

    train_pd = train.toPandas()
    test_pd = test.toPandas()

    feat_cols = ["month", "hour", "Slope", "roughness","ws10", "ws100","wd10","wd100"]
    x_train = train_pd[feat_cols]
    y_train = train_pd[["VAERDI"]]
    x_test = test_pd[feat_cols]
    y_test = test_pd[["VAERDI"]]
    timeline = test_pd[["TIME_CET"]]

    x_train = tools.normalize_zcenter(x_train).to_numpy()
    x_test = tools.normalize_zcenter(x_test).to_numpy()

    dtype = torch.float
    x_train_tensor = torch.tensor(x_train, dtype = dtype)
    x_test_tensor = torch.tensor(x_test, dtype = dtype)

    y_train_tensor = torch.tensor(y_train.values, dtype = dtype)
    y_test_tensor = torch.tensor(y_test.values, dtype = dtype)

    input_size = 8
    output_size = 1
    hidden_size = 300
    lr = 0.05
    epochs = 101

    model = Net()

    optimizer = optim.Adam(model.parameters(),lr=lr)
    # nn.MSELoss() is implemented by default as: ((input-target)**2).mean()
    loss_fn = torch.nn.MSELoss()

    epoch_data = []

    for epoch in range(1, epochs):
        optimizer.zero_grad()
        y_pred = model(x_train_tensor)

        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()

        optimizer.step()

        #   _,pred = y_pred_test.data.max(1)
        #   accuracy = pred.eq(y_test_tensor.data).sum().item() / y_test.values.size
        epoch_data.append([epoch, loss.data.item()])

    tools.save_list(epoch_data, "epoch_data.txt")
    
    y_pred_tensor = model(x_test_tensor)
    y_pred = y_pred_tensor.detach().numpy()

    y_test["pred"] = pd.Series(y_pred.tolist()).apply(lambda x: x[0])
    y_test["time"] = pd.Series(timeline["TIME_CET"])
    y_test = y_test.groupby(y_test["time"]).agg({"VAERDI":lambda x:sum(x), "pred":lambda x:sum(x)}).reset_index()
    y_test.to_csv("y_test.csv", index=False)

    spark.stop()

    