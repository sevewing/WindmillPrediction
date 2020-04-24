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


def get_net_pipeline(inputCol, labelCol, lr, iters, input_size, hidden_size, output_size):
    # Setup features
    vec_assembler = VectorAssembler(inputCols=inputCol, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )    
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        lr=lr
    )
    # Create a SparkTorch Model with torch distributed. Barrier execution is on by default for this mode.
    spark_model = SparkTorch(
        inputCol='scaledFeatures',
        labelCol=labelCol,
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=iters,
        verbose=1
    )
    # Can be used in a pipeline and saved.
    return Pipeline(stages=[vec_assembler, scaler, spark_model])

if __name__ == "__main__":
    # initialise sparkContext\
    spark = SparkSession.builder \
            .appName("WindTurbine_ws") \
            .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    settlement, windmill = load_files()
    settlement.persist()
    # windmill.persist()

    # upscaling
    # feat_cols = ["GSRN", "Turbine_type", "Slope", "roughness", "grid"]
    windmill_W = windmill.where("Turbine_type = 'W'")
    windmill_W.persist()

    train = settlement.where("TIME_CET not like '2019-03-31 02%'").where("TIME_CET not like '2019-12-30%'") \
        .sample(fraction=0.00001, seed=5)
    train = tools.upscaling(train, windmill_W)

    test = settlement.where("TIME_CET like '2019-12-30%'")
    test = tools.upscaling(test, windmill_W)

    feat_cols = ["month", "hour", "Slope", "roughness","ws10", "ws100","wd10","wd100"]
    # Can be used in a pipeline and saved.
    pl = get_net_pipeline(feat_cols,'VARDI',0.3,50,8,100,1)
    p = pl.fit(train)

    p.write().overwrite().save(constant.model_path)
    # p = PysparkPipelineWrapper.unwrap(PipelineModel.load('location'))

    predictions = p.transform(test)
    predictions.write().overwrite().save("predictions")