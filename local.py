import pandas as pd
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import *
import warnings
warnings.filterwarnings("ignore")

import schemas
import tools

def load_parquet(path, schema:StructType = None):
    """
    load apache parquet file
    """
    return sqlContext.read.schema(schema).parquet(path) if schema is not None else sqlContext.read.parquet(path)

def load_parquet_from_weather_toPandas(path, schema:StructType = None):
    """
    load apache parquet file
    """
    weather = load_parquet(path) \
                    .withColumnRenamed("__index_level_0__", "TIME") \
                    .dropna() \
                    .withColumn("id", monotonically_increasing_id())
    weather.createOrReplaceTempView("weather_temp")
    weather_dic = spark.sql("select * from weather_temp where id in (select max(id) as id from weather_temp group by TIME)").toPandas()
    return weather_dic

def load_csv(path, schema:StructType = None):
    """
    load csv file
    """
    # return sqlContext.read.schema(schema).csv(path, sep=";", header=True, schema=schema) if schema is not None else sqlContext.read.schema(schema).csv(path, sep=";", header=True)
    return sqlContext.read.csv(path, sep=";", header=True, schema=schema)

def udf_by_grid(df:pd.DataFrame, type = DoubleType()):
    return udf(lambda g, t: list(df[df['TIME'] == t][g])[0], type)

def udf_by_ws():
    schema = StructType([
        StructField("u_interp", DoubleType(), True),
        StructField("v_interp", DoubleType(), True)
    ])
    return udf(lambda s1, d1, s2, d2, z: tools.wind_interp(s1, d1, s2, d2, z), schema)

def udf_regist():
    udf_ws10  = udf_by_grid(ws10_dic, DoubleType())
    udf_ws100  = udf_by_grid(ws100_dic, DoubleType())
    udf_wd10  = udf_by_grid(wd10_dic, IntegerType())
    udf_wd100  = udf_by_grid(wd100_dic, IntegerType())
    udf_ws_interp  = udf_by_ws()
    return udf_ws10, udf_ws100, udf_wd10, udf_wd100, udf_ws_interp


if __name__ == "__main__":
    # initialise sparkContext\
    spark = SparkSession.builder \
        .master("local") \
        .appName("WindTurbine") \
        .config("spark.executor.memory", "8gb") \
        .config("spark.cores.max", "4") \
        .getOrCreate()

    sc = spark.sparkContext

    # using SQLContext to read parquet file
    sqlContext = SQLContext(sc)
       
    # to read parquet file
    settlement = load_parquet("data/ITU_DATA/settlement/2019.parquet" ,schemas.settlement_schema)
    settlement = settlement.dropna(subset =["VAERDI"]) \
                .withColumn("VAERDI", settlement["VAERDI"].cast("double")) \
                .where("TIME_CET like '%:00:%'")
    settlement = settlement.sample(fraction=0.00001, seed=3)

    windmills = load_csv("data/windmill_cleaned.csv", schemas.windmills_schema) \
                .where("Navhub_height is not null") \
                .where("grid != 0")

    ws10_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet")
    ws100_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_speed_100m.parquet")
    wd10_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m.parquet")
    wd100_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m.parquet")

    udf_ws10, udf_ws100, udf_wd10, udf_wd100, udf_ws_interp = udf_regist()

    basicDF = settlement.join(windmills, on="GSRN") \
                    .withColumn("ws10", udf_ws10(windmills.grid, settlement.TIME_CET)) \
                    .withColumn("ws100", udf_ws100(windmills.grid, settlement.TIME_CET)) \
                    .withColumn("wd10", udf_wd10(windmills.grid, settlement.TIME_CET)) \
                    .withColumn("wd100", udf_wd100(windmills.grid, settlement.TIME_CET))


