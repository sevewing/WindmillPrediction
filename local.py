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
                    .withColumn("id", monotonically_increasing_id())
    weather.createOrReplaceTempView("weather_temp")
    weather_dic = spark.sql("select * from weather_temp where id in (select max(id) as id from weather_temp group by TIME)").toPandas()
    return weather_dic

def load_csv(path, schema:StructType = None):
    """
    load csv file
    """
    return sqlContext.read.schema(schema).csv(path, sep=";", header=True, schema=schema) if schema is not None else sqlContext.read.schema(schema).csv(path, sep=";", header=True)

def regist_udf_grid(name:str, df:pd.DataFrame, type = DoubleType()):
    _ = spark.udf.register(name, lambda g, t: list(df[df['TIME'] == t][g])[0], type)

# def regist_udf_ws(name:str):
    # _ = spark.udf.register(name, lambda s1, d1, s2, d2: tools.wind_convert(s1, d1, s2, d2), DoubleType())


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
    settlement = load_parquet("data/ITU_DATA/settlement/2018.parquet" ,schemas.settlement_schema) \
                 .fillna({"VAERDI":0})
    settlement = settlement.withColumn("VAERDI", settlement["VAERDI"].cast(DoubleType())).where("TIME_CET like '%:00:%'").select("*")
    settlement.createOrReplaceTempView("settlement")

    windmills = load_csv("data/windmill_cleaned.csv", schemas.windmills_schema)
    windmills.createOrReplaceTempView("windmills")
    # windmills = spark.sql("select * from windmills where grid != 0")
    # windmills.createOrReplaceTempView("windmills")

    ws10_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet")
    ws100_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_speed_100m.parquet")
    wd10_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m.parquet")
    wd100_dic = load_parquet_from_weather_toPandas("data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m.parquet")

    regist_udf_grid("extract_ws10", ws10_dic, DoubleType())
    regist_udf_grid("extract_ws100", ws100_dic, DoubleType())
    regist_udf_grid("extract_wd10", wd10_dic, IntegerType())
    regist_udf_grid("extract_wd100", wd100_dic, IntegerType())
    # regist_udf_ws("interp_ws")


    schema = StructType([
        StructField("u_interp", DoubleType(), True),
        StructField("v_interp", DoubleType(), True)
    ])
    udf_ws = udf(lambda s1, d1, s2, d2, z: tools.wind_convert(s1, d1, s2, d2, z), schema)

# basicDF = spark.sql("select s.GSRN, s.TIME_CET, s.VAERDI, w.Turbine_type, w.Parent_GSRN, w.Placement, w.UTM_x, w.UTM_y, "
#                 "w.Capacity_kw, w.Rotor_diameter, w.Navhub_height, w.grid, w.grid_in_range, "
#                 "extract_ws10(w.grid, s.TIME_CET) as ws10, extract_ws100(w.grid, s.TIME_CET) as ws100 "
#                 "from settlement s join windmills w on s.GSRN = w.GSRN where w.grid != 0")

    basicDF = spark.sql(
    "select s.GSRN, s.TIME_CET, w.Navhub_height, "
    "extract_ws10(w.grid, s.TIME_CET) as ws10, extract_ws100(w.grid, s.TIME_CET) as ws100, "
    "extract_wd10(w.grid, s.TIME_CET) as wd10, extract_wd100(w.grid, s.TIME_CET) as wd100 "
    "from settlement s join windmills w on s.GSRN == w.GSRN where w.grid != 0")
    basicDF.withColumn("wsCol", udf_ws("ws10","wd10","ws100","wd10","Navhub_height")) \
            .select("wsCol.u_interp", "wsCol.v_interp").show(1)
    # basicDF.createOrReplaceTempView("basicDF")



# .select("wsCol(0)".alias("u_interp"), "wsCol"(1).alias("v_interp"))
    
    
