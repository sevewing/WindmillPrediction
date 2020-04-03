from math import log,sqrt,atan2,pi,cos,sin
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.sql.types import *
import pandas as pd

pow_law = lambda v2, z2, z_hat, a : v2 * ( (z_hat/z2) ** a) + 1e-06

def wind_speed_to_vector(s, d):
    u = s * cos(pi/180 * (270-d))
    v = s * sin(pi/180 * (270-d))
    return u, v

def _pow_interpolation(v1, v2, z1, z2, z_hat):
    a1 = abs(v1)
    a2 = abs(v2)
    if v1 < 0 and v2 >= 0 or v1 >= 0 and v2 < 0:
        a2 = a1 + a2
    # elif s1 >= 0 and s2 < 0:
    #     a2 = a1 + a2

    a = log(a2/(a1+1e-06)+1e-06) / log(z2/z1)
    
    return round(pow_law(v2, z2, z_hat, a),3)

def wind_interp(s1, d1, s2, d2, z_hat):
    u1, v1 = wind_speed_to_vector(s1, d1)
    u2, v2 = wind_speed_to_vector(s2, d2)
    u_i = _pow_interpolation(u1, u2, 10, 100, z_hat)
    v_i = _pow_interpolation(v1, v2, 10, 100, z_hat)
    return u_i, v_i


cols = ["GSRN", "TIME_CET", "Turbine_type", "Placement", "Capacity_kw", "Rotor_diameter", "Navhub_height", "Slope", "roughness", "VAERDI"]

class Aggregate:
    def __init__(self, ws10_dic, ws100_dic, wd10_dic, wd100_dic):
        self.udf_type = udf(lambda x: {"H": 1.0, "W": 2.0, "P": 3.0, "M": 4.0}.get(x, 0.0), FloatType())
        self.udf_placement = udf(lambda x: {"LAND": 1.0, "HAV": 2.0}.get(x, 0.0), FloatType())
        self.udf_month = udf(lambda x: int(x[5:7]), IntegerType())
        self.udf_hour = udf(lambda x: int(x[11:13]), IntegerType())
        self.udf_ws10  = self.udf_by_grid(ws10_dic, FloatType())
        self.udf_ws100  = self.udf_by_grid(ws100_dic, FloatType())
        self.udf_wd10  = self.udf_by_grid(wd10_dic, IntegerType())
        self.udf_wd100  = self.udf_by_grid(wd100_dic, IntegerType())
        self.udf_ws_interp  = self.udf_by_ws()

    def udf_by_grid(self, df:pd.DataFrame, type = FloatType()):
        return udf(lambda g, t: list(df[df['TIME'] == t[:14]+'00:00'][g])[0], type)

    def udf_by_ws(self):
        schema = StructType([
            StructField("u_interp", FloatType(), True),
            StructField("v_interp", FloatType(), True)
        ])
        return udf(lambda s1, d1, s2, d2, z: wind_interp(s1, d1, s2, d2, z), schema)

    def aggregate(self, df, join_df):
        df = df.join(join_df, on="GSRN") \
                        .select(cols,"grid")

        df = df.withColumn("month", self.udf_month(df.TIME_CET)) \
                .withColumn("hour", self.udf_hour(df.TIME_CET)) \
                .withColumn("ws10", self.udf_ws10(df.grid, df.TIME_CET)) \
                .withColumn("ws100", self.udf_ws100(df.grid, df.TIME_CET)) \
                .withColumn("wd10", self.udf_wd10(df.grid, df.TIME_CET)) \
                .withColumn("wd100", self.udf_wd100(df.grid, df.TIME_CET))
   
        return df

    def aggregate_with_interp(self, df, join_df):
        df = aggregate(df, join_df)
        df = df.withColumn("wsCol", \
                    self.udf_ws_interp(df.ws10, df.wd10, df.ws100, df.wd100, df.Navhub_height)) \
                    .select(cols, "month", "hour", "wsCol.u_interp", "wsCol.v_interp")
        return df

class Spk:
    def __init__(self):
        # initialise sparkContext\
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("WindTurbine") \
            .config("spark.executor.memory", "8g") \
            .config("spark.cores.max", "4") \
            .getOrCreate()

        sc = self.spark.sparkContext

        # using SQLContext to read parquet file
        self.sqlContext = SQLContext(sc)

    def load_parquet(self, path, schema:StructType = None):
        """
        load apache parquet file
        """
        return self.sqlContext.read.schema(schema).parquet(path) if schema is not None else self.sqlContext.read.parquet(path)

    def load_weather_toPandas(self, path, schema:StructType = None):
        """
        load apache parquet file
        """
        weather = self.load_parquet(path) \
                        .withColumnRenamed("__index_level_0__", "TIME") \
                        .dropna() \
                        .withColumn("id", monotonically_increasing_id())
        weather.createOrReplaceTempView("weather_temp")
        weather_dic = self.spark.sql("select * from weather_temp where id in (select max(id) as id from weather_temp group by TIME)").toPandas()
        return weather_dic

    def load_windmill(self, path, schema:StructType = None):
        """
        load csv file
        """
        return self.sqlContext.read.csv(path, sep=",", header=True, schema=schema)