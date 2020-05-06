import pandas as pd
from math import log,sqrt,atan2,pi,cos,sin

from pyspark.sql.functions import udf, sum, max, avg, first
from pyspark.sql.types import *

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


cols_basic = ["GSRN", "TIME_CET", "Turbine_type", "Placement", "Capacity_kw", "Rotor_diameter", "Navhub_height", "Slope", "roughness", "VAERDI", "grid"]

def _udf_by_grid(df:pd.DataFrame, type = FloatType()):
    return udf(lambda g, t: list(df[df['TIME'] == t[:14]+'00:00'][g])[0] if df['TIME'].astype(str).str.contains(t[:14]+'00:00').any() else 0, type)

def _udf_by_ws():
    schema = StructType([
        StructField("u_interp", FloatType(), True),
        StructField("v_interp", FloatType(), True)
    ])
    return udf(lambda s1, d1, s2, d2, z: wind_interp(s1, d1, s2, d2, z), schema)

udic={}
def get_udf(ws10_dic,ws100_dic,wd10_dic,wd100_dic,tmp2_dic,tmp100_dic):
    udic["type"] = udf(lambda x: {"H": 1.0, "W": 2.0, "P": 3.0, "M": 4.0}.get(x, 0.0), FloatType())
    udic["placement"] = udf(lambda x: {"LAND": 1.0, "HAV": 2.0}.get(x, 0.0), FloatType())
    udic["month"] = udf(lambda x: int(x[5:7]), IntegerType())
    udic["hour"] = udf(lambda x: int(x[11:13]), IntegerType())
    udic["mins"] = udf(lambda x: int(x[14:16]), IntegerType())
    udic["ws10"]  = _udf_by_grid(ws10_dic, FloatType())
    udic["ws100"]  = _udf_by_grid(ws100_dic, FloatType())
    udic["wd10"]  = _udf_by_grid(wd10_dic, IntegerType())
    udic["wd100"]  = _udf_by_grid(wd100_dic, IntegerType())
    udic["tmp2"]  = _udf_by_grid(tmp2_dic, FloatType())
    udic["tmp100"]  = _udf_by_grid(tmp100_dic, FloatType())
    udic["ws_interp"]  = _udf_by_ws()


def _normal_f_extract(df):
    return df.withColumn("month", udic["month"](df.TIME_CET)) \
                .withColumn("hour", udic["hour"](df.TIME_CET)) \
                .withColumn("mins", udic["mins"](df.TIME_CET)) \
                .withColumn("ws10", udic["ws10"](df.grid, df.TIME_CET)) \
                .withColumn("ws100", udic["ws100"](df.grid, df.TIME_CET)) \
                .withColumn("wd10", udic["wd10"](df.grid, df.TIME_CET)) \
                .withColumn("wd100", udic["wd100"](df.grid, df.TIME_CET)) \
                .withColumn("tmp2", udic["tmp2"](df.grid, df.TIME_CET)) \
                .withColumn("tmp100", udic["tmp100"](df.grid, df.TIME_CET))
                
def aggregate(df, join_df):
    df = df.join(join_df, on="GSRN").select(cols_basic)
    df = _normal_f_extract(df) \
        .withColumn("Turbine_type", udic["type"](df.Turbine_type)) \
        .withColumn("Placement", udic["placement"](df.Placement))
    return df

def interp(df, join_df):
    cols = cols_basic + ["month", "hour", "mins","wsCol.u_interp", "wsCol.v_interp"]
    df = aggregate(df, join_df)
    df = df.withColumn("wsCol", \
                udic["ws_interp"](df.ws10, df.wd10, df.ws100, df.wd100, df.Navhub_height)) \
                .select(cols)
    return df

def upscaling(df, join_df):
    df = df.join(join_df, on="GSRN").select(cols_basic)
    # cols = cols_basic + ["month", "hour", "mins"]
    df = df.groupby("TIME_CET", "grid") \
            .agg(first("Slope").alias("Slope"), \
            first("roughness").alias("roughness"), \
            sum("VAERDI").alias("VAERDI"))
            # .select(cols)
    df = _normal_f_extract(df)

    return df

def normalize_zcenter(df):
    """
    Normalize value to zero centered.
    """
    return (df - df.mean()) / df.std()

def normalize_maxmin(data, max=1, min=0):
    """
    Normalize the data to real values(from new max 0 to new min 1).
    """
    return (data - data.min()) / (data.max() - data.min()) * (max - min) + min

def save_list(lst, path):
    with open(path, 'w+') as f:
        for i in lst:
            f.write("%s\n" % i) 


