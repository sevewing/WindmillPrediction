import pandas as pd
import numpy as np
from math import log,sqrt,atan2,pi,cos,sin

# from pyspark.sql.functions import udf, sum, max, avg, first
# from pyspark.sql.types import *
# udic = {}

def to_vector(s, d):
    """
    Convert Wind speed and wind direction to u, v component.
    s: Wind speed
    d: Wind direction
    u: x-axis component
    v: y-axis component
    """
    u = s * cos(pi/180 * (270-d))
    v = s * sin(pi/180 * (270-d))
    return u, v

# thermal_gradient  = lambda v, z, z_hat, a : (t2-t1)/(z2-z1) y x
thermal_interpolation = lambda t1, t2, z1, z2, z_hat : (z_hat - z1) * (t2 - t1)  / (z2 - z1) + t1
pow_law = lambda v, z, z_hat, a : v * ( (z_hat/z) ** a)
log_law = lambda v, z, z_hat, rl : v * np.log(z_hat/rl) / np.log(z/rl)
rn_exponent = lambda rn : 0.096 * np.log10(rn) + 0.016 * np.log10(rn) ** 2 + 0.24

def pow_exponent(v1, v2, z1, z2):
    """
    Calaulate windshear in reverse by two layers
    """
    a1 = abs(v1)
    a2 = abs(v2)
    if v1 < 0 and v2 >= 0 or v1 >= 0 and v2 < 0:
        a2 = a1 + a2
    exp = np.log(a2/(a1+1e-06)+1e-06) / np.log(z2/z1)
    return exp


def wind_vinterp_exp(u1, v1, u2, v2, z_hat):
    """
    Wind u,v component vertical interpolation by a invert calculation of exponent.
    u1, v1 at height(AGL) 10m
    u2, v2 at height(AGL) 100m
    z_hat: a target height
    """
    exp_u = pow_exponent(u1, u2, 10, 100)
    exp_v = pow_exponent(v1, v2, 10, 100)
    u_i = round(pow_law(u1, 10, z_hat, exp_u),3)
    v_i = round(pow_law(v1, 10, z_hat, exp_v),3)
    return u_i, v_i

def wind_vinterp_rn(u, v, z_hat, rl):
    """
    Wind u,v component vertical interpolation by roughness length.
    u, v at height(AGL) 10m
    z_hat: a target height
    rl: roughness length
    """
    u_i = round(pow_law(u, 10, z_hat, rl),3)
    v_i = round(pow_law(v, 10, z_hat, rl),3)
    return u_i, v_i

# def wind_interp_linear(u1, v1, u2, v2, z_hat):
#     u_i = round(linear_law(10, 100, u1, u2, z_hat),3)
#     v_i = round(linear_law(10, 100, v1, v2, z_hat),3)
#     return u_i, v_i


# cols_basic = ["GSRN", "TIME_CET", "Capacity_kw", "Rotor_diameter", "Navhub_height", "Slope", "Roughness", "Elevation", "VAERDI", "grid"]
# cols_basic = ["GSRN","TIME_CET", "Navhub_height", "Roughness", "VAERDI", "grid"]
# cols_ext = ["hour", "wu10", "wv10", "tmp2", "tmp100"]

# def _udf_by_grid(df:pd.DataFrame, tp=DoubleType()):
#     return udf(lambda g, t: df[df['TIME'] == t[:14]+'00:00'][g].tolist()[0] if df['TIME'].astype(str).str.contains(t[:14]+'00:00').any() else 0, tp)


# uv_schema = StructType([
#         StructField("u_interp", DoubleType(), True),
#         StructField("v_interp", DoubleType(), True)
#     ])

# def get_udf(ws10_dic, ws100_dic, wu10_dic,wv10_dic,wu100_dic,wv100_dic,tmp2_dic,tmp100_dic):
#     # udic["type"] = udf(lambda x: {"H": 1, "W": 2, "P": 3, "M": 4}.get(x, 0), IntegerType())
#     # udic["placement"] = udf(lambda x: {"LAND": 1, "HAV": 2}.get(x, 0), IntegerType())
#     udic["month"] = udf(lambda x: int(x[5:7]), IntegerType())
#     udic["hour"] = udf(lambda x: int(x[11:13]), IntegerType())
#     udic["mins"] = udf(lambda x: int(x[14:16]), IntegerType())
#     udic["ws10"]  = _udf_by_grid(ws10_dic)
#     udic["ws100"]  = _udf_by_grid(ws100_dic)
#     udic["wu10"]  = _udf_by_grid(wu10_dic)
#     udic["wv10"]  = _udf_by_grid(wv10_dic)
#     udic["wu100"]  = _udf_by_grid(wu100_dic)
#     udic["wv100"]  = _udf_by_grid(wv100_dic)
#     udic["tmp2"]  = _udf_by_grid(tmp2_dic)
#     udic["tmp100"]  = _udf_by_grid(tmp100_dic)
#     udic["ws_interp_pow"]  = udf(lambda v, z, rl: round(pow_law(v, 10, z, rl),3), DoubleType())
#     udic["wuv_interp_pow"]  = udf(lambda u, v, z, rl: wind_interp_pow(u, v, z, rl), uv_schema)
#     udic["ws_interp_linear"]  = udf(lambda u1, v1, u2, v2, z: wind_interp_linear(u1, v1, u2, v2, z), uv_schema)
#     udic["wuv_interp"]  = udf(lambda u1, v1, u2, v2, z: wind_interp(u1, v1, u2, v2, z), uv_schema)
#     # udic["tmp_interp"]  = udf(lambda v1, v2, z_hat: _pow_interp(v1, v2, 2, 100, z_hat), DoubleType())
#     udic["ws_hub"] = udf(lambda u, v: round((u ** 2 + v ** 2) ** 0.5, 3), DoubleType())
#     udic["ws_^2"] = udf(lambda x: round(x ** 2, 3), DoubleType())
#     udic["ws_^3"] = udf(lambda x: round(x ** 3, 3), DoubleType())

# def get_udf(wu10_dic,wv10_dic,tmp2_dic,tmp100_dic):
#     udic["hour"] = udf(lambda x: int(x[11:13]), IntegerType())
#     udic["wu10"]  = _udf_by_grid(wu10_dic)
#     udic["wv10"]  = _udf_by_grid(wv10_dic)
#     udic["tmp2"]  = _udf_by_grid(tmp2_dic)
#     udic["tmp100"]  = _udf_by_grid(tmp100_dic)

# def _normal_f_extract(df):
#     return df.withColumn("month", udic["month"](df.TIME_CET)) \
#                 .withColumn("hour", udic["hour"](df.TIME_CET)) \
#                 .withColumn("mins", udic["mins"](df.TIME_CET)) \
#                 .withColumn("ws10", udic["ws10"](df.grid, df.TIME_CET)) \
#                 .withColumn("ws100", udic["ws100"](df.grid, df.TIME_CET)) \
#                 .withColumn("wu10", udic["wu10"](df.grid, df.TIME_CET)) \
#                 .withColumn("wv10", udic["wv10"](df.grid, df.TIME_CET)) \
#                 .withColumn("wu100", udic["wu100"](df.grid, df.TIME_CET)) \
#                 .withColumn("wv100", udic["wv100"](df.grid, df.TIME_CET)) \
#                 .withColumn("tmp2", udic["tmp2"](df.grid, df.TIME_CET)) \
#                 .withColumn("tmp100", udic["tmp100"](df.grid, df.TIME_CET))

# def _normal_f_extract(df):
#     return df.withColumn("hour", udic["hour"](df.TIME_CET)) \
#                 .withColumn("wu10", udic["wu10"](df.grid, df.TIME_CET)) \
#                 .withColumn("wv10", udic["wv10"](df.grid, df.TIME_CET)) \
#                 .withColumn("tmp2", udic["tmp2"](df.grid, df.TIME_CET)) \
#                 .withColumn("tmp100", udic["tmp100"](df.grid, df.TIME_CET))

# def _second_f_extract(df):
#     df = df.withColumn("ws_hub", udic["ws_hub"](df.u_interp, df.v_interp))
#     df = df.withColumn("ws_^2", udic["ws_^2"](df.ws_hub))
#     df = df.withColumn("ws_^3", udic["ws_^3"](df.ws_hub)) 
#     return df

# def aggregate(df, join_df):
#     df = df.join(join_df, on="GSRN").select(join_df.columns + ["TIME_CET", "VAERDI"])
#     df = _normal_f_extract(df)
#     return df


# def interp_pow(df, join_df):
#     # cols = cols_basic + cols_ext + ["wsCol.u_interp", "wsCol.v_interp", "tmp_interp"]
#     # cols = join_df.columns + cols_basic + cols_ext + ["wsCol.u_interp", "wsCol.v_interp", "tmp_interp"]
#     df = aggregate(df, join_df)
#     df = df.withColumn("wsCol", udic["wuv_interp_pow"](df.wu10, df.wv10, df.Navhub_height, df.Roughness))
#     cols = df.columns
#     cols.remove("wsCol")
#     df = df.select(cols + ["wsCol.u_interp", "wsCol.v_interp"])
#     # .withColumn("tmp_interp", udic["tmp_interp"](df.tmp2, df.tmp100, df.Navhub_height))
#     return df

# def interp_linear(df, join_df):
#     cols = cols_basic + cols_ext + ["wsCol.u_interp", "wsCol.v_interp", "tmp_interp"]
#     df = aggregate(df, join_df)
#     df = df.withColumn("wsCol", udic["ws_interp_linear"](df.wu10, df.wv10, df.wu100, df.wv100, df.Navhub_height)) \
#                 .select(cols)
#                 # .withColumn("tmp_interp", udic["tmp_interp"](df.tmp2, df.tmp100, df.Navhub_height)) \
#     return df

# def interp(df, join_df):
#     cols = cols_basic + cols_ext + ["wsCol.u_interp", "wsCol.v_interp", "tmp_interp"]
#     df = aggregate(df, join_df)
#     df = df.withColumn("wsCol", udic["wuv_interp"](df.ws10, df.wd10, df.ws100, df.wd100, df.Navhub_height)) \
#                 .select(cols)
#                 # .withColumn("tmp_interp", udic["tmp_interp"](df.tmp2, df.tmp100, 2, 100, df.Navhub_height)) \
#     return df

# def upscaling(df, join_df):
#     df = df.join(join_df, on="GSRN").select(cols_basic)
#     # cols = cols_basic + ["month", "hour", "mins"]
#     df = df.groupby("TIME_CET", "grid") \
#             .agg(first("Slope").alias("Slope"), \
#             first("Roughness").alias("Roughness"), \
#             sum("VAERDI").alias("VAERDI"))
#             # .select(cols)
#     df = _normal_f_extract(df)

#     return df

def normalize_zcenter(df):
    """
    Normalize value to zero centered.
    """
    return (df - df.mean()) / df.std()

def normalize_maxmin(df, max=1, min=0):
    """
    Normalize the data to real values(from new max 0 to new min 1).
    """
    return (df - df.min()) / (df.max() - df.min()) * (max - min) + min

# def normalize_features(df, features):
#     """
#     Normalize features by maxmin, normalize power["VAERDI"] by 'max_VAERDI'
#     """
#     df[features] = normalize_maxmin(df[features])
#     df['VAERDI'] = df['VAERDI'] / df['max_VAERDI'] 
#     df = df.fillna(0)
#     return df

def normalize_features(df, features):
    """
    Normalize features by maxmin, normalize power["VAERDI"] by 'max_VAERDI'
    """
    df[features] = normalize_zcenter(df[features])
    df['VAERDI'] = df['VAERDI'] / df['max_VAERDI'] 
    df = df.fillna(0)
    return df


def save_list(lst, path):
    with open(path, 'w+') as f:
        for i in lst:
            f.write("%s\n" % i) 
