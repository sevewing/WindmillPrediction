from pyspark.sql.types import *

# windmill_path = "./windmill_cleaned.csv"
# settlement_path = "/datasets/energinet/settlement/2019.parquet"
# ws10_path = "/datasets/energinet/prognosis/ENetNEA/wind_speed_10m.parquet"
# ws100_path = "/datasets/energinet/prognosis/ENetNEA/wind_speed_100m.parquet"
# wd10_path = "/datasets/energinet/prognosis/ENetNEA/wind_direction_10m.parquet"
# wd100_path = "/datasets/energinet/prognosis/ENetNEA/wind_direction_100m.parquet" 
# model_path = "./model.parquet"

windmill_path = "data/windmill_cleaned.csv"
settlement_path = "data/ITU_DATA/settlement/2019.parquet"
ws10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet"
ws100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_100m.parquet"
wd10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m.parquet"
wd100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m.parquet" 
model_path = "model.parquet"

settlement_schema = StructType([
    StructField("GSRN", StringType(), False),
    StructField("VAERDI", StringType(), False),
    StructField("TIME_CET", StringType(), False)
])

windmills_schema = StructType([
    StructField("GSRN",StringType(),False),
    StructField("Turbine_type",StringType(),True),
    StructField("Parent_GSRN",StringType(),True),
    StructField("BBR_municipal",StringType(),True),
    StructField("Placement",StringType(),True),
    StructField("UTM_x",StringType(),True),
    StructField("UTM_y",StringType(),True),
    StructField("Capacity_kw", FloatType(), True),
    StructField("Rotor_diameter", FloatType(), True),
    StructField("Navhub_height", FloatType(), False),
    StructField("grid", StringType(), False),
    StructField("grid_in_range",StringType(),True),
    StructField("Land_cover", FloatType(), True),
    StructField("Slope", FloatType(), True),
    StructField("roughness", FloatType(), True)
])						