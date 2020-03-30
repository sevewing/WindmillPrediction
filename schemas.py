from pyspark.sql.types import *

settlement_schema = StructType([
    StructField("GSRN", StringType(), False),
    # StructField("TS_ID", StringType(), True),
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