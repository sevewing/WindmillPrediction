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
    StructField("Capacity_kw", DoubleType(), True),
    StructField("Rotor_diameter", DoubleType(), True),
    StructField("Navhub_height", DoubleType(), False),
    StructField("grid", StringType(), False),
    StructField("grid_in_range",StringType(),True)
])						