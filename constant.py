from pyspark.sql.types import *

# windmill_same_cond_path = "./data/windmills_same_cond.csv"
# settlement_12_path = "./data/ITU_DATA/settlement/201819_12_cleaned.parquet"

# ws10_c_path = "./data/ITU_DATA/prognosis/ENetNEA/ws_10m_cmp.parquet"
# ws100_c_path = "./data/ITU_DATA/prognosis/ENetNEA/ws_100m_cmp.parquet"
# wu10_path = "./data/ITU_DATA/prognosis/ENetNEA/wu_10m.parquet"
# wv10_path = "./data/ITU_DATA/prognosis/ENetNEA/wv_10m.parquet"
# wu100_path = "./data/ITU_DATA/prognosis/ENetNEA/wu_100m.parquet"
# wv100_path = "./data/ITU_DATA/prognosis/ENetNEA/wv_100m.parquet"
# tmp2_c_path = "./data/ITU_DATA/prognosis/ENetNEA/temperatur_2m_compress.parquet" 
# tmp100_c_path = "./data/ITU_DATA/prognosis/ENetNEA/temperatur_100m_compress.parquet" 
# wd10_path = "/datasets/energinet/prognosis/ENetNEA/wind_direction_10m.parquet"
# wd100_path = "/datasets/energinet/prognosis/ENetNEA/wind_direction_100m.parquet" 
# model_path = "./model.parquet"

windmill_path = "data/windmill_cleaned.csv"
windmill_one_path = "data/windmill_570714700000004070.csv"
windmill_LW_600_path = "data/windmill_LW_600.csv"
windmill_geo_analysis_path = "data/windmill_for_geoanalysis.csv"
windmill_same_cond_path = "data/windmills_same_cond.csv"

settlement_path = "data/ITU_DATA/settlement/201{8,9}.parquet"
settlement_2019_one_path = "data/ITU_DATA/settlement/2019_570714700000004070.parquet"
settlement_201819_one_path = "data/ITU_DATA/settlement/201819_570714700000004070.parquet"
settlement_2019_path = "data/ITU_DATA/settlement/2019_cleaned.parquet"
settlement_12_path = "data/ITU_DATA/settlement/201819_12_cleaned.parquet"
settlement_201812_path = "data/ITU_DATA/settlement/2018_12.parquet"
settlement_201912_path = "data/ITU_DATA/settlement/2019_12_cleaned.parquet"
# settlement_570714700000000362_path = "data/ITU_DATA/settlement/2019_570714700000000362.parquet"

wu10_path = "data/ITU_DATA/prognosis/ENetNEA/wu_10m.parquet"
wv10_path = "data/ITU_DATA/prognosis/ENetNEA/wv_10m.parquet"
wu100_path = "data/ITU_DATA/prognosis/ENetNEA/wu_100m.parquet"
wv100_path = "data/ITU_DATA/prognosis/ENetNEA/wv_100m.parquet"

# wui10_path = "data/ITU_DATA/prognosis/ENetNEA/ws_ui_10m.parquet"
# wvi10_path = "data/ITU_DATA/prognosis/ENetNEA/ws_vi_10m.parquet"
# wui100_path = "data/ITU_DATA/prognosis/ENetNEA/ws_ui_100m.parquet"
# wvi100_path = "data/ITU_DATA/prognosis/ENetNEA/ws_vi_100m.parquet"

# ws10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet"
# ws100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_100m.parquet"
# wd10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m.parquet"
# wd100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m.parquet" 

tmp2_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_2m.parquet" 
tmp100_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_100m.parquet" 

ws10_path = "data/ITU_DATA/prognosis/ENetNEA/ws_10m_cmp.parquet"
ws100_path = "data/ITU_DATA/prognosis/ENetNEA/ws_100m_cmp.parquet"
wd10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m_compress.parquet"
wd100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m_compress.parquet" 
tmp2_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_2m_compress.parquet" 
tmp100_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_100m_compress.parquet" 

# model_path = "model.parquet"


settlement_schema = StructType([
    StructField("GSRN", StringType(), False),
    StructField("VAERDI", StringType(), False),
    StructField("TIME_CET", StringType(), False)
])
settlement_schema_2 = StructType([
    StructField("GSRN", StringType(), False),
    StructField("VAERDI", DoubleType(), False),
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
    StructField("Navhub_height", DoubleType(), True),
    StructField("grid", StringType(), True),
    StructField("grid_in_range",StringType(),True),
    StructField("Land_cover", DoubleType(), True),
    StructField("Slope", DoubleType(), True),
    StructField("Elevation", DoubleType(), True),
    StructField("Roughness", DoubleType(), True)
])


windmills_schema_2 = StructType([
    StructField("GSRN",StringType(),False),
    StructField("Capacity_kw", DoubleType(), True),
    StructField("Rotor_diameter", DoubleType(), True),
    StructField("Navhub_height", DoubleType(), True),
    StructField("Slope", DoubleType(), True),
    StructField("Elevation", DoubleType(), True),
    StructField("Roughness", DoubleType(), True),
    StructField("grid", StringType(), True),
])	

windmills_schema_3 = StructType([
    StructField("GSRN",StringType(),False),
    StructField("Turbine_type",StringType(),True),
    StructField("Placement",StringType(),True),
    StructField("UTM_x",StringType(),True),
    StructField("UTM_y",StringType(),True),
    StructField("Capacity_kw", DoubleType(), True),
    StructField("Rotor_diameter", DoubleType(), True),
    StructField("Navhub_height", DoubleType(), True),
    StructField("grid", StringType(), True),
    StructField("Slope", DoubleType(), True),
    StructField("Elevation", DoubleType(), True),
    StructField("Roughness", DoubleType(), True)
])