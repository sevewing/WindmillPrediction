# from pyspark.sql.types import *

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

windmill_path = "./data/windmill_cleaned.csv"
windmill_park_path = "./data/windmill_park_cleaned.csv"
# windmill_geo_analysis_path = "data/windmill_for_geoanalysis.csv"
windmill_SL_SC_path = "./data/windmill_SL_SC.csv"
windmill_SH_SC_path = "./data/windmill_SH_SC.csv"
windmill_PL_SC_path = "./data/windmill_PL_SC.csv"
windmill_PH_SC_path = "./data/windmill_PH_SC.csv"

SL_SC_TRAIN_path = "./traintestdata/SL_SC_TRAIN.parquet"
SL_SC_EVL_path = "./traintestdata/SL_SC_EVL.parquet"
SH_SC_TRAIN_path = "./traintestdata/SH_SC_TRAIN.parquet"
SH_SC_EVL_path = "./traintestdata/SH_SC_EVL.parquet"
PL_SC_TRAIN_path = "./traintestdata/PL_SC_TRAIN.parquet"
PL_SC_EVL_path = "./traintestdata/PL_SC_EVL.parquet"
PH_SC_TRAIN_path = "./traintestdata/PH_SC_TRAIN.parquet"
PH_SC_EVL_path = "./traintestdata/PH_SC_EVL.parquet"

settlement_path = "./data/ITU_DATA/settlement/201{8,9}_cleaned.parquet"
settlement_2018_path = "./data/ITU_DATA/settlement/2018_cleaned.parquet"
settlement_2019_path = "./data/ITU_DATA/settlement/2019_cleaned.parquet"
settlement_train_path = "./data/ITU_DATA/settlement/20189_train_cleaned.parquet"

wu10_path = "./data/ITU_DATA/prognosis/ENetNEA/wu_10m_cmp.parquet"
wv10_path = "./data/ITU_DATA/prognosis/ENetNEA/wv_10m_cmp.parquet"
wu100_path = "./data/ITU_DATA/prognosis/ENetNEA/wu_100m_cmp.parquet"
wv100_path = "./data/ITU_DATA/prognosis/ENetNEA/wv_100m_cmp.parquet"

ws10_path = "./data/ITU_DATA/prognosis/ENetNEA/ws_10m_cmp.parquet"
ws100_path = "./data/ITU_DATA/prognosis/ENetNEA/ws_100m_cmp.parquet"
wd10_path = "./data/ITU_DATA/prognosis/ENetNEA/wd_10m_cmp.parquet"
wd100_path = "./data/ITU_DATA/prognosis/ENetNEA/wd_100m_cmp.parquet" 
tmp2_path = "./data/ITU_DATA/prognosis/ENetNEA/tmp_2m_cmp.parquet" 
tmp100_path = "./data/ITU_DATA/prognosis/ENetNEA/tmp_100m_cmp.parquet" 

# ws10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet"
# ws100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_speed_100m.parquet"
# wd10_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_10m.parquet"
# wd100_path = "data/ITU_DATA/prognosis/ENetNEA/wind_direction_100m.parquet" 
# tmp2_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_2m.parquet" 
# tmp100_path = "data/ITU_DATA/prognosis/ENetNEA/temperatur_100m.parquet" 

error_path = "./result/errors/"
model_path = "./result/model/"
plot_path = "./result/plot/"


org_cols = ['Capacity_kw', 'Navhub_height', 'Rotor_diameter', 
'wu10', 'wv10', 'wu100', 'wv100', 'tmp2', 'tmp100',
'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']
wsr_cols = ['Capacity_kw','Rotor_diameter',
'hws_u_wsr','hws_v_wsr','hws_uv_wsr','hws_uv_wsr^2','hws_uv_wsr^3','htmp_exp',
'hws_uv_wsr_q0.1','hws_uv_wsr_q0.5','hws_uv_wsr_q0.9',
'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']
geo_cols = ['Capacity_kw', 'Rotor_diameter',
'Slope', 'Aspect', 
'hws_u_rn','hws_v_rn','hws_uv_rn','hws_uv_rn^2','hws_uv_rn^3','htmp_exp', 
'hws_uv_rn_q0.1','hws_uv_rn_q0.5','hws_uv_rn_q0.9',
'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']
semigeo_cols = ['Capacity_kw', 'Rotor_diameter',
'Slope', 'Aspect', 'Roughness',
'hws_u_wsr','hws_v_wsr','hws_uv_wsr','hws_uv_wsr^2','hws_uv_wsr^3','htmp_exp', 
'hws_uv_wsr_q0.1','hws_uv_wsr_q0.5','hws_uv_wsr_q0.9',
'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']


# settlement_schema = StructType([
#     StructField("GSRN", StringType(), False),
#     StructField("VAERDI", StringType(), False),
#     StructField("TIME_CET", StringType(), False)
# ])
# settlement_schema_2 = StructType([
#     StructField("GSRN", StringType(), False),
#     StructField("VAERDI", DoubleType(), False),
#     StructField("TIME_CET", StringType(), False)
# ])

# windmills_schema = StructType([
#     StructField("GSRN",StringType(),False),
#     StructField("Turbine_type",StringType(),True),
#     StructField("Parent_GSRN",StringType(),True),
#     StructField("BBR_municipal",StringType(),True),
#     StructField("Placement",StringType(),True),
#     StructField("UTM_x",StringType(),True),
#     StructField("UTM_y",StringType(),True),
#     StructField("Capacity_kw", DoubleType(), True),
#     StructField("Rotor_diameter", DoubleType(), True),
#     StructField("Navhub_height", DoubleType(), True),
#     StructField("grid", StringType(), True),
#     StructField("grid_in_range",StringType(),True),
#     StructField("Land_cover", DoubleType(), True),
#     StructField("Slope", DoubleType(), True),
#     StructField("Elevation", DoubleType(), True),
#     StructField("Roughness", DoubleType(), True)
# ])


# windmills_schema_2 = StructType([
#     StructField("GSRN",StringType(),False),
#     StructField("Capacity_kw", DoubleType(), True),
#     StructField("Rotor_diameter", DoubleType(), True),
#     StructField("Navhub_height", DoubleType(), True),
#     StructField("Slope", DoubleType(), True),
#     StructField("Elevation", DoubleType(), True),
#     StructField("Roughness", DoubleType(), True),
#     StructField("grid", StringType(), True),
# ])	

# windmills_schema_3 = StructType([
#     StructField("GSRN",StringType(),False),
#     StructField("Turbine_type",StringType(),True),
#     StructField("Placement",StringType(),True),
#     StructField("UTM_x",StringType(),True),
#     StructField("UTM_y",StringType(),True),
#     StructField("Capacity_kw", DoubleType(), True),
#     StructField("Rotor_diameter", DoubleType(), True),
#     StructField("Navhub_height", DoubleType(), True),
#     StructField("grid", StringType(), True),
#     StructField("Slope", DoubleType(), True),
#     StructField("Elevation", DoubleType(), True),
#     StructField("Roughness", DoubleType(), True)
# ])