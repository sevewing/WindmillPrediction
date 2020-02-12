from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import udf
import pandas as pd

# initialise sparkContext
spark = SparkSession.builder \
    .master('local') \
    .appName('WindTurbine') \
    .config('spark.executor.memory', '8gb') \
    .config("spark.cores.max", "4") \
    .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
sqlContext = SQLContext(sc)

# to read parquet file
settlement = sqlContext.read.parquet('data/ITU_DATA/settlement/2018.parquet')
windmills = sqlContext.read.csv('data/windmill_cleaned.csv', sep=';', header=True)
wind_speed_10m = sqlContext.read.parquet('data/ITU_DATA/prognosis/ENetNEA/wind_speed_10m.parquet')

wind_speed_10m = wind_speed_10m.withColumnRenamed('__index_level_0__', 'TIME')
wind_speed_10m = wind_speed_10m.withColumn("id", monotonically_increasing_id())
wind_speed_10m.createOrReplaceTempView("wind_speed_10m")
wind_speed_10m = spark.sql(
    "select * from wind_speed_10m where id in (select max(id) as id from wind_speed_10m group by TIME)")
# wind_speed_10m.createOrReplaceTempView("wind_speed_10m")

settlement.createOrReplaceTempView("settlement")
settlement = spark.sql("select s.GSRN, s.VAERDI, s.TIME_CET from settlement s where s.TIME_CET like '%:00:%'")
settlement.createOrReplaceTempView("settlement")

windmills.createOrReplaceTempView("windmills")
windmills = spark.sql("select * from windmills where grid != 0")
windmills.createOrReplaceTempView("windmills")

test = wind_speed_10m.toPandas()

grid = udf(lambda g, t: list(test[test['TIME'] == t][g])[0], DoubleType())
_ = spark.udf.register("extractInfo", grid)

basicDF = spark.sql("select s.GSRN, s.TIME_CET, s.VAERDI, w.Turbine_type, w.Parent_GSRN, w.Placement, w.UTM_x, w.UTM_y, "
          "w.Capacity_kw, w.Rotor_diameter, w.Navhub_height, w.grid, w.grid_in_range, extractInfo(w.grid, s.TIME_CET) as wind_speed_10 from settlement s join windmills w on s.GSRN == w.GSRN")
basicDF.createOrReplaceTempView("basicDF")
basicDF.show(1)

#
# spark.sql("select b.GSRN, b.TIME, b.VAERDI, b.Turbine_type, b.Parent_GSRN, b.Placement, b.UTM_x, b.UTM_y, "
#           "b.Capacity_kw, b.Rotor_diameter, b.Navhub_height, b.grid, b.grid_in_range, extractInfo(b.grid, w.TIME) as wind_speed "
#           "from basicDF b join wind_speed_10m w on b.TIME == w.TIME").show(1)
