import warnings
warnings.filterwarnings("ignore")

from sparktorch import serialize_torch_obj, SparkTorch, PysparkPipelineWrapper
import torch
import torch.nn as nn
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler

import constant
import tools
from tools import Aggregate, Spk

def load_files(spark):
# to read parquet file
    settlement = spark.load_parquet(constant.settlement_path, constant.settlement_schema)
    settlement = settlement.dropna(subset =["VAERDI"]) \
                .withColumn("VAERDI", settlement["VAERDI"].cast("float"))
                # .where("TIME_CET like '%:00:%'")
    
    windmill = spark.load_windmill(constant.windmill_path, constant.windmills_schema)
    windmill = windmill.where("grid != 0") \
                .fillna(0.1)
                # .withColumn("Turbine_type", aggs.udf_type(windmill.Turbine_type)) \
                # .withColumn("Placement", aggs.udf_placement(windmill.Placement))

    return settlement, windmill


class RegressionNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RegressionNet, self).__init__()
   
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

def get_net_pipeline(inputCol, labelCol,lr,iters, input_size, hidden_size, output_size):
    # Setup features
    vec_assembler = VectorAssembler(inputCols=inputCol, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
            
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        lr=lr
    )

    # Create a SparkTorch Model with torch distributed. Barrier execution is on by default for this mode.
    spark_model = SparkTorch(
        inputCol='scaledFeatures',
        labelCol=labelCol,
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=iters,
        verbose=1
    )

    # Can be used in a pipeline and saved.
    return Pipeline(stages=[vec_assembler, scaler, spark_model])

if __name__ == "__main__":
    # initialise sparkContext\
    spark = Spk()
    aggs = Aggregate(spark)

    settlement, windmill = load_files(spark)
    settlement.persist()
    windmill.persist()

    # upscaling
    feat_cols = ["GSRN", "Turbine_type", "Slope", "roughness", "grid"]
    windmill_W = windmill.select(feat_cols).where("Turbine_type = 'W'")

    train = settlement.where("TIME_CET not like '2019-03-31 02%'").where("TIME_CET not like '2019-12-30%'") \
        .sample(fraction=0.00003, seed=5)
    train = aggs.upscaling(train, windmill_W)
    # train.persist()

    # test = settlement.where("TIME_CET like '2019-12-30%'")
    # test = aggs.upscaling(test, windmill_W)

    feat_cols = ["month", "hour", "Slope", "roughness","ws10", "ws100","wd10","wd100"]
    # Can be used in a pipeline and saved.
    # pl = get_net_pipeline(feat_cols,'VARDI',0.3,50,11,1,300)
    # p = pl.fit(train)

    # p.write().overwrite().save("location")
    # p = PysparkPipelineWrapper.unwrap(PipelineModel.load('location'))

    # predictions = p.transform(test)
    # predictions.write().overwrite().save("predictions")