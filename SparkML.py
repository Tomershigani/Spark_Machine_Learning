# -*- coding: utf-8 -*-



import findspark
findspark.init()
from pyspark.sql import SparkSession

def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc

spark, sc = init_spark('demo')

df = spark.read.json("data.json")
df.printSchema()
df.show()

# import datetime
from pyspark.sql.functions import *
data = df
data = data.withColumn("arrival",  from_unixtime(data.Arrival_Time/1000))
data = data.withColumn("creation",  from_unixtime(data.Creation_Time/1000000000))
data.filter(data.arrival > data.creation).show( 100, truncate=False)

from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.sql.functions import col, when

#Orgenazing the data

devic_typs = data.select('User').rdd.flatMap(lambda x: x).collect()
devic_typs = list(dict.fromkeys(devic_typs))

data = data.filter(data.User!='null')

data = data.withColumn("user_id", when(col("User")=='a', 1).when(col("User")=='b', 2)
                           .when(col("User")=='c', 3).when(col("User")=='d', 4).
                           when(col("User")=='e', 5). when(col("User")=='f', 6)\
                           .when(col("User")=='g', 7).when(col("User")=='h', 8)\
                            .when(col("User")=='i', 9))



colm = data.columns
assembler = VectorAssembler(
    inputCols=["user_id","x", "y", "z","Index","Creation_Time"],
    outputCol="features")


output = assembler.transform(data)

output = output.filter(output.gt!='null')

output = output.withColumn("label", when(col("gt")=='stand', 1).when(col("gt")=='sit', 2)
                           .when(col("gt")=='walk', 3).when(col("gt")=='stairsup', 4).
                           when(col("gt")=='stairsdown', 5). when(col("gt")=='bike', 6))


output = output.drop(*colm)

gt_typs = output.select('label').rdd.flatMap(lambda x: x).collect()
gt_typs = list(dict.fromkeys(gt_typs))

output_sample = output.sample(False, 0.05)
train, test = output_sample.randomSplit([0.7, 0.3])

#Cross validation          
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator,TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MultilabelMetrics


# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")#,maxBins=42,maxDepth=10,numTrees=50)
# Evaluate model
rfevaluator = MulticlassClassificationEvaluator()

# Create ParamGrid for Cross Validation
rfparamGrid = (ParamGridBuilder()\
             .addGrid(rf.maxDepth, [5,10,20])\
             .addGrid(rf.maxBins, [32,42,52])\
             .addGrid(rf.numTrees, [20,50,70])\
             .build())


rfcv = TrainValidationSplit(
      estimator=rf,
      estimatorParamMaps=rfparamGrid,
      evaluator= rfevaluator,
      trainRatio=0.8)  # data is separated by 80% and 20%, in which the former is used for training and the latter for evaluation


# Run cross validations.
rfcvModel = rfcv.fit(train)

print(rfcvModel)

# Use test set here so we can measure the accuracy of our model on new data
rfpredictions = rfcvModel.transform(test)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
print('Accuracy:', rfevaluator.evaluate(rfpredictions))

list(zip(rfcvModel.validationMetrics, rfcvModel.getEstimatorParamMaps()))

rfcvModel.params

sample_model_result = rfevaluator.copy()

from pyspark.ml.classification import RandomForestClassifier

train, test = output_sample.randomSplit([0.7, 0.3])

rfClassifier = RandomForestClassifier(labelCol="label", featuresCol="features",maxBins=42,maxDepth=10,numTrees=50)

trainedModel = rfClassifier.fit(train)

cvPredDF = trainedModel.transform(test)
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")