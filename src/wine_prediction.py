import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def data_cleaning(df):
    # cleaning header 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

    

"""main function for application"""
if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .appName('wine_prediction_cccs643') \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    if len(sys.argv) > 3:
        print("Usage: pyspark_wine_training.py <input_file>  <valid_path> <s3_output_bucketname>", file=sys.stderr)
        sys.exit(-1)
    elif len(sys.argv) == 3:
        input_path = sys.argv[1]
        valid_path = sys.argv[2]
        output_path = sys.argv[3] + "testmodel.model"
    else:
        input_path = "s3://wine-data-ar2637-14/train.csv"
        valid_path = "s3://wine-data-ar2637-14/valid.csv"
        output_path="s3://wine-data-ar2637-14/testmodel.model"

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    train_data = data_cleaning(df)

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(valid_path))
    
    valid_data = data_cleaning(df)

    required_features = ['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol',]
    
    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    train_data.cache()
    valid_data.cache()
    
    rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=150,maxBins=8, maxDepth=15,seed=150,impurity='gini')
    
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_data)

    predict = model.transform(valid_data)

 
    final_results = predict.select(['predict', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='predict', metricName='accuracy')

    accuracy = evaluator.evaluate(predict)
    print('Test Accuracy = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    cvmodel = None
    paramGrid = ParamGridBuilder() \
            .addGrid(rf.maxBins, [9, 8, 4])\
            .addGrid(rf.maxDepth, [25, 6 , 9])\
            .addGrid(rf.numTrees, [500, 50, 150])\
            .addGrid(rf.minInstancesPerNode, [6])\
            .addGrid(rf.seed, [100, 200, 5043, 1000])\
            .addGrid(rf.impurity, ["entropy","gini"])\
            .build()
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

  
    cvmodel = crossval.fit(train_data)
    
    model = cvmodel.bestModel
    print(model)
    predict = model.transform(valid_data)
    final_results = predict.select(['predict', 'label'])
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy1 = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    model_path =output_path
    model.write().overwrite().save(model_path)
    sys.exit(0)