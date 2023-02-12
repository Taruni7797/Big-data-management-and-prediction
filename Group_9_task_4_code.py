# importing all the necessary libraries and packages required for the program
from pyspark.sql import SparkSession  
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Imputer
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
import pandas as pd
import numpy as np
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from itertools import chain
from pyspark.sql.functions import count, mean, when, lit, create_map, regexp_extract
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession,DataFrame
from pyspark import SparkContext, SparkConf
from pyspark.sql import*
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName("BDM-TASK-4").getOrCreate()                                                        #creating spark session and spark variable

df = spark.read.csv('/user/kaggle/kaggle_data/marketing_campaign.csv',sep='\t',header=True, inferSchema=True)                                                  #reading the csv file into a dataframe
df.head()

from sklearn.impute import KNNImputer                                                                                   #importing KNNImputer from sci-kit learn

knn = KNNImputer(n_neighbors=10, add_indicator=True)                                                                    #storing into knn object with 10 euclidean distance 10

stringIndex = StringIndexer(inputCols=['Marital_Status','Education','Dt_Customer'], outputCols=['Mar_st','Edu','Dt'])   #Converting string values into double values

#stringIndex_model = stringIndex.fit(df)

pipeline = Pipeline(stages=[stringIndex])                                                                               #sending the stages into pipelines
piped = pipeline.fit(df).transform(df)                                                                                  #Transforming and fitting the dataframe into the pipeline
#piped.drop('Education')
DF  = piped.toPandas()
DF.pop('Education')
DF.pop('Marital_Status')
DF.pop('Dt_Customer')
#print(DF)
DF = DF.apply(pd.to_numeric)                                                                                            #Converting the dataframe to numeric values
#df = stringIndex_model.transform(df).drop('Marital_Status','Education','Dt_Customer')


#df1 = df1.withColumn("Mar_st", df["Mar_st"].cast(IntegerType()))

#df1 = df1.withColumn("Edu", df["Edu"].cast(IntegerType()))

#df1 = df1.withColumn("Dt", df["Dt"].cast(IntegerType()))

#df1.head()

#print(type(df1))

#DF = pd.to_numeric(df1)
DF["Income"]  = np.where(DF["Income"].between(0,10000), 'nan', DF["Income"])                                            #Replacing the cells in income columns which are between 0 and 10000 to nan, as we considered them as outliers
DF["ID"] = np.where(DF["ID"].between(1,1000),'nan',DF["ID"])                                                            #Replacing the cells in ID columns which are between 1 and 1000 with nan as we consider them as outliers
head = list(DF.columns)
head.append("KNN_INC")
head.append("KNN_ID")

knn.fit(DF)                                                                                                             #fitting the model into KNN imputer

test = knn.transform(DF)                                                                                                #Transforming the fitted knn

np.isnan(test)                                                                                                          #Now checking the null values

test1 = pd.DataFrame(test)
test1.set_axis(head,axis=1,inplace=True)
test1[:20]                                                                                                              #printing the output of 1st 20 rows

test1.to_csv('Task_4_output_1.csv', index = False)                                                                      #Writing output into csv file, which will stored on local host

dfd = pd.read_csv('Task_4_output_1.csv')
df = spark.createDataFrame(dfd)                                                                                         #Reading the output from the task1 output file as csv
df.printSchema()

cols = [f"any({col} is null) as {col}" for col in df.columns]
df.selectExpr(cols).show()
df.show()
df.dtypes

df1=df.select(df['Teenhome'],df['Edu'],df['Income'],df['Mar_st'],df['Kidhome'],df['MntWines'],df['MntMeatProducts'])
df.dtypes

assembler = VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol='features')       #performing vector assembler with the required features

train_data,test_data=df.randomSplit([0.8,0.2],seed=80)                                                                  #Distributing training and test data in 80:20 percentages

train_data.show()
test_data.show()
lm=LinearRegression(labelCol='Edu')                                                                                     #Performing linear regression on the education column
pipeline2 = Pipeline(stages=[assembler,lm])                                                                             #Pipelining with stages containing vector assembler and linear regression
piped2 = pipeline2.fit(train_data)                                                                                      #Fitting the training data into pipeline

df2 = piped2.transform(test_data)                                                                                       #Transforming the test_data


df2.show()
evaluator = RegressionEvaluator(labelCol="Edu", metricName="rmse")                                                      #creating a evaluator object
rms = (evaluator.evaluate(df2))//1                                                                                           #evaluating the data frame for rmse of linear regression from the pipeline
print(rms)                                                                                                           #printing the rmse value for linear regression

#random forest
df3= df.select(df['Teenhome'],df['Edu'],df['Income'],df['Mar_st'],df['Kidhome'],df['MntWines'],df['MntMeatProducts'])

assembler=VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol="features")     #performing vector assembler with the required features

training_df,testing_df=df3.randomSplit([0.8,0.2],seed=50)                                                               #Distributing training and test data in 80:20 percentages
training_df.count()
testing_df.count()

pipeline3 = Pipeline(stages=[assembler,lm])                                                                             #Pipelining with stages containing vector assembler and linear regression
piped3 = pipeline3.fit(training_df)                                                                                     #Fitting the training data into pipeline

df3 = piped3.transform(testing_df)                                                                                      #Transforming the test_data

rf_classifier=RandomForestClassifier(labelCol="Edu")                                                                    #Classifying with random forest
df3.show()
evaluator = RegressionEvaluator(labelCol="Edu", metricName="rmse")                                                      #creating a evaluator object
rmse = (evaluator.evaluate(df3))//1                                                                                          #Calculating rmse of random forest
print(rmse)                                                                                                          #printing the rmse value of random forest