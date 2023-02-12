import pandas as pd
import numpy as np
#importing cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn import metrics
#importing sparkcontext
from pyspark import SparkContext, SparkConf
#importing sqlcontext
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession,DataFrame
from pyspark.ml.feature import StringIndexer
# Importing the string indexer which convert categorial values to continous values

spark = SparkSession.builder.appName("BDM-TASK-1-Group-9").getOrCreate()                                                        #creating spark session and spark variable

df = spark.read.csv('/user/kaggle/kaggle_data/marketing_campaign.csv',sep='\t',header=True, inferSchema=True)

cols = [f"any({col} is null) as {col}" for col in df.columns]                          #checking if any column has null values
df.selectExpr(cols).show()                                                             #printing the columns and if has null values as true else false
df.show()                                                                              # printing the dataset

from sklearn.impute import KNNImputer                                                  #importing KNNImputer

knn = KNNImputer(n_neighbors=10, add_indicator=True)                                   #setting paramaters for the KNNImputer

stringIndex = StringIndexer(inputCols=['Marital_Status','Education','Dt_Customer'], outputCols=['Mar_st','Edu','Dt'])  #using stringerindexer to convert all the string values to double

stringIndex_model = stringIndex.fit(df)                                                #fitting the dataframe for stringIndex_model            

df = stringIndex_model.transform(df).drop('Marital_Status','Education','Dt_Customer')  #dropping the columns having string values after conversion

df = df.withColumn("Mar_st", df["Mar_st"].cast(IntegerType()))                         #Type casting Mar_st to integer

df = df.withColumn("Edu", df["Edu"].cast(IntegerType()))                               #type casting Edu to integer

df = df.withColumn("Dt", df["Dt"].cast(IntegerType()))                                 #type casting Dt to integer

df.show()                                                                              #printing the new dataset

DF = df.toPandas().apply(pd.to_numeric)                                                #tranforming the datatranform from spark to pandas
DF["Income"]  = np.where(DF["Income"].between(0,10000), 'nan', DF["Income"])           #Setting range value for income and checking for missing values
DF["ID"] = np.where(DF["ID"].between(1,1000),'nan',DF["ID"])                           #checking for missing values in ID and setting an out of range value
head = list(DF.columns)                                                                
head.append("KNN_INC")                                                                 #appening the values obtained from KNN for income
head.append("KNN_ID")                                                                  #appending the values obtained from KNN for ID

knn.fit(DF)                                                                            #fitting the data frame 

test = knn.transform(DF)                                                               #tranforming the data frame

np.isnan(test)                                                                         #checking for any null values

test1 = pd.DataFrame(test)                                                          
test1.set_axis(head,axis=1,inplace=True)
print(test1)                                                                           
test1.to_csv('task_1_output.csv', index = False)                                        #printing the ouput to a csv file which would be used as an input for next task