from pyspark.sql import SparkSession                                                #importing spark session
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors                                               #importing vector
from pyspark.sql.functions import col                                               #importing column function from sql
from pyspark.sql import functions as F                                              #importing function
from pyspark.ml.regression import LinearRegression, RandomForestRegressionModel, RandomForestRegressor #importing the regressions and models required
import pandas as pd                                                                 #importing pandas library
from pyspark.ml.feature import VectorAssembler                                       #importing vector assembler
from pyspark.sql.functions import count, mean, when, lit, create_map, regexp_extract #importing few necessary sql functions
from pyspark import SparkContext, SparkConf

spark = SparkSession.builder.appName("BDM-TASK-2-Group-9").getOrCreate()                                                        #creating spark session and spark variable


dfd = pd.read_csv('task_1_output.csv')
df = spark.createDataFrame(dfd)  
df.printSchema()                                                                    #printing the schema

print('Dataset') 
df.show()                                                                           #printing the dataset being used as input

cols = [f"any({col} is null) as {col}" for col in df.columns]                       #checking if the any of the columns have null values
print('Checking for columsn with null values')
df.selectExpr(cols).show()                                                          #printing the column values

assembler =VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol='features')  #creating a vector assembler and specifying input columns and merging them as features
output=assembler.transform(df)                                                      #tranforming the data frame for predictions
final_df=output.select('features','Edu')                                            #creating a new data frame and setting final input and output columns
train_data,test_data=final_df.randomSplit([0.8,0.2],seed=80)                        #splitting the data into train and test datasets

print('Train dataset for linear regression')
train_data.show()                                                                   #printing the train dataset

print('Test dataset for linear regression')
test_data.show()                                                                    #printing the test dataset

lm=LinearRegression(labelCol='Edu')                                                 #applying linear regression function for the column edu
model=lm.fit(train_data)                                                            #fitting the tarining dataset
model = model.transform(test_data)                                                  #transforming the testing dataset

print('Prediction data for linear regression')
model.show()                                                                        #printing the predictions made using linear regression

#random forest
df2=df.select(df['Teenhome'],df['Edu'],df['Income'],df['Mar_st'],df['Kidhome'],df['MntWines'],df['MntMeatProducts']) #selecting the required rows for rnadom forest regression

assembler=VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol="features")    #creating a vector assembler and specifying input columns and merging them as features
output2=assembler.transform(df2)                                                    #tranforming the data frame for predictions
output2.select(["features","Edu"])                                                  #selecting the final input and output columsn
model_df=output2.select(["features","Edu"])                                         #creating a new data frame and setting final input and output columns
training_df,testing_df=model_df.randomSplit([0.7,0.3],seed=80)                      #splitting the data into train and test datasets

print('Train dataset for Random Forest Regression')
training_df.show()                                                                  # printing the train dataset

print('Test dataset for Random Forest regression') 
testing_df.show()                                                                    #printing the test dataset

rf = RandomForestRegressor().setFeaturesCol("features").setLabelCol("Edu");          #applying random forest regression function for the column edu
model = rf.fit(training_df);                                                         #fitting the training data into the model
model = model.transform(testing_df);                                                 #tranforming the testing data
print('Prediction data for Random Forest regression')               
model.show()                                                                         #printing out the final random ofrest generated predictions

