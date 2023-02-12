from pyspark.sql import SparkSession 
from pyspark.sql import SQLContext                                                     #importing spark session
from pyspark.ml.linalg import Vectors                                                     #importing vector
from pyspark.sql.functions import col                                                     #importing column function from sql
from pyspark.sql import functions as F                                                    #importing function
from pyspark.ml.regression import LinearRegression, RandomForestRegressionModel, RandomForestRegressor #importing the regressions and models required
import pandas as pd                                                                       #importing pandas library
from pyspark.ml.feature import VectorAssembler                                            #importing vector assembler
from pyspark.sql.functions import count, mean, when, lit, create_map, regexp_extract      #importing few necessary sql functions
from pyspark.ml.evaluation import RegressionEvaluator                                     #importing regression evaluator
from pyspark import SparkContext, SparkConf

spark = SparkSession.builder.appName("BDM-TASK-2-Group-9").getOrCreate()                                                        #creating spark session and spark variable


dfd = pd.read_csv('task_1_output.csv')
df = spark.createDataFrame(dfd)  

print('Dataset') 
df.show()                                                                           #printing the dataset being used as input

assembler =VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol='features')  #creating a vector assembler and specifying input columns and merging them as features
output=assembler.transform(df)                                                      #tranforming the data frame for predictions
final_df=output.select('features','Edu')                                            #creating a new data frame and setting final input and output columns
train_data,test_data=final_df.randomSplit([0.8,0.2],seed=80)                        #splitting the data into train and test datasets

lm=LinearRegression(labelCol='Edu')                                                 #applying linear regression function for the column edu
model=lm.fit(train_data)                                                            #fitting the tarining dataset                                                                      #printing the predictions made using linear regression
m=round(model.coefficients[0],2)                                                    #rouding model coefficients
b=round(model.intercept,2)                                                          #rounding model intercepts
print(f"""the formula for linear regression is admit={m}*features+{b}""")           #printing the formula for linear regression

pd.DataFrame({"Coefficients" : model.coefficients},index=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'])  #mentioning the input columns
res=model.evaluate(test_data)                                                       #evaluating the test data
res.residuals.show()                                                                #printing the residuals
unlabelled_data=test_data.select('features')                                        #selecting the input columns which are features
predictions=model.transform(unlabelled_data)                                        #transforming the data
print("MAE:" , res.meanAbsoluteError)                                               #printing out the mean absolute error
print("MSE:", res.meanSquaredError)                                                 #printing out the mean squared error
print("RMSE:",(res.rootMeanSquaredError)//1)                                             #printing out the rootmeansquared error
print("R2:",res.r2)                                                                 #printing out the R2
print("Adj R2:",res.r2adj)                                                          #printing out the adjusted r2

#random forest
df2=df.select(df['Teenhome'],df['Edu'],df['Income'],df['Mar_st'],df['Kidhome'],df['MntWines'],df['MntMeatProducts']) #selecting the required rows for rnadom forest regression

assembler=VectorAssembler(inputCols=['Kidhome','Teenhome','Income','Mar_st','MntWines','MntMeatProducts'],outputCol="features")    #creating a vector assembler and specifying input columns and merging them as features
output2=assembler.transform(df2)                                                    #tranforming the data frame for predictions
output2.select(["features","Edu"])                                                  #selecting the final input and output columsn
model_df=output2.select(["features","Edu"])                                         #creating a new data frame and setting final input and output columns
training_df,testing_df=model_df.randomSplit([0.7,0.3],seed=80)                      #splitting the data into train and test datasets


rf = RandomForestRegressor().setFeaturesCol("features").setLabelCol("Edu");          #applying random forest regression function for the column edu
model = rf.fit(training_df);                                                         #fitting the training data into the model
model = model.transform(testing_df);                                                 #tranforming the testing data

evaluator = RegressionEvaluator(labelCol="Edu", metricName="rmse")                   #using regression evaluator for rsme metric
rmse = (evaluator.evaluate(model))//1                                                     #calculating the rsme
print(rmse)                                                                          #printing out the rsme
