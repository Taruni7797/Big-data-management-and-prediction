import sys
from pyspark import SparkContext, SparkConf                                                                            #importing sparkContext
from pyspark.sql.types import IntegerType                                                                              #importing IntegerType for type casting
from pyspark.ml.feature import StandardScaler                                                                          #importing StandardScaler 
from pyspark.ml.feature import PCA                                                                                     #importing PCA
from pyspark.sql.session import SparkSession                                                                           #importing SparkSession  
from pyspark.ml.feature import StringIndexer                                                                           #importing stringIndexer
from pyspark.ml.feature import OneHotEncoder                                                                           #importing OneHotEncoder                                 
from pyspark.ml.linalg import Vectors                                                                                  #importing Vectors
from pyspark.ml.feature import VectorAssembler                                                                         #importing VectorAssembler 

if __name__ == "__main__":
    conf = SparkConf().setAppName("BDM")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.appName("BDMPyspark").getOrCreate()                                                    #Creating a SparkSession
    df = spark.read.csv('/user/kaggle/kaggle_data/marketing_campaign.csv', sep='\t', inferSchema=True, header=True)     #Reading file from marketing_campaign.csv into a dataframe  
    df.show()                                                                                                           #Printing the dataframe 
    df2 = df.select("ID","Education","Marital_Status","Kidhome",'Teenhome')
    df2.show()                                                                                                          #Printing the dataframe for selected columns
    Education_indexer = StringIndexer(inputCol="Education", outputCol="EducationIndex")                                 #StringIndexer encodes a string Education to a column of EducationIndex
    df3 = Education_indexer.fit(df2).transform(df2)                                                                     #tranforming the data frame for Education_indexer
    Marital_indexer = StringIndexer(inputCol="Marital_Status", outputCol="MaritalIndex")                                #StringIndexer encodes a string Marital_Status to a column of MaritalIndex    
    df3 = Marital_indexer.fit(df3).transform(df3)                                                                       #tranforming the data frame for Marital_indexer
    df3.show()                                                                                                          #Printing the dataframe with indexers
    onehotencoder_Education_vector = OneHotEncoder(inputCol="EducationIndex", outputCol="Education_vec")                #One Hot Encoding for converting EducationIndex into a binary vector.
    df4 = onehotencoder_Education_vector.fit(df3).transform(df3)                                                        #tranforming the data frame for Education_vector
    onehotencoder_Marital_vector = OneHotEncoder(inputCol="MaritalIndex", outputCol="Marital_vec")                      #One Hot Encoding for converting MaritalIndex into a binary vector.
    df4 = onehotencoder_Marital_vector.fit(df4).transform(df4)                                                          #tranforming the data frame for Marital_vector
    df4.show()
    assembler = VectorAssembler(inputCols=['ID','Education_vec','Marital_vec','Kidhome','Teenhome'], outputCol="variable")  #creating a vector assembler and specifying input columns and merging them as variable
    feature_vectors = assembler.setHandleInvalid("keep").transform(df4)                                                   #tranforming the data frame for vectors
    feature_vectors.select('variable').show()                                                                             #printing feature vector with select variable
    scaler = StandardScaler(inputCol="variable", outputCol="Standard variable", withStd=True, withMean=True)              #standardization of variable for better results
    scalerModel = scaler.fit(feature_vectors)                                                                             #fitting the scalar for feature_vectors
    std_feature_vectors = scalerModel.transform(feature_vectors)                                                          #tranforming scalerModel to standard feature vector
    print("==========Standardized data===========") 
    std_feature_vectors.select('Standard variable').show()                                                                #Display only standardized variables
    pca = PCA(k=3, inputCol="Standard variable", outputCol="Main component score")                                        #calling the PCA API and input is a standardized variate and the output is a principal component score with up to the third principal component
    pcaModel = pca.fit(std_feature_vectors)                                                                               #fitting standard feature vector for pca model
    print("===========Eigenvector================")
    print(pcaModel.pc)                                                                                                    #printing the pc
    print("===========Contribution rate==========")
    print(pcaModel.explainedVariance)                                                                                     #printing the pca model explainedVariance
    pca_score = pcaModel.transform(std_feature_vectors).select("Main component score")                                    #tranforming standard feature vector by selecting Main component score into pca score
    print("==========Main component score========")
    pca_score.show()                                                                                                      #Printing the Main component score
    
