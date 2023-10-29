# diabetes-prediction
This is my experimental repository calculating the chance of developing diabetes based on a variety of factors, developed and researched along with Kokand University.

# To begin, we need to install the necessary dependencies and set up a Spark session. We can do this by running the following code:

 Install PySpark
!pip install pyspark

 Create a Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("spark").getOrCreate()

# Next, we'll clone the diabetes dataset from a GitHub repository and take a look at its contents:

 Clone the diabetes dataset
!git clone https://github.com/education454/diabetes_dataset

 Check the dataset files
!ls diabetes_dataset

 Create a Spark DataFrame from the dataset
df = spark.read.csv("/content/diabetes_dataset/diabetes.csv", header=True, inferSchema=True)

 Display the DataFrame
df.show()

# We'll now perform some essential data cleaning and preparation steps. First, let's check for null values in the dataset:

 Check for null values
for col in df.columns:
    print(col + ":", df[df[col].isNull()].count())

# Next, we need to address any unnecessary values in specific columns. We'll identify and replace them with the mean value:

 Look for unnecessary values
def count_zeros():
    columns_list = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for i in columns_list:
        print(i + ":", df[df[i] == 0].count())

 Calculate and replace unnecessary values with the mean
from pyspark.sql.functions import *

for i in df.columns[1:6]:
    data = df.agg({i: "mean"}).first()[0]
    print("Mean value for {} is {}".format(i, int(data)))
    df = df.withColumn(i, when(df[i] == 0, int(data)).otherwise(df[i]))

 Display the updated DataFrame
df.show()

# Now, let's analyze the correlation between input and output variables and perform feature selection:

 Find the correlation with the outcome
for i in df.columns:
    print("Correlation to outcome for {} is {}".format(i, df.stat.corr("Outcome", i)))

 Feature selection
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], outputCol='features')
output_data = assembler.transform(df)

# In this section, we'll split the dataset into training and testing sets and build a logistic regression model:

 Create the final data
final_data = output_data.select('features', 'Outcome')

 Split the dataset and build the model
train, test = final_data.randomSplit([0.7, 0.3])
models = LogisticRegression(labelCol='Outcome')
model = models.fit(train)

 Model summary
summary = model.summary
summary.predictions.describe().show()

# Now, we'll evaluate the model's performance and save it:

 Evaluate the model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = model.evaluate(test)
predictions.predictions.show(20)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='Outcome')
model_evaluation = evaluator.evaluate(model.transform(test))
print("Model evaluation score:", model_evaluation)

 Save the model
model.save("diabetes_model")

 Load the saved model back into the environment
from pyspark.ml.classification import LogisticRegressionModel

loaded_model = LogisticRegressionModel.load('diabetes_model')

# In the final part, we will use the saved model to make predictions on new data:

 Create a new Spark DataFrame for testing
test_df = spark.read.csv('/content/diabetes_dataset/new_test.csv', header=True, inferSchema=True)

 Create an additional feature column
test_data = assembler.transform(test_df)

 Use the model to make predictions
results = loaded_model.transform(test_data)
results.select('features', 'prediction').show()
