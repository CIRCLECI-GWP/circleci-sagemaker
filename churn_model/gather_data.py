import boto3
import pandas as pd
import numpy as np
import os
import io

bucket = "circleci-sagemaker"
region_name = "us-east-1"
model_name = os.environ["MODEL_NAME"]


# Set up the session and client we will need for this step
boto_session = boto3.Session(region_name=region_name)
s3_client = boto_session.client(service_name="s3")


# Data retrieval and processing taken from
# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.ipynb
# You would likely replace this part for your own use case, such as querying from Snowflake or Redshift
s3_client.download_file(f"sagemaker-sample-files", "datasets/tabular/synthetic/churn.txt", "churn.txt")
churn = pd.read_csv("./churn.txt")

churn = churn.drop("Phone", axis=1)
churn["Area Code"] = churn["Area Code"].astype(object)
churn = churn.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

model_data = pd.get_dummies(churn)
model_data = pd.concat(
    [model_data["Churn?_True."], model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)], axis=1
)

train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
)


# Upload training and validation data to S3
csv_buffer = io.BytesIO()
train_data.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=f"{model_name}/train/train.csv")

csv_buffer = io.BytesIO()
validation_data.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=f"{model_name}/validation/validation.csv")
