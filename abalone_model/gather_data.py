import boto3
import pandas as pd
import numpy as np
import os
import io
from zipfile import ZipFile

bucket = "circleci-sagemaker"
region_name = "us-east-1"
model_name = os.environ["MODEL_NAME"]


# Set up the session and client we will need for this step
boto_session = boto3.Session(region_name=region_name)
s3_client = boto_session.client(service_name="s3")


# Data retrieval and processing taken from
# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone.ipynb
# You would likely replace this part for your own use case, such as querying from Snowflake or Redshift

# S3 bucket where the training data is located
data_bucket = f"sagemaker-sample-files"
data_prefix = "datasets/tabular/uci_abalone"

for data_category in ["train", "validation"]:
    data_key = "{0}/{1}/abalone.{1}".format(data_prefix, data_category)
    output_key = "{0}/{1}/{1}.libsvm".format(model_name, data_category)
    data_filename = "abalone.{}".format(data_category)
    s3_client.download_file(data_bucket, data_key, data_filename)
    s3_client.upload_file(data_filename, bucket, output_key)
