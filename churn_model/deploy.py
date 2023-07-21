import sagemaker
import boto3
import os
from time import strftime, gmtime, sleep


# Environment variables
# See this link for more details: https://circleci.com/docs/set-environment-variable/
bucket = "circleci-sagemaker"
region_name = "us-east-1"
model_name = os.environ["MODEL_NAME"]
model_description = os.environ["MODEL_DESC"]
role_arn = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
endpoint_instance_type = "ml.t2.medium"
endpoint_instance_count = 1
current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())


# Set up the sessions and clients we will need for this step
boto_session = boto3.Session(region_name=region_name)
sagemaker_client = boto_session.client(service_name="sagemaker")
sagemaker_runtime_client = boto_session.client(service_name="sagemaker-runtime")
sagemaker_session = sagemaker.Session(
    boto_session = boto_session,
    sagemaker_client = sagemaker_client,
    sagemaker_runtime_client = sagemaker_runtime_client,
    default_bucket = bucket
)


# Get the latest approved model package of the model group in question
model_package_arn = sagemaker_client.list_model_packages(
    ModelPackageGroupName = model_name,
    ModelApprovalStatus = "Approved",
    SortBy = "CreationTime",
    SortOrder = "Descending"
)['ModelPackageSummaryList'][0]['ModelPackageArn']

# Get a list of existing models with model_name
models_list = sagemaker_client.list_models(
    NameContains=model_name
)['Models']

# Create the model
timed_model_name = f"{model_name}-{current_time}"
container_list = [{"ModelPackageName": model_package_arn}]

create_model_response = sagemaker_client.create_model(
    ModelName = timed_model_name,
    ExecutionRoleArn = role_arn,
    Containers = container_list
)
print(f"Created model ARN: {create_model_response['ModelArn']}")


# Get a list of existing endpoint configs with model_name
endpoint_configs_list = sagemaker_client.list_endpoint_configs(
    NameContains=model_name
)['EndpointConfigs']

# Create endpoint config
create_endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName = timed_model_name,
    ProductionVariants = [
        {
            "InstanceType": endpoint_instance_type,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": endpoint_instance_count,
            "ModelName": timed_model_name,
            "VariantName": "AllTraffic",
        }
    ]
)
print(f"Created endpoint config ARN: {create_endpoint_config_response['EndpointConfigArn']}")


# Get a list of existing endpoints with model_name
endpoints_list = sagemaker_client.list_endpoints(
    NameContains=model_name
)['Endpoints']

# Create or update the endpoint
if endpoints_list:
    create_update_endpoint_response = sagemaker_client.update_endpoint(
        EndpointName = model_name,
        EndpointConfigName = timed_model_name
    )
else:
    create_update_endpoint_response = sagemaker_client.create_endpoint(
        EndpointName = model_name,
        EndpointConfigName = timed_model_name
    )

# Wait for endpoint ot be InService status
describe_endpoint_response = sagemaker_client.describe_endpoint(EndpointName=model_name)
while describe_endpoint_response['EndpointStatus'] != "InService":
    print(describe_endpoint_response['EndpointStatus'])
    sleep(60)
    describe_endpoint_response = sagemaker_client.describe_endpoint(EndpointName=model_name)

endpoint_arn = create_update_endpoint_response['EndpointArn']
print(f"Created endpoint ARN: {endpoint_arn}")


# Cleanup
# If model already existed, delete old versions
if models_list:
    for model in models_list:
        delete_model_name = model['ModelName']
        sagemaker_client.delete_model(ModelName=delete_model_name)
        print(f"Model {delete_model_name} deleted.")

# If endpoint config already existed, delete old versions
if endpoint_configs_list:
    for endpoint_config in endpoint_configs_list:
        delete_endpoint_config_name = endpoint_config['EndpointConfigName']
        sagemaker_client.delete_endpoint_config(EndpointConfigName=delete_endpoint_config_name)
        print(f"Endpoint config {delete_endpoint_config_name} deleted.")
