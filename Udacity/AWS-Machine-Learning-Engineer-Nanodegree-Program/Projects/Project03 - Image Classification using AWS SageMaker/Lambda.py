# First of three function,  serialize ImageData is responsible for serialize target data to S3.

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    file_name = '/tmp/image.png'
    s3.download_file(bucket, key, file_name)

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



# Second function classificationPart is responsible for the classification part

import json
import base64
import boto3

ENDPOINT = 'image-classification-2022-04-03-15-19-28-098'

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    runtime= boto3.client('runtime.sagemaker')

    # Instantiate a Predictor
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')]

    # We return the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences'],
        }
    }

# The last one filterLowConfidenceInferences is responsible of filtering low-confidence inferences.

import json


THRESHOLD = .90

class Threshold_Error(Exception):
    pass


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = json.loads(event['inferences']) if type(event['inferences']) == str else event['inferences']

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) > THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Threshold_Error ("THRESHOLD_CONFIDENCE_ARE_NOT_MET_YET")

    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences'],
        }
    }