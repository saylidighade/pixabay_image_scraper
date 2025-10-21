import boto3
import os
from botocore.exceptions import NoCredentialsError

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


local_folder = 'instagram/'
bucket_name = 'content-marketing-pixabay-images'  # just the bucket name

for root, dirs, files in os.walk(local_folder):
    for file in files:
        local_file = os.path.join(root, file)
        s3_file = os.path.join('data/instagram', os.path.relpath(local_file, local_folder))
        upload_to_aws(local_file, bucket_name, s3_file)
