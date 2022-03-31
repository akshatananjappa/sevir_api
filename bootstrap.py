import boto3
import os
import sys

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key='os.environ.get("AWS_SECRET_ACCESS_KEY")
)

DOWNLOAD_PATH = os.environ.get('S3_DOWNLOAD_PATH', '/opt/')

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'sevir-data')

S3_FOLDER = os.environ.get('S3_BUCKET_FOLDER', 'neurips-2020-sevir')

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

_ = download_s3_folder(S3_BUCKET_NAME, S3_FOLDER, DOWNLOAD_PATH)

sys.path.append(os.path.join(DOWNLOAD_PATH, S3_FOLDER, 'venv/lib'))
