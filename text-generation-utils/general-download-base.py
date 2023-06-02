import os
import boto3
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM

# Get the path of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the model directory within the script directory
model_dir = os.path.join(script_dir, 'cache')
model_dir_save = os.path.join(script_dir, 'base')
os.makedirs(model_dir, exist_ok=True)

# Set the TRANSFORMERS_CACHE environment variable to the model directory
os.environ['TRANSFORMERS_CACHE'] = model_dir

print(f"model dir - {model_dir}")
# Load and save the model and tokenizer

model = AutoModelForCausalLM.from_pretrained(
  "<MODELNAME>"
)

tokenizer = AutoTokenizer.from_pretrained(
  "<MODELNAME>"
)

tokenizer.save_pretrained(model_dir_save)
model.save_pretrained(model_dir_save)

# Save to S3
s3 = boto3.resource('s3')

# Replace 'mybucket' with your S3 bucket name
bucket_name = 'paradigm-llm-models'

# Specify the directory in the bucket where you want to save your model
bucket_directory = '<MODELNAME>/'

# Upload tokenizer and model
for root, dirs, files in os.walk(model_dir_save):
    for file in files:
        local_path = os.path.join(root, file)
        s3_path = os.path.join(bucket_directory, os.path.relpath(local_path, script_dir))

        s3.meta.client.upload_file(local_path, bucket_name, s3_path)
