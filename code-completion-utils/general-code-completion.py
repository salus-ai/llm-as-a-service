from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTNeoXForCausalLM
from pydantic import BaseModel
import threading
# from transformers import TrainingArguments, Trainer, TextDataset, DataCollatorForLanguageModeling
import boto3
import os
from datetime import datetime
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device_general = 'cuda' if torch.cuda.is_available() else 'cpu'
# Print the device that will be used
if device_general == 'cuda':
    print("Running on GPU")
else:
    print("Running on CPU")


def download_directory_from_s3(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        # Ensure the key isn't the name of the directory itself
        for key in [obj['Key'] for obj in result['Contents'] if not obj['Key'].endswith('/')]:
            # only download files that start with the prefix (directory)
            if key.startswith(s3_folder):
                # Remove the directory name and slash from the key name & create local file path
                local_file_path = os.path.join(local_dir, os.path.relpath(key, s3_folder))
                # Ensure local directory exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                # Download the file from S3
                s3.download_file(bucket_name, key, local_file_path)
                print(f"Downloaded file: {local_file_path}")

bucket_name = 'paradigm-llm-models'
s3_folder = '<MODELNAME>/<MODELVERSION>/'
local_dir = 'model_dir'

download_directory_from_s3(bucket_name, s3_folder, local_dir)
print("Downloaded model to local directory")

description = """
Salus LLM-as-a-Service 🚀

A Scalable API for any open-source or proprietary LLM
"""

app = FastAPI(
    title="Salus",
    description=description,
    version="0.2.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TextGenerationInput(BaseModel):
    text: str

model = None
tokenizer = None
text_generator = None

def load_model():
    global model, tokenizer, text_generator
    model = AutoModelForCausalLM.from_pretrained(local_dir, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set device for pipeline
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("Pipeline is set to run on GPU")
    else:
        print("Pipeline is set to run on CPU")

    # Create the text generation pipeline using the local model and tokenizer
    # text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    # pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    # )


@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=load_model, args=())
    thread.start()

@app.get("/")
def hello_world():
    return {"message": "Paradigm API for LLM"}

@app.post("/generate/")
async def generate_text(input_data: TextGenerationInput):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # if len(input_data.text) > 1000:  # Adjust this limit as needed
    #     raise HTTPException(status_code=400, detail="Input text is too long")

    try:
        # modified_input = f"<|prompter|>{input_data.text}<|endoftext|><|assistant|>"
        # # generated_text = text_generator(modified_input, max_length=100)
        # generated_text = text_generator(
        #         modified_input,
        #         max_length=200,
        #         do_sample=True,
        #         top_k=10,
        #         num_return_sequences=1,
        #         eos_token_id=tokenizer.eos_token_id,
        #     )
        if 'OpenAssistant' in '<MODELNAME>':
            modified_input = f"<|prompter|>{input_data.text}<|endoftext|><|assistant|>"
        else:
            modified_input = input_data.text

        input_ids = tokenizer.encode(modified_input, return_tensors="pt").to(device_general)

        if device_general == 'cuda':
            with torch.cuda.amp.autocast():
                output = model.generate(input_ids, max_length=100, do_sample=True, early_stopping=True, eos_token_id=model.config.eos_token_id, num_return_sequences=1)
        else:
            output = model.generate(input_ids, max_length=100, do_sample=True, early_stopping=True, eos_token_id=model.config.eos_token_id, num_return_sequences=1)

        output = output.to('cpu')

        output_text = tokenizer.decode(output[0], skip_special_tokens=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # return {"generated_text": generated_text[0]['generated_text'].split("<|assistant|>")[1]}
    # return {"generated_text": generated_text[0]['generated_text']}
    return {"generated_text": output_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
