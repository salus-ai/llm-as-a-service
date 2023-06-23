from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTNeoXForCausalLM
from pydantic import BaseModel
import threading
from transformers import TrainingArguments, Trainer, TextDataset, DataCollatorForLanguageModeling
import boto3
import os
from datetime import datetime
import logging
import torch
from typing import List
from datasets import concatenate_datasets, load_dataset

from typing import List, Dict
from PyPDF2 import PdfReader

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

class FineTuneInput(BaseModel):
    dataset_path: str
    num_train_epochs: int = 1  # default is 1
    per_device_train_batch_size: int = 1  # default is 1

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

def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PdfReader(pdf_file_obj)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def fine_tune_and_upload(dataset_dir: str, num_train_epochs: int, per_device_train_batch_size: int):
    logger.info(f"Starting fine-tuning with dataset {dataset_dir}")

    # Parse S3 URI
    assert dataset_dir.startswith("s3://")
    bucket_name, key = dataset_dir[5:].split("/", 1)

    # Initialize S3 resource
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # List all files in the directory
    files: List[str] = []
    for obj in bucket.objects.filter(Prefix=key):
        local_file_path = f"local_dataset_{len(files)}{os.path.splitext(obj.key)[-1]}"
        bucket.download_file(obj.key, local_file_path)
        files.append(local_file_path)

    if not files:
        raise Exception("No files found in the dataset directory")

    # Load dataset
    dataset = load_dataset('text', data_files=files)

    # Tokenize dataset
    def tokenize_function(examples):
    # Only tokenize non-empty and non-whitespace lines
        return tokenizer([text for text in examples['text'] if text.strip() != ''], truncation=True, padding=False)

    # Apply tokenization function to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

    # Make sure we're shuffling only the 'train' split
    tokenized_dataset = tokenized_dataset['train'].shuffle(seed=42)


    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # Train model
    trainer.train()
    logger.info("Finished training, saving model")

    # Save model
    trainer.save_model("./results")
    tokenizer.save_pretrained("./results")
    logger.info("Model saved, uploading to S3")

    # Upload to S3
    s3 = boto3.client('s3')
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for file in os.listdir("./results"):
        s3.upload_file(f"./results/{file}", "paradigm-llm-models", f"<MODELNAME>/finetuned_model_revision_{current_datetime}/{file}")

    logger.info("Model uploaded to S3")



@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=load_model, args=())
    thread.start()

@app.get("/")
def hello_world():
    return {"message": "Paradigm API for LLM"}

@app.post("/generate/")
async def generate_text(input_data: TextGenerationInput):
    # if model is None or tokenizer is None or text_generator is None:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # if len(input_data.text) > 1000:  # Adjust this limit as needed
    #     raise HTTPException(status_code=400, detail="Input text is too long")

    try:
        # generated_text = text_generator(input_data.text, max_length=100)
        # if 'OpenAssistant' in '<MODELNAME>':
        #     modified_input = f"<|prompter|>{input_data.text}<|endoftext|><|assistant|>"
        # else:
        #     modified_input = input_data.text

        modified_input = input_data.text

        input_ids = tokenizer.encode(modified_input, return_tensors="pt").to(device_general)

        if device_general == 'cuda':
            with torch.cuda.amp.autocast():
                output = model.generate(input_ids, max_length=100, do_sample=True, early_stopping=True, eos_token_id=model.config.eos_token_id, num_return_sequences=1)
        else:
            output = model.generate(input_ids, max_length=100, do_sample=True, early_stopping=True, eos_token_id=model.config.eos_token_id, num_return_sequences=1)

        output = output.to('cpu')

        output_text = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"Output successfully generated - {output_text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # return {"generated_text": generated_text[0]['generated_text']}
    return {"generated_text": output_text}

@app.post("/fine_tune/")
async def fine_tune_model(background_tasks: BackgroundTasks, input_data: FineTuneInput):
    background_tasks.add_task(fine_tune_and_upload, input_data.dataset_path, input_data.num_train_epochs, input_data.per_device_train_batch_size)
    return {"message": "Fine-tuning started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
