from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODEL_NAME = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model = model.half().cuda()

inp = "What is this"

input_ids = tokenizer.encode(inp, return_tensors="pt").cuda()

with torch.cuda.amp.autocast():
    output = model.generate(input_ids, max_length=256, do_sample=True, early_stopping=True, eos_token_id=model.config.eos_token_id, num_return_sequences=1)

output = output.cpu()

output_text = tokenizer.decode(output[0], skip_special_tokens=False)