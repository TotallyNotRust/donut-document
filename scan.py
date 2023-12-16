from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import json
import torch
from tqdm.auto import tqdm
import numpy as np

from PIL import Image

from donut import JSONParseEvaluator

from datasets import load_dataset

import argparse

# find what accelerator to use, can be either cpu, gpu or cuda
# by default use cpu as that will work on all systems
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", help="Set file", default="invoice.png", required=False)

args = parser.parse_args()

image = Image.open(args.file).convert("RGB")

processor = DonutProcessor.from_pretrained("TotallyNotRust/donut")
model = VisionEncoderDecoderModel.from_pretrained("TotallyNotRust/donut")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()
model.to(device)

output_list = []
accs = []

dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
decoder_input_ids = decoder_input_ids.to(device)

outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

seq = processor.batch_decode(outputs.sequences)[0]
seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
seq = processor.token2json(seq)

print(seq)