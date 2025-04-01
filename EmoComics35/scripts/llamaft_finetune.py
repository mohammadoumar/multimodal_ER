# ***************** Fine-Tuning LLMs on Comics dataset *********************** #

# ********** Libraries and GPU *************

import os
import ast
import sys
import json
import torch
import pickle
import subprocess

sys.path.append('../')

import pandas as pd

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from sklearn.metrics import classification_report
from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import PeftModel
#from utils.post_processing import post_process

try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")


CURRENT_DIR = Path.cwd()
EC35_DIR = CURRENT_DIR.parent
DATASET_DIR = EC35_DIR / "json_datasets"

MMER_DIR = EC35_DIR.parent
LLAMA_FACTORY_DIR = MMER_DIR / "LLaMA-Factory"

BASE_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
LOGGING_DIR = EC35_DIR / "lfc_log_dir"
OUTPUT_DIR = EC35_DIR / "model_outputs" / f"""EC35_mm_{BASE_MODEL.split("/")[1]}"""
#OUTPUT_DIR = OUTPUT_DIR.as_posix()

#print(CURRENT_DIR, EC35_DIR, DATASET_DIR, LLAMA_FACTORY_DIR, BASE_MODEL, OUTPUT_DIR, sep="\n")


# ****************** DATASET FILES ******************


# # *** TRAIN/TEST DATASET NAME/FILENAME *** #

train_dataset_name = f"""EC35_mm_pg_v2_train.json"""
test_dataset_name = f"""EC35_mm_pg_v2_test.json"""

train_dataset_file = DATASET_DIR / train_dataset_name
test_dataset_file = DATASET_DIR / test_dataset_name


# # # *** TRAIN ARGS FILE PATH *** #

if not os.path.exists(os.path.join(EC35_DIR, "lfc_model_args")):
    os.mkdir(os.path.join(EC35_DIR, "lfc_model_args"))

train_file = EC35_DIR / "lfc_model_args" / f"""{train_dataset_name.split(".")[0].split("train")[0]}{BASE_MODEL.split("/")[1]}.json"""
#print(train_file)

# *** UPDATE dataset_info.json file in LLaMA-Factory *** #


dataset_info_line =  {
  "file_name": f"{train_dataset_file}",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  }
}

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "r") as jsonFile:
    data = json.load(jsonFile)

data["EC35_mm"] = dataset_info_line

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "w") as jsonFile:
    json.dump(data, jsonFile)
    
#     # # ************************** TRAIN MODEL ******************************#

NB_EPOCHS = 5

args = dict(
    
  stage="sft",                           # do supervised fine-tuning
  do_train=True,

  model_name_or_path=BASE_MODEL,         # use bnb-4bit-quantized Llama-3-8B-Instruct model
  num_train_epochs=NB_EPOCHS,            # the epochs of training
  output_dir=str(OUTPUT_DIR),                 # the path to save LoRA adapters
  overwrite_output_dir=True,             # overrides existing output contents

  dataset="EC35_mm",                      # dataset name
  template="mllama",                     # use llama3 prompt template
  #train_on_prompt=True,
  val_size=0.1,
  max_samples=10000,                       # use 500 examples in each dataset

  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  per_device_train_batch_size=2,         # the batch size
  gradient_accumulation_steps=4,         # the gradient accumulation steps
  lr_scheduler_type="cosine",            # use cosine learning rate scheduler
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  #temperature=0.5,
  
  warmup_ratio=0.1,                      # use warmup scheduler    
  learning_rate=5e-5,                    # the learning rate
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  
  fp16=True,                             # use float16 mixed precision training
  quantization_bit=4,                    # use 4-bit QLoRA  
  #use_liger_kernel=True,
  #quantization_device_map="auto",
  
  logging_steps=10,                      # log every 10 steps
  save_steps=5000,                       # save checkpoint every 1000 steps    
  logging_dir=str(LOGGING_DIR),
  
  # use_unsloth=True,
  report_to="none"                       # discards wandb

)

json.dump(args, open(train_file, "w", encoding="utf-8"), indent=2)

p = subprocess.Popen(["llamafactory-cli", "train", train_file], cwd=LLAMA_FACTORY_DIR)
p.wait()


# ********************** INFERENCES ON FINE_TUNED MODEL ******************** #

# LOAD MODEL, ADD LORA ADAPTERS #

args = dict(
    
  model_name_or_path=BASE_MODEL, # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path=str(OUTPUT_DIR),            # load the saved LoRA adapters  
  template="mllama",                     # same to the one in training
  
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  #device_map="auto"
)

model = ChatModel(args)
# if torch.cuda.is_available():
    
#     device = torch.device("cuda")

# model = MllamaForConditionalGeneration.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# # model.load_adapter(str(OUTPUT_DIR))
# # model.eval()
# # print(model.active_adapters())
# # print(model.num_parameters())

# # correct impementation 160-170



# #model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# inference_model = PeftModel.from_pretrained(model, str(OUTPUT_DIR), "lfc adapter")
# inference_model.eval()
# print(inference_model.active_adapters)
# print(inference_model.get_nb_trainable_parameters())
# processor = AutoProcessor.from_pretrained(BASE_MODEL, max_seq_length = 4096)


# # LOAD TEST SET #

# with open(test_dataset_file, "r+") as fh:
#     test_dataset = json.load(fh)

# test_prompts = []
# test_grounds = []

def transform_dict(d):
    if isinstance(d, dict):
        return {key_map.get(k, k): transform_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [transform_dict(i) for i in d]
    elif isinstance(d, str):
        return key_map.get(d, d)
    return d


# for sample in test_dataset:
#     test_prompts.append(sample)

# for prompt in test_prompts:
#     #print(prompt['images'])
#     #print(prompt['messages'])
#     #image=prompt['images']
#     image=Image.open(prompt['images'][0])
#     #['images']
#     key_map = {'from': 'role', 'human': 'user', 'gpt': 'assistant', 'value': 'content'}
#     transformed_data = transform_dict(prompt)
#     print(transformed_data) # type: ignore
#     input_text = processor.apply_chat_template(transformed_data, add_generation_prompt=True) # type: ignore
#     #print(input_text)
#     inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt"
# ).to(model.device)
    
#     #print(inputs.input_ids.shape)
#     #print(inputs.pixel_values.shape)
    
#     output = model.generate(**inputs, max_new_tokens=256)[0]
    
#     # input_length = inputs.input_ids.shape[1]
#     # generated_tokens=output[input_length:]
    
#     # print(processor.decode(generated_tokens))
#     break

#PeftModel.set_adapter(OUTPUT_DIR)

#model.load_adapter(str(OUTPUT_DIR))
#model.set_adapter(str(OUTPUT_DIR))

#adapter_model = PeftModel.from_pretrained(model, OUTPUT_DIR)

#print(adapter_model.get_nb_trainable_parameters)

# Apply LoRA adapters to the model
#$model.eval()

#print(adapter_model.active_adapter)
#print(adapter_model.print_trainable_parameters)
#print(model.get_nb_trainable_parameters)
#print(adapter_model.active_peft_config)
#print(model.active_adapters)
#print(model.get_adapter_state_dict())
#print(model.active_adapters())
#print(model.num_parameters())
#print(model.)

#model = ChatModel(args)#.to(device) # type: ignore

# # LOAD TEST SET #

with open(test_dataset_file, "r+") as fh:
    test_dataset = json.load(fh)

test_prompts = []
test_grounds = []

for sample in test_dataset:
    test_prompts.append(sample)
    #test_prompts.append(sample['messages'])
    test_grounds.append(sample['messages'][1]['value'])


# INFERENCE ON TEST SET #

test_predictions = []

for prompt in tqdm(test_prompts):
    
    #print(prompt)
    
    key_map = {'from': 'role', 'human': 'user', 'gpt': 'assistant', 'value': 'content'}
    transformed_data = transform_dict(prompt['messages'])

    # messages = []
    # messages.append({"role": "user", "content": [
    #             {"type": "image", "image": prompt['images']},
    #             {"type": "text", "text": prompt['messages'][0]['value']}
    #         ]})

    response = ""
    
    for new_text in model.stream_chat(transformed_data, images=prompt['images']): # type: ignore
        #print(new_text, end="", flush=True)
        response += new_text
        #print()
    test_predictions.append(response)

    torch_gc()
    #print(test_predictions)
    #break
    
results_d = {"grounds": test_grounds,
             "predictions": test_predictions}

with open(os.path.join(OUTPUT_DIR, f"""EC35_mm_results_{NB_EPOCHS}.pickle"""), 'wb') as fh:

    pickle.dump(results_d, fh)