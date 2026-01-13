import torch
import os
import json
import argparse 
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/work/nvme/bcjw/zliang2/Qwen3-4B-Thinking-2507', help='Model name or path')
parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
parser.add_argument('--end_idx', type=int, default=None, help='End index for processing')
parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
parser.add_argument('--output_path', type=str, default=None, help='Path to save the output hidden states')

args = parser.parse_args()


model_name = args.model_name 
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    
if torch.cuda.is_available():
        model = model.to("cuda")


current_question = None
results = []


with open(args.input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = args.start_idx

end_idx = args.end_idx if args.end_idx is not None else len(lines)
lines = lines[start_idx:end_idx]

for line in tqdm(lines, desc=f"Processing [{start_idx}:{end_idx}]"):
    json_object = json.loads(line)


    prompt = json_object["prompt"]
    response = json_object["forward_text"]
    full_forward_text = prompt + response

    position_list = json_object['extracted_position']

    enc = tok(full_forward_text, return_tensors="pt",add_special_tokens=False)
    print(f"Input length: {enc['input_ids'].size(1)} tokens")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        base_out = model.model(  
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    last_hidden = base_out.last_hidden_state # [1, T, D]
    print(f"Last hidden state shape from base model: {last_hidden.shape}")

    hidden = last_hidden.squeeze(0)  # [T, D]
    selected_hidden = hidden[position_list, :].to(torch.float16).cpu().contiguous()
    
    del hidden, base_out
    torch.cuda.empty_cache() 
    input_ids = enc["input_ids"].squeeze(0).cpu()
    print("Selected hidden state shape:", selected_hidden.shape)

    results.append({
        "select_hidden_states": selected_hidden,      # [L+1, T', D]
        "full_input_ids": input_ids,       # [T]
        "full_forward_text": full_forward_text,
        "original_response": json_object['full_text'],
        "question": json_object['question'],
        'is_correct': json_object['is_correct'],   
    })

save_path = args.output_path + f"_hiddenstates_{start_idx}_{end_idx}.pt" if args.output_path else f"step_hiddenstates_{start_idx}_{end_idx}.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(results, save_path,_use_new_zipfile_serialization=True)
print(f"Saved hidden states to {save_path}")
print("Processing complete.")
