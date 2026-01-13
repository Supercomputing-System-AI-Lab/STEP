import json
import argparse
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from tests.evaluator import math_equal

def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text

def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def prepare_prompt(question: str, tokenizer) -> str:
    """Prepare prompt for a single question"""
   
    messages = [
        {'role': "system", "content": "Please reason step by step, and put your final answer within \\boxed{}"},
        {"role": "user", "content": question}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt

def read_jsonl(file_path: str):
    """Read a JSONL file and return a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_on_double_newline(tokenizer, text: str):
    
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    positions, segments = [], []
    current_tokens = []
    all_tokens_decoded = [tokenizer.decode([tok_id]) for tok_id in token_ids]
    
    for idx, decoded in enumerate(all_tokens_decoded):
        current_tokens.append(token_ids[idx])
        
        if "\n\n" in decoded:
            chunk = tokenizer.decode(current_tokens)
            positions.append(idx)
            segments.append(chunk)
            current_tokens = []

    
    if current_tokens:
        segments.append(tokenizer.decode(current_tokens))
    try:
        assert "".join(segments) == text, "Segments do not reconstruct the original text."
    except AssertionError as e:
        print("One of the segments failed to reconstruct the original text.")
        return None, None
    
    return positions, segments


if __name__ == "__main__":
    
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    args = parser.parse_args()


    valid_data = []
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    count = 0
    
    data = read_jsonl(args.input_file)

        
    for d in data:
        stored_data = {}
        question = d['question']
        prompt = prepare_prompt(question, tokenizer)
        
        stored_data['prompt'] = prompt
        stored_data['question'] = question
        stored_data['ground_truth'] = d['ground_truth']
        
        if d['extracted_answer'] is None:
            continue
        
        stored_data['extracted_answer'] = d['extracted_answer']
        stored_data['full_text'] = d['response']
        stored_data['is_correct'] = equal_func(d['extracted_answer'], d['ground_truth'])
        try:
            assert '<think>' in d['response'], "Expected '<think>' in response"
        except AssertionError as e:
            print(f"Skipping item due to missing <think> tag.")
            continue
        # assert '</think>' in d['response'], "Expected '</think>' in response"
        text = d['response'].split('</think>')[0]
        stored_data['forward_text'] = text

        positions, segments = split_on_double_newline(tokenizer, prompt + stored_data['forward_text'])
        
        if positions is None or segments is None:
            print(f"Skipping item due to segmentation error.")
            continue
        
        stored_data['segmented_text_list'] = segments
        stored_data['extracted_position'] = positions
        valid_data.append(stored_data)


    output_basename = os.path.splitext(os.path.basename(args.input_file))[0] + '.jsonl'
    output_path = os.path.join(args.output_file, output_basename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
