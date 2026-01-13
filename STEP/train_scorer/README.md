# Train Scorer Pipeline

This folder contains a 3-step pipeline to build a scorer model from reasoning traces:
1) `data_pre_process.py`: convert raw reasoning traces (JSONL) into model-ready prompts and segment metadata.
2) `data_hidden_state.py`: run the model forward pass and extract hidden states for selected positions.
3) `train_scorer.py`: train an MLP scorer on the extracted hidden states.

## 1) Prepare reasoning traces (JSONL)

Collect reasoning traces from your target model and store them in JSONL format. Each line should be a JSON object that includes at least:
- `question`: the input question.
- `response`: the model output, including a `<think> ... </think>` reasoning block.
- `ground_truth`: the expected answer.
- `extracted_answer`: the parsed final answer (must not be `None`).

Example JSONL line:
```json
{"question":"...","response":"<think>...reasoning...\n\n...</think>\nFinal answer: ...","ground_truth":"42","extracted_answer":"42"}
```

Notes:
- `data_pre_process.py` uses the text before `</think>` as the forward text.
- It splits the prompt + reasoning by double newlines (`\n\n`) to build `extracted_position`.

## 2) Pre-process traces

Run:
```bash
python data_pre_process.py \
  --input_file /path/to/raw_traces.jsonl \
  --output_file /path/to/preprocessed_dir \
  --tokenizer_path /path/to/tokenizer_or_model
```

Outputs a JSONL file (same basename as `input_file`) in `--output_file`. Each item contains:
- `prompt`, `question`, `ground_truth`, `extracted_answer`, `full_text`
- `forward_text` (reasoning portion)
- `segmented_text_list`, `extracted_position`
- `is_correct`

## 3) Extract hidden states

Run:
```bash
python data_hidden_state.py \
  --model_name /path/to/model \
  --input_path /path/to/preprocessed_dir/raw_traces.jsonl \
  --output_path /path/to/hidden_states/output_prefix \
  --start_idx 0 \
  --end_idx 1000
```

This produces a `.pt` file like:
```
output_prefix_hiddenstates_0_1000.pt
```

Each entry contains:
- `select_hidden_states` (selected hidden states at extracted positions)
- `full_input_ids`, `full_forward_text`, `original_response`, `question`, `is_correct`

## 4) Train the scorer

Run:
```bash
python train_scorer.py \
  --train_dir /path/to/hidden_states_dir \
  --step_sample_ratio 1.0 \
  --config_name qwen3_4b_default
```

Outputs:
- checkpoints: `./checkpoints/<config_name>/`
- plots: `./plots/<config_name>/` (loss curve + validation metrics)

## Notes:
- `train_scorer.py` expects `.pt` files produced by `data_hidden_state.py`.
