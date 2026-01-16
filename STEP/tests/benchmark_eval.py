import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional

from evaluator import math_equal
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if "\\text{" in text and "}" in text:
        while "\\text{" in text:
            start = text.find("\\text{")
            if start == -1:
                break
            end = text.find("}", start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting"""
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None


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


def load_benchmark(benchmark_path: str) -> List[dict]:
    benchmark = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question")
            answer = record.get("answer")
            if question is None:
                raise ValueError(f"Missing 'question' on line {line_no}")
            benchmark.append({"question": question, "answer": answer})
    return benchmark


def ensure_output_dir(
    output_root: str, benchmark_path: str, model_path: str, num_traces: int, run_label: Optional[str] = None
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    benchmark_name = Path(benchmark_path).stem
    model_name = Path(model_path).name or Path(model_path).stem
    model_label = model_name.replace(" ", "_")
    label = run_label.strip().replace(" ", "_") if run_label else ""
    parts = [benchmark_name, model_label, f"n{num_traces}"]
    if label:
        parts.append(label)
    parts.append(timestamp)
    run_dir = os.path.join(output_root, "_".join(parts))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_single_question(
    llm: LLM,
    tokenizer,
    question: str,
    ground_truth: Optional[str],
    sampling_params: SamplingParams,
    question_index: int,
    output_dir: str,
):
    prompt = prepare_prompt(question, tokenizer)
    bt = time()
    outputs = llm.generate([prompt], sampling_params)
    generation_time = time() - bt

    request_output = outputs[0]
    answers_for_prompt = []
    weights_for_prompt = []
    generation_results = {}
    correct_traces = 0
    total_traces = len(request_output.outputs)
    total_tokens = 0

    for j, output in enumerate(request_output.outputs):
        generated_text = output.text
        extracted_answer = extract_answer(generated_text)
        token_count = len(output.token_ids)
        total_tokens += token_count
        score = getattr(output, "final_score", None)
        is_trace_correct = (
            ground_truth is not None and extracted_answer is not None and equal_func(extracted_answer, ground_truth)
        )
        if is_trace_correct:
            correct_traces += 1
        if extracted_answer is not None and score is not None:
            answers_for_prompt.append(extracted_answer)
            weights_for_prompt.append(score)

        generation_results[f"generation_{j}"] = {
            "token_length": token_count,
            "generated_text": generated_text.strip(),
            "finish_reason": output.finish_reason,
            "stop_reason": output.stop_reason,
            "final_score": score,
            "prompt_index": question_index,
            "request_id": getattr(request_output, "request_id", None),
            "extracted_answer": extracted_answer,
            "is_trace_correct": is_trace_correct,
        }

        print(
            f"question {question_index} output {j} token count: {token_count}, "
            f"finish_reason={output.finish_reason}, stop_reason={output.stop_reason}, final_score={score}, "
            f"trace_correct={is_trace_correct}"
        )

    final_answer = weighted_majority_vote(answers_for_prompt, weights_for_prompt) if answers_for_prompt else None
    is_final_correct = (
        final_answer is not None and ground_truth is not None and equal_func(final_answer, ground_truth)
    )

    question_result = {
        "question_index": question_index,
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "is_final_correct": is_final_correct,
        "num_correct_traces": correct_traces,
        "total_traces": total_traces,
        "generation_time": generation_time,
        "answers_for_prompt": answers_for_prompt,
        "weights_for_prompt": weights_for_prompt,
        "generation_results": generation_results,
    }

    question_file = os.path.join(output_dir, f"question_{question_index:04d}.json")
    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_result, f, ensure_ascii=False, indent=2)

    return {
        "question_index": question_index,
        "question_file": os.path.basename(question_file),
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "is_final_correct": is_final_correct,
        "num_correct_traces": correct_traces,
        "total_traces": total_traces,
        "correct_trace_ratio": correct_traces / total_traces if total_traces else 0.0,
        "generation_time": generation_time,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark inference over a JSONL dataset.")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to JSONL benchmark file with 'question' and 'answer' fields (e.g., /work/nvme/bcjw/bhuang4/hmmt24/hmmt_2024.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory to store per-question outputs and the summary.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional custom label appended to the generated output directory name.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the vLLM model weights.",
    )
    parser.add_argument(
        "--STEP-step-scorer-path",
        required=True,
        help="Path to the step scorer checkpoint.",
    )
    parser.add_argument("--num-traces", type=int, default=64, help="Number of generations per question.")
    parser.add_argument("--max-tokens", type=int, default=60000, help="Maximum new tokens per generation.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to reserve for the LLM (vLLM gpu_memory_utilization).",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    output_dir = ensure_output_dir(
        args.output_dir, args.benchmark, args.model_path, args.num_traces, run_label=args.run_label
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    sampling_params = SamplingParams(
        n=args.num_traces,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        STEP_enable=True,
        STEP_step_scorer_path=args.STEP_step_scorer_path,
        disable_log_stats=False,
    )

    summary_entries = []
    total_start = time()
    for idx, item in enumerate(benchmark):
        question = item["question"]
        ground_truth = item.get("answer")
        print(f"Running question {idx + 1}/{len(benchmark)}")
        summary_entry = run_single_question(
            llm=llm,
            tokenizer=tokenizer,
            question=question,
            ground_truth=ground_truth,
            sampling_params=sampling_params,
            question_index=idx,
            output_dir=output_dir,
        )
        summary_entries.append(summary_entry)
    total_generation_time = time() - total_start

    total_questions = len(summary_entries)
    avg_tokens_per_problem = (
        sum(entry["total_tokens"] for entry in summary_entries) / total_questions if total_questions else 0.0
    )
    final_accuracy = (
        sum(1 for entry in summary_entries if entry["is_final_correct"]) / total_questions if total_questions else 0.0
    )
    summary = {
        "benchmark_path": args.benchmark,
        "model_path": args.model_path,
        "output_dir": output_dir,
        "num_questions": total_questions,
        "num_traces_per_question": args.num_traces,
        "total_generation_time": total_generation_time,
        "generation_time_per_question": total_generation_time / total_questions if total_questions else 0.0,
        "final_accuracy": final_accuracy,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "questions": summary_entries,
    }

    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved per-question outputs to {output_dir}")
    print(f"Total generation time: {total_generation_time}")
    print(f"Generation time per question: {total_generation_time / total_questions if total_questions else 0.0}")
    print(f"Final benchmark accuracy: {final_accuracy}")
    print(f"Avg tokens per problem: {avg_tokens_per_problem}")


if __name__ == "__main__":
    main()
