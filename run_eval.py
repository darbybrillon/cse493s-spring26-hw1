import os
import re
from collections import Counter
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B"
DATASET_NAME = "OpenRLHF/aime-2024"
BATCH_SIZE = 8
ANSWER_TOKENS = 2048

# SEQ_BUDGETS = [1024, 2048, 4000, 8000, 16000, 32000]
PARALLEL_THINK_BUDGET = 4000
# PARALLEL_N_VALUES = [1, 2, 4, 8, 16, 32]

SEQ_BUDGETS = [1024, 4000, 16000, 32000]
PARALLEL_N_VALUES = [1, 2, 4, 8]

hf_token = os.environ.get("HF_TOKEN")

def extract_thinking_and_answer(text):
    """Split raw output into (thinking_text, answer_text). Used for token counting."""
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1 and end != -1:
        think = text[start + len("<think>"): end]
        answer = text[end + len("</think>"):]
    elif start != -1:
        think = text[start + len("<think>"):]
        answer = ""
    else:
        think = ""
        answer = text
    return think.strip(), answer.strip()

def count_tokens(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def majority_vote(preds):
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
dataset = load_dataset(DATASET_NAME, split="train", token=hf_token)
llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)

def build_prompt(ex, enable_thinking):
    return tokenizer.apply_chat_template(
        ex["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

def run_sequential_stop(budget):
    print(f"\nSequential STOP | budget={budget}")
    params_think = SamplingParams(
        max_tokens=budget, temperature=0.0,
        stop=["</think>"], include_stop_str_in_output=False,
    )
    params_answer = SamplingParams(max_tokens=ANSWER_TOKENS, temperature=0.0)

    correct_exact, correct_flex, total, total_think_tokens = 0, 0, 0, 0
    per_problem_tokens = []

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        labels = [int(ex["label"]) for ex in batch]
        prompts = [build_prompt(ex, enable_thinking=True) for ex in batch]

        think_outputs  = llm.generate(prompts, params_think)
        answer_prompts = []
        batch_think_counts = []

        for j, out in enumerate(think_outputs):
            think_toks = len(out.outputs[0].token_ids)
            total_think_tokens += think_toks
            batch_think_counts.append(think_toks)
            answer_prompts.append(prompts[j] + out.outputs[0].text + "</think>\n")

        answer_outputs = llm.generate(answer_prompts, params_answer)
        for j, ans_out in enumerate(answer_outputs):
            answer_toks = len(ans_out.outputs[0].token_ids)
            per_problem_tokens.append(batch_think_counts[j] + answer_toks) 
            full_text = answer_prompts[j] + ans_out.outputs[0].text
            gold = labels[j]
            if extract_answer(full_text, "exact_match")      == gold: correct_exact += 1
            if extract_answer(full_text, "flexible_extract") == gold: correct_flex  += 1
            total += 1

    acc_exact = correct_exact / total
    acc_flex = correct_flex  / total
    return acc_exact, acc_flex, total_think_tokens, per_problem_tokens

def run_parallel(n_samples):
    print(f"\nParallel | n={n_samples}")
    params = SamplingParams(
        max_tokens=PARALLEL_THINK_BUDGET + ANSWER_TOKENS,
        temperature=0.6, top_p=0.95, top_k=50, n=n_samples,
    )

    correct_majority, correct_best, total, total_think_tokens = 0, 0, 0, 0
    per_problem_tokens = []

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        labels = [int(ex["label"]) for ex in batch]
        prompts = [build_prompt(ex, enable_thinking=True) for ex in batch]
        outputs = llm.generate(prompts, params)

        for j, out in enumerate(outputs):
            gold  = labels[j]
            preds_exact, preds_flex = [], []
            problem_total = 0

            for sample in out.outputs:
                think_text, ans_text = extract_thinking_and_answer(sample.text)
                think_toks = count_tokens(think_text, tokenizer)
                ans_toks = count_tokens(ans_text, tokenizer)
                total_think_tokens += think_toks
                problem_total += think_toks + ans_toks
                preds_exact.append(extract_answer(sample.text, "exact_match"))
                preds_flex.append(extract_answer(sample.text, "flexible_extract"))

            per_problem_tokens.append(problem_total)
            if majority_vote(preds_exact) == gold:   correct_majority += 1
            if any(p == gold for p in preds_exact):  correct_best     += 1
            total += 1

    acc_majority = correct_majority / total
    acc_best = correct_best / total
    return acc_majority, acc_best, total_think_tokens, per_problem_tokens


if __name__ == "__main__":
    seq_results = []
    for budget in SEQ_BUDGETS:
        acc_e, acc_f, tt, ppt = run_sequential_stop(budget)
        seq_results.append((budget, acc_e, acc_f, tt, ppt))

    par_results = []
    for n in PARALLEL_N_VALUES:
        acc_maj, acc_best, tt, ppt = run_parallel(n)
        par_results.append((n, acc_maj, acc_best, tt, ppt))
