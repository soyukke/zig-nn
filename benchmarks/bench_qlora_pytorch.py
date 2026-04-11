#!/usr/bin/env python3
"""
PyTorch QLoRA benchmark for Gemma 3 1B
Conditions matched to Zig implementation:
- Model: Gemma 3 1B (loaded from local GGUF, dequantized to float16)
- LoRA: rank=8, Q/V only
- Data: 4 sentences, seq_len=32
- Optimizer: Adam, lr=3e-5, weight_decay=0.01
- Steps: 100
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# --- Config ---
GGUF_DIR = os.path.join(os.path.dirname(__file__), "..", "models") + "/"
GGUF_FILE = "gemma3-1b.gguf"
SEQ_LEN = 32
NUM_STEPS = 100
LR = 3e-5
WEIGHT_DECAY = 0.01
LORA_RANK = 8
LORA_ALPHA = 16

# --- Training data (same as Zig) ---
TRAIN_TEXTS = [
    "The capital of Japan is Tokyo, which is one of the largest cities in the world with over thirteen million people.",
    "The capital of France is Paris, which is known for the Eiffel Tower and its beautiful art museums and cafes.",
    "The capital of Japan is Tokyo, a modern city that blends traditional culture with cutting edge technology.",
    "The capital of France is Paris, the city of light that attracts millions of tourists from around the world.",
]


def main():
    print("=== PyTorch QLoRA Benchmark: Gemma 3 1B ===\n")

    # --- Load model from local GGUF ---
    print("Loading model from GGUF...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = "mps"
        print(f"  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"  Device: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = "cpu"
        print(f"  Device: CPU")

    # Load from local GGUF (dequantized to float16)
    model = AutoModelForCausalLM.from_pretrained(
        GGUF_DIR,
        gguf_file=GGUF_FILE,
        dtype=torch.float16,
    )
    model = model.to(device)

    # Load tokenizer from the GGUF directory (sentencepiece should be embedded)
    try:
        tokenizer = AutoTokenizer.from_pretrained(GGUF_DIR)
    except Exception:
        # If tokenizer not in GGUF dir, try to build from GGUF metadata
        print("  Tokenizer not found in GGUF dir, using model's config")
        tokenizer = None

    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Config: hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

    # --- Apply LoRA ---
    print(f"\nApplying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA}, Q/V only)...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Tokenize training data ---
    print(f"\nTokenizing {len(TRAIN_TEXTS)} sequences (seq_len={SEQ_LEN})...")
    if tokenizer is None:
        # Fallback: use simple byte-level encoding for benchmark purposes
        print("  WARNING: No tokenizer available, using dummy token IDs")
        input_ids_list = []
        labels_list = []
        for text in TRAIN_TEXTS:
            tokens = list(range(1, SEQ_LEN + 2))  # dummy tokens
            input_ids_list.append(tokens[:SEQ_LEN])
            labels_list.append(tokens[1:SEQ_LEN + 1])
            print(f"  Seq: {SEQ_LEN} tokens (dummy)")
    else:
        input_ids_list = []
        labels_list = []
        for text in TRAIN_TEXTS:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > SEQ_LEN + 1:
                tokens = tokens[:SEQ_LEN + 1]
            elif len(tokens) < SEQ_LEN + 1:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                tokens = tokens + [pad_id] * (SEQ_LEN + 1 - len(tokens))

            input_ids_list.append(tokens[:SEQ_LEN])
            labels_list.append(tokens[1:SEQ_LEN + 1])
            print(f"  Seq: {len(tokens)-1} tokens")

    input_ids_tensor = torch.tensor(input_ids_list, device=device)  # (4, SEQ_LEN)
    labels_tensor = torch.tensor(labels_list, device=device)  # (4, SEQ_LEN)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # --- Training loop ---
    print(f"\nTraining ({NUM_STEPS} steps, lr={LR}, wd={WEIGHT_DECAY})...")
    model.train()

    total_fwd = 0
    total_bwd = 0
    total_opt = 0

    timer_start = time.time()

    for step in range(NUM_STEPS):
        step_loss = 0.0

        for seq_idx in range(len(TRAIN_TEXTS)):
            input_ids = input_ids_tensor[seq_idx:seq_idx+1]  # (1, SEQ_LEN)
            labels = labels_tensor[seq_idx:seq_idx+1]  # (1, SEQ_LEN)

            # Forward
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.time()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / len(TRAIN_TEXTS)  # gradient accumulation

            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            total_fwd += time.time() - t_fwd

            # Backward
            t_bwd = time.time()
            loss.backward()

            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            total_bwd += time.time() - t_bwd

            step_loss += loss.item() * len(TRAIN_TEXTS)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

        # Optimizer step
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t_opt = time.time()

        optimizer.step()
        optimizer.zero_grad()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        total_opt += time.time() - t_opt

        avg_loss = step_loss / len(TRAIN_TEXTS)
        if step < 5 or step % 10 == 0 or step == NUM_STEPS - 1:
            print(f"  Step {step:>3}: loss = {avg_loss:.4f}")

    total_time = time.time() - timer_start
    ms_per_step = total_time / NUM_STEPS * 1000

    print(f"\n=== Results ===")
    print(f"  Total training time: {total_time*1000:.0f}ms ({ms_per_step:.0f}ms/step)")
    print(f"  Profile: forward={total_fwd*1000:.0f}ms, backward={total_bwd*1000:.0f}ms, optimizer={total_opt*1000:.0f}ms")
    print(f"  Note: Model is float16 (GGUF dequantized). Zig uses Q4/Q8 quantized weights.")

    # --- Generate after fine-tuning ---
    if tokenizer is not None:
        print("\n  Generating text after fine-tuning:")
        model.eval()
        for prompt in ["The capital of Japan", "The capital of France"]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=1.0,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'    "{text}"')


if __name__ == "__main__":
    main()
