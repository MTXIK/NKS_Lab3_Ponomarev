import gc
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen3-1.7B"
LORA_ADAPTER_PATH = "./qwen3_book_lora_adapter"

SYSTEM_PROMPT = (
    "You are an educational assistant specializing in practical machine learning. "
    "Answer clearly and directly based on the topic of the book "
    "'Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow'. "
    "Do not show reasoning. Do not use <think> blocks."
)


def clear_cuda_cache():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model():
    print("[INFO] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_ADAPTER_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("[INFO] Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    base_model.config.use_cache = True

    print("[INFO] Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
    )

    model.eval()

    print("[INFO] Model is ready.")

    return tokenizer, model


def build_prompt(tokenizer, user_message, history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for item in history[-5:]:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})

    messages.append({
        "role": "user",
        "content": user_message + " /no_think",
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return prompt


def generate_answer(tokenizer, model, user_message, history, max_new_tokens=250):
    prompt = build_prompt(tokenizer, user_message, history)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.time() - start_time

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not answer:
        answer = "[The model did not generate an answer.]"

    return answer, elapsed


def main():
    clear_cuda_cache()

    print("=" * 80)
    print("NKS Lab 3 — Console Chat")
    print("Model: Qwen/Qwen3-1.7B + LoRA adapter")
    
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("Type 'exit' or 'выход' to stop.")
    print("=" * 80)

    tokenizer, model = load_model()

    history = []

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"exit", "quit", "выход"}:
            print("Exiting...")
            break

        if not user_message:
            continue

        answer, elapsed = generate_answer(
            tokenizer=tokenizer,
            model=model,
            user_message=user_message,
            history=history,
            max_new_tokens=250,
        )

        history.append({
            "user": user_message,
            "assistant": answer,
        })

        print("\nAssistant:")
        print(answer)
        print(f"\n[Latency: {elapsed:.2f} sec]")


if __name__ == "__main__":
    main()