import gc
import time
import torch

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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


app = FastAPI(
    title="NKS Lab 3 — Fine-tuned Qwen3 Chat API",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatMessage(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    latency_seconds: float


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

    print("[INFO] Model loaded successfully.")

    return tokenizer, model


clear_cuda_cache()
tokenizer, model = load_model()


def build_prompt(user_message: str, history: list[ChatMessage]) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for item in history[-5:]:
        messages.append({
            "role": "user",
            "content": item.user,
        })
        messages.append({
            "role": "assistant",
            "content": item.assistant,
        })

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


def generate_answer(user_message: str, history: list[ChatMessage], max_new_tokens: int = 250):
    prompt = build_prompt(user_message, history)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )

    latency = time.time() - start_time

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not answer:
        answer = "[The model did not generate an answer.]"

    return answer, latency


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter": LORA_ADAPTER_PATH,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is empty.")

        answer, latency = generate_answer(
            user_message=request.message,
            history=request.history,
            max_new_tokens=250,
        )

        return ChatResponse(
            response=answer,
            latency_seconds=latency,
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )