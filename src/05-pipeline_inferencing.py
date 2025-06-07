import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Tắt Triton và TorchInductor ---
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# (Tuỳ chọn) Bật hỗ trợ TF32 nếu card GPU có Tensor Core
torch.set_float32_matmul_precision('high')

# --- Load biến môi trường từ file .env ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Các hàm helper ---
def _load_model_and_tokenizer():
    """
    Load mô hình gốc và mô hình đã fine-tune (chỉ chạy một lần).
    Trả về tokenizer và model đã sẵn sàng để sinh văn bản.
    """
    base_model = "google/gemma-3-1b-it"
    model_dir = "./fine_tuned_model"  # Đường dẫn chứa trọng số LoRA

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    return tokenizer, model

def _generate_response(prompt, tokenizer, model):
    """
    Sinh câu trả lời dựa trên prompt người dùng.
    Trả về chuỗi văn bản đã xử lý.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Xử lý hậu kỳ: loại bỏ prompt nếu bị lặp lại
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    return full_output.strip()

# --- Hàm chính input → output ---
def get_health_advice(user_input):
    """
    Nhận input người dùng và trả về câu trả lời từ model.
    Ví dụ: "Tại sao rửa tay quan trọng?" → "Rửa tay giúp phòng ngừa vi khuẩn..."
    """
    # Lazy load mô hình
    if not hasattr(get_health_advice, 'tokenizer'):
        get_health_advice.tokenizer, get_health_advice.model = _load_model_and_tokenizer()

    # Prompt mẫu
    prompt = (
        "### Vai trò:\n"
        "Bạn là trợ lý ảo chuyên về sức khỏe cộng đồng tại Việt Nam.\n\n"
        "### Yêu cầu:\n"
        "- Trả lời rõ ràng, chính xác, dễ hiểu, ngắn gọn.\n"
        "- Nếu không biết thì phải trả lời là không biết.\n\n"
        f"### Câu hỏi:\n{user_input}\n\n"
        "### Trả lời:"
    )

    # Sinh câu trả lời
    return _generate_response(prompt, get_health_advice.tokenizer, get_health_advice.model)

# --- Chạy thử từ command line ---
if __name__ == "__main__":
    print("🩺 Trợ lý sức khỏe AI. Gõ câu hỏi và nhấn Enter (để thoát, nhấn Enter rỗng).")
    while True:
        user_question = input("🧑 Bạn hỏi: ").strip()
        if not user_question:
            print("Tạm biệt! 👋")
            break
        answer = get_health_advice(user_question)
        print("🤖 Trả lời:", answer)
