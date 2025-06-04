from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path: str):
    """
    Loads a pre-trained CausalLM model and its tokenizer.

    Args:
        model_path (str): The path to the directory containing the saved model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model, placed on the appropriate device.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {e}")
        # You might want to handle this more robustly, e.g., raise an error or exit
        return None, None 

    # Load model onto the appropriate device with suitable dtype (Gemma often uses float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    return tokenizer, model, device

# --- Usage Example ---
# Define your model path
model_directory_path = "./fine_tuned_model"

# Load the model and tokenizer using the new function
tokenizer, model, device = load_model_and_tokenizer(model_directory_path)

# Ensure both were loaded successfully before proceeding
if tokenizer is not None and model is not None:
    def answer_question(question: str) -> str:
        prompt = question.strip() + "\nTrả lời: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        return answer

    question = "Bảo hiểm y tế ở Việt Nam là cái gì?"
    print("Bot:", answer_question(question))
else:
    print("Failed to load model or tokenizer. Exiting.")