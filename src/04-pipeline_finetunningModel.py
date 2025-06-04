import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model




def load_dataset() -> pd.DataFrame | None:
    '''
    This function is to load dataset for fine-tunning.

    Args:
        None.

    Returns:
        A `pd.DataFrame` dataset if successful. Otherwise, it returns `None`.
    '''
    try:
        df = pd.read_json('data/gold/question_answer_pairs.json')
    except FileNotFoundError:
        print('Training dataset cannot be found!')
        return None
    except Exception as e:
        print(f'Error loading dataset: {e}')
        return None
    return df

def load_model(model_name: str = 'google/gemma-3-1b-it') -> tuple:
    """
    Load a pre-trained model and tokenizer for fine-tuning with error handling.
    
    Args:
        model_name (str): Name/path of the model (e.g., "google/gemma-2b-it")
    
    Returns:
        tuple: (model, tokenizer) if successful, `None` on failure
    """
    try:
        # Configure 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically place layers on available devices
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"Successfully loaded model {model_name} and tokenizer")
        print(f"Model device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        return model, tokenizer

    except Exception as e:
        print(f'Error while loading model for fine-tuning: {str(e)}')
        return None, None

def preprocess_data(df: pd.DataFrame, tokenizer, max_length: int = 512):
    df["text"] = df["question"].str.strip() + "\nTrả lời: " + df["answer"].str.strip()
    
    # Kiểm tra độ dài trước khi tokenize
    df["text_length"] = df["text"].apply(lambda x: len(tokenizer.tokenize(x)))
    if (df["text_length"] > max_length).any():
        print(f"Warning: Some texts exceed max length {max_length} and will be truncated")
    
    dataset = Dataset.from_pandas(df[["text"]])
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"  # Thêm để đảm bảo đầu ra là tensor
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 2. Function thiết lập LoRA
def setup_lora(model, lora_r: int = 8, lora_alpha: int = 16):
    # Nên kiểm tra model có hỗ trợ LoRA không
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Cần phù hợp với kiến trúc Gemma
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA configured. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}/{sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"Error setting up LoRA: {e}")
        return model  # Trả về model gốc nếu LoRA fail

# 3. Function thiết lập Trainer
def setup_trainer(model, tokenizer, train_dataset, output_dir="./results", batch_size=4, learning_rate=2e-5, epochs=3):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        save_steps=500,
        logging_steps=100,
        report_to="tensorboard"
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

# 4. Main function
def fine_tune_pipeline(
    model_name: str = "google/gemma-3-1b-it",
    dataset_path: str = "data/gold/question_answer_pairs.json",
    output_dir: str = "./fine_tuned_model",
    use_lora: bool = True,
    max_length: int = 512
) -> bool:  # Thay None bằng bool để biết kết quả
    try:
        # 1. Load model
        model, tokenizer = load_model(model_name)
        if model is None or tokenizer is None:
            return False

        # 2. Load và validate dataset
        df = pd.read_json(dataset_path)
        if df.empty or len(df) < 10:  # Kiểm tra dataset đủ lớn
            print("Dataset too small or empty")
            return False

        # 3. Tiền xử lý
        tokenized_data = preprocess_data(df, tokenizer, max_length)
        if len(tokenized_data) == 0:
            return False

        # 4. LoRA
        if use_lora:
            model = setup_lora(model)
            if model is None:
                return False

        # 5. Training
        trainer = setup_trainer(model, tokenizer, tokenized_data, output_dir)
        trainer.train()

        # 6. Lưu model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 7. Đánh giá sau training
        print("Training completed successfully")
        return True

    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return False








def main() -> int:
    try:
        dataset = load_dataset()
        if dataset is None:
            return 1  # Error code
        
        success = fine_tune_pipeline(
            model_name="google/gemma-3-4b-it",
            dataset_path="data/gold/question_answer_pairs.json",
            output_dir="./fine_tuned_model",
            use_lora=True,
            max_length=512
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
if __name__ == '__main__':
    main()