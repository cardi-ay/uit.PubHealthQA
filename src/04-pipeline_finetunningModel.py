import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

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
        df = pd.read_json(r'data\gold\question_answer_pairs.json')
    except FileNotFoundError:
        print('Training dataset can not be found!')
        return None
    except Exception as e:
        print(f'There is an error while loading dataset for fine-tunning as: {e}')
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

def preprocess_data(df: pd.DataFrame, tokenizer, text_column: str = "text", max_length: int = 512) -> dict:
    """
    Tokenize và định dạng dataset
    
    Args:
        df: DataFrame chứa dữ liệu
        tokenizer: Tokenizer đã load
        text_column: Tên cột chứa văn bản
        max_length: Độ dài tối đa của input
    
    Returns:
        Dict chứa tokenized datasets
    """
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)
    
    return {
        "train": df.apply(tokenize_function, axis=1).tolist()  # Giả sử toàn bộ df là train
    }

# 2. Function thiết lập LoRA
def setup_lora(model, lora_r: int = 8, lora_alpha: int = 16) -> torch.nn.Module:
    """
    Áp dụng LoRA cho model
    
    Args:
        model: Model gốc
        lora_r: Rank của LoRA
        lora_alpha: Scaling factor
        
    Returns:
        Model đã áp dụng LoRA
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)

# 3. Function thiết lập Trainer
def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "./results",
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    epochs: int = 3
) -> Trainer:
    """
    Tạo Trainer object cho fine-tuning
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer đã load
        train_dataset: Dataset đã tokenize
        output_dir: Thư mục output
        batch_size: Batch size
        learning_rate: Tốc độ học
        epochs: Số epoch
        
    Returns:
        Trainer object
    """
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
    model_name: str = "google/gemma-2b-it",
    dataset_path: str = "data/gold/question_answer_pairs.json",
    output_dir: str = "./fine_tuned_model"
) -> None:
    """
    End-to-end fine-tuning pipeline
    
    Args:
        model_name: Tên model trên Hugging Face
        dataset_path: Đường dẫn dataset
        output_dir: Thư mục lưu model
    """
    try:
        # Load model và tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load và tiền xử lý dữ liệu
        df = pd.read_json(dataset_path)
        tokenized_data = preprocess_data(df, tokenizer)
        
        # Áp dụng LoRA
        model = setup_lora(model)
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Huấn luyện
        trainer = setup_trainer(model, tokenizer, tokenized_data["train"])
        trainer.train()
        
        # Lưu model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in fine-tuning pipeline: {str(e)}")
        raise








def main():
    dataset = load_dataset()
    if dataset is None:
        return None
    
    model, tokenizer = load_model()  # Đổi tên biến thành 'tokenizer'
    if model is None or tokenizer is None:  # Kiểm tra cả tokenizer
        return None
    
    print(dataset.head(10))

    return 0

if __name__ == '__main__':
    logicCode = main()
    if logicCode is None:
        print('0')
        # print('The programming encountered an error so it has been stopped immidiately!)
    else:
        print('1')