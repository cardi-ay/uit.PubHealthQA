import os
import torch
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig

nltk.download('punkt')
os.makedirs('./img', exist_ok=True)

def load_model_and_tokenizer(model_name='google/gemma-3-1b-it'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    return model, tokenizer

def load_and_preprocess_data(path, tokenizer, max_length=1024):
    df = pd.read_json(path)
    df['text'] = df['question'].str.strip() + "\nTrả lời: " + df['answer'].str.strip()
    dataset = Dataset.from_pandas(df[['text']])

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch['text'],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        tokens['labels'] = tokens['input_ids']
        return tokens

    return dataset.map(tokenize_fn, batched=True)

def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

def train(model_name, data_path, output_dir, use_lora=True, epochs=3, max_token_length = 1024):
    model, tokenizer = load_model_and_tokenizer(model_name)
    if use_lora:
        model = apply_lora(model)
    dataset = load_and_preprocess_data(data_path, tokenizer, max_length = max_token_length)
    split = dataset.train_test_split(test_size=0.2)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)

def main():
    train(
        model_name='google/gemma-3-1b-it',
        data_path='data/gold/question_answer_pairs.json',
        output_dir='./fine_tuned_model',
        use_lora=True,
        epochs=6,
        max_token_length=512
    )

if __name__ == '__main__':
    main()
