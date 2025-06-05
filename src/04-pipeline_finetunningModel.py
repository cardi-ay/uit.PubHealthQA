import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')

# Ensure image directory exists
os.makedirs('./img', exist_ok=True)

class FineTuningMetrics:
    def __init__(self, tokenizer, embedder_model='keepitreal/vietnamese-sbert'):
        self.tokenizer = tokenizer
        self.embedder = SentenceTransformer(embedder_model)
        self.smoother = SmoothingFunction()
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'bleu_score': [],
            'cosine_sim': [],
            'epoch': []
        }
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU scores
        bleu_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = nltk.word_tokenize(pred)
            label_tokens = nltk.word_tokenize(label)
            bleu_scores.append(
                sentence_bleu([label_tokens], pred_tokens, smoothing_function=self.smoother.method1)
            )
        avg_bleu = np.mean(bleu_scores)
        
        # Compute cosine similarity
        embeddings_pred = self.embedder.encode(decoded_preds)
        embeddings_label = self.embedder.encode(decoded_labels)
        cosine_sims = []
        for i in range(len(embeddings_pred)):
            sim = cosine_similarity(
                embeddings_pred[i].reshape(1, -1),
                embeddings_label[i].reshape(1, -1)
            )[0][0]
            cosine_sims.append(sim)
        avg_cosine = np.mean(cosine_sims)
        
        return {
            'bleu_score': avg_bleu,
            'cosine_sim': avg_cosine
        }
    
    def update_metrics(self, train_loss=None, eval_loss=None, bleu_score=None, cosine_sim=None, epoch=None):
        if train_loss is not None:
            self.metrics_history['train_loss'].append(train_loss)
        if eval_loss is not None:
            self.metrics_history['eval_loss'].append(eval_loss)
        if bleu_score is not None:
            self.metrics_history['bleu_score'].append(bleu_score)
        if cosine_sim is not None:
            self.metrics_history['cosine_sim'].append(cosine_sim)
        if epoch is not None:
            self.metrics_history['epoch'].append(epoch)
    
    def plot_metrics(self, fold=None):
        plt.figure(figsize=(15, 10))
        
        # Plot training and evaluation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['epoch'], self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['epoch'], self.metrics_history['eval_loss'], label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Evaluation Loss')
        plt.legend()
        
        # Plot BLEU score
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['epoch'], self.metrics_history['bleu_score'])
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('BLEU Score')
        
        # Plot Cosine Similarity
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history['epoch'], self.metrics_history['cosine_sim'])
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.title('Cosine Similarity')
        
        # Save plot
        filename = f"./img/metrics_{fold}.png" if fold is not None else "./img/metrics.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        csv_filename = f"./img/metrics_{fold}.csv" if fold is not None else "./img/metrics.csv"
        metrics_df.to_csv(csv_filename, index=False)

def load_dataset() -> pd.DataFrame | None:
    '''Load dataset for fine-tuning.'''
    try:
        df = pd.read_json('data/gold/question_answer_pairs.json')
        return df
    except Exception as e:
        print(f'Error loading dataset: {e}')
        return None

def load_model(model_name: str = 'google/gemma-3-1b-it') -> tuple:
    """Load model and tokenizer with quantization."""
    try:
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
            trust_remote_code=True
        )
        
        return model, tokenizer
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        return None, None

def preprocess_data(df: pd.DataFrame, tokenizer, max_length: int = 512):
    """Preprocess and tokenize data."""
    df["text"] = df["question"].str.strip() + "\nTrả lời: " + df["answer"].str.strip()
    
    # Create both input and target columns for evaluation
    df["input_text"] = df["question"].str.strip()
    df["target_text"] = df["answer"].str.strip()
    
    dataset = Dataset.from_pandas(df[["text", "input_text", "target_text"]])
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Create labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)

def setup_lora(model, lora_r: int = 8, lora_alpha: int = 16):
    """Configure LoRA for the model."""
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA configured. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}/{sum(p.numel() for p in model.parameters())}")
        return model
    except Exception as e:
        print(f"Error setting up LoRA: {e}")
        return model

def generate_predictions(model, tokenizer, dataset, max_length=50):
    """Generate predictions for evaluation."""
    model.eval()
    predictions = []
    references = []
    
    for example in dataset:
        input_text = example["input_text"]
        target_text = example["target_text"]
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(decoded_output)
        references.append(target_text)
    
    return predictions, references

def evaluate_model(model, tokenizer, dataset, metrics_tracker):
    """Evaluate model and compute metrics."""
    predictions, references = generate_predictions(model, tokenizer, dataset)
    
    # Compute BLEU scores
    bleu_scores = []
    smoother = SmoothingFunction()
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)
        bleu_scores.append(
            sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother.method1)
        )
    avg_bleu = np.mean(bleu_scores)
    
    # Compute cosine similarity
    embedder = SentenceTransformer('keepitreal/vietnamese-sbert')
    embeddings_pred = embedder.encode(predictions)
    embeddings_ref = embedder.encode(references)
    cosine_sims = []
    for i in range(len(embeddings_pred)):
        sim = cosine_similarity(
            embeddings_pred[i].reshape(1, -1),
            embeddings_ref[i].reshape(1, -1)
        )[0][0]
        cosine_sims.append(sim)
    avg_cosine = np.mean(cosine_sims)
    
    metrics_tracker.update_metrics(
        bleu_score=avg_bleu,
        cosine_sim=avg_cosine
    )
    
    return {
        'bleu_score': avg_bleu,
        'cosine_sim': avg_cosine
    }

def setup_trainer(model, tokenizer, train_dataset, eval_dataset=None, 
                output_dir="./results", batch_size=4, learning_rate=2e-5, 
                epochs=3, metrics_tracker=None):
    """Configure the Trainer with metrics tracking."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        logging_strategy="epoch",
        fp16=True,
        report_to="tensorboard",
        load_best_model_at_end=True if eval_dataset is not None else False
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU scores
        bleu_scores = []
        smoother = SmoothingFunction()
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = nltk.word_tokenize(pred)
            label_tokens = nltk.word_tokenize(label)
            bleu_scores.append(
                sentence_bleu([label_tokens], pred_tokens, smoothing_function=smoother.method1)
            )
        avg_bleu = np.mean(bleu_scores)
        
        # Compute cosine similarity
        embedder = SentenceTransformer('keepitreal/vietnamese-sbert')
        embeddings_pred = embedder.encode(decoded_preds)
        embeddings_label = embedder.encode(decoded_labels)
        cosine_sims = []
        for i in range(len(embeddings_pred)):
            sim = cosine_similarity(
                embeddings_pred[i].reshape(1, -1),
                embeddings_label[i].reshape(1, -1)
            )[0][0]
            cosine_sims.append(sim)
        avg_cosine = np.mean(cosine_sims)
        
        if metrics_tracker:
            metrics_tracker.update_metrics(
                bleu_score=avg_bleu,
                cosine_sim=avg_cosine
            )
        
        return {
            'bleu_score': avg_bleu,
            'cosine_sim': avg_cosine
        }

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset is not None else None
    )

def k_fold_fine_tuning(model, tokenizer, dataset, n_splits=5, epochs_per_fold=3, 
                      output_dir_base="./kfold_results", use_lora=True):
    """Perform k-fold cross-validation with metrics tracking."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_metrics = []
    
    # Convert dataset to pandas DataFrame for splitting
    df = dataset.to_pandas()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n=== Starting Fold {fold + 1}/{n_splits} ===")
        
        # Create output directory for this fold
        output_dir = f"{output_dir_base}/fold_{fold + 1}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Split data
        train_data = dataset.select(train_idx.tolist())
        val_data = dataset.select(val_idx.tolist())
        
        # Initialize metrics tracker for this fold
        metrics_tracker = FineTuningMetrics(tokenizer)
        
        # Setup trainer
        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            output_dir=output_dir,
            epochs=epochs_per_fold,
            metrics_tracker=metrics_tracker
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate(val_data)
        
        # Generate predictions for additional evaluation
        additional_metrics = evaluate_model(model, tokenizer, val_data, metrics_tracker)
        
        # Update metrics with epoch information
        for epoch in range(epochs_per_fold):
            metrics_tracker.update_metrics(
                epoch=epoch + 1,
                train_loss=train_result.metrics[f'epoch_{epoch + 1}_train_loss'],
                eval_loss=eval_result[f'epoch_{epoch + 1}_eval_loss'] if f'epoch_{epoch + 1}_eval_loss' in eval_result else None
            )
        
        # Save metrics and plots
        metrics_tracker.plot_metrics(fold=fold + 1)
        all_metrics.append(metrics_tracker.metrics_history)
        
        # Save fold results
        fold_results.append({
            'fold': fold + 1,
            'train_loss': train_result.training_loss,
            'eval_loss': eval_result['eval_loss'],
            'bleu_score': additional_metrics['bleu_score'],
            'cosine_sim': additional_metrics['cosine_sim']
        })
        
        # Save model for this fold
        trainer.save_model(f"{output_dir}/model")
        
        # Clear memory
        torch.cuda.empty_cache()
    
    # Save combined metrics across all folds
    combined_metrics = pd.concat([pd.DataFrame(m) for m in all_metrics])
    combined_metrics.to_csv(f"{output_dir_base}/all_folds_metrics.csv", index=False)
    
    return pd.DataFrame(fold_results)

def fine_tune_pipeline(
    model_name: str = "google/gemma-3-1b-it",
    dataset_path: str = "data/gold/question_answer_pairs.json",
    output_dir: str = "./fine_tuned_model",
    use_lora: bool = True,
    max_length: int = 512,
    epochs: int = 3,
    k_fold: int = None
) -> bool:
    """Main fine-tuning pipeline with metrics tracking."""
    try:
        # 1. Load model
        model, tokenizer = load_model(model_name)
        if model is None or tokenizer is None:
            return False

        # 2. Load and validate dataset
        df = load_dataset()
        if df is None or len(df) < 10:
            print("Dataset too small or empty")
            return False

        # 3. Preprocess data
        tokenized_data = preprocess_data(df, tokenizer, max_length)
        if len(tokenized_data) == 0:
            return False

        # 4. Apply LoRA if enabled
        if use_lora:
            model = setup_lora(model)
            if model is None:
                return False

        # Initialize metrics tracker
        metrics_tracker = FineTuningMetrics(tokenizer)

        # 5. Training - either k-fold or standard
        if k_fold and k_fold > 1:
            print(f"Starting {k_fold}-fold cross-validation training...")
            results = k_fold_fine_tuning(
                model=model,
                tokenizer=tokenizer,
                dataset=tokenized_data,
                n_splits=k_fold,
                epochs_per_fold=epochs,
                output_dir_base=output_dir,
                use_lora=use_lora
            )
            print("\nCross-validation results:")
            print(results)
            results.to_csv(f"{output_dir}/cross_validation_results.csv", index=False)
        else:
            print("Starting standard training...")
            # Split data into train and eval (80-20)
            train_test_split = tokenized_data.train_test_split(test_size=0.2)
            train_data = train_test_split['train']
            eval_data = train_test_split['test']
            
            trainer = setup_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_data,
                eval_dataset=eval_data,
                output_dir=output_dir,
                epochs=epochs,
                metrics_tracker=metrics_tracker
            )
            
            # Train and evaluate
            trainer.train()
            trainer.evaluate()
            
            # Update metrics with epoch information
            for epoch in range(epochs):
                metrics_tracker.update_metrics(epoch=epoch + 1)
            
            # Save final metrics and model
            metrics_tracker.plot_metrics()
            trainer.save_model(output_dir)

        print("Training completed successfully")
        return True

    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        return False

def main() -> int:
    try:
        success = fine_tune_pipeline(
            model_name="google/gemma-3-1b-it",
            dataset_path="data/gold/question_answer_pairs.json",
            output_dir="./fine_tuned_model",
            use_lora=True,
            max_length=512,
            epochs=3,
            k_fold=5  # Set to 5 for k-fold cross-validation
        )
        return 0 if success else 1
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
if __name__ == '__main__':
    main()