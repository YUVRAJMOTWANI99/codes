import os

import evaluate
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer)

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)


class PreparePipeline:

    def _re_ft_local_transformer(self, pipeline: str, base_files_path: str) -> None:
        """
        Fine tune local transformer model after intent updates
        """
        # Prepare data for fine-tuning
        # Load raw dataset
        raw_df = pd.read_csv(f"{base_files_path}/generated_examples_for_ft.csv")
        raw_df.columns = ["category", "text"]
        # Add integer ID to intent labels
        labels = raw_df['category'].unique()
        iD = range(len(labels))
        label2id = dict(zip(labels, iD))
        id2label = dict(zip(iD, labels))
        raw_df["category_id"] = raw_df["category"].map(label2id)
        # Train test split
        final_data = raw_df.sample(frac=1)  # shuffle
        _, _, _, _, _, train_idx = train_test_split(
            final_data,
            final_data["category"],
            final_data.index,
            test_size=config[pipeline]["FINETUNE"]["TEST_SPLIT"],
            stratify=final_data["category"]
        )
        final_data["train"] = 0
        final_data.loc[train_idx, "train"] = 1

        # Save preprocessed data
        final_data.to_csv(f"{base_files_path}/preprocessed_data.csv", index=False)

        # train test split
        X_train = final_data[final_data["train"] == 1]["text"]
        y_train = final_data[final_data["train"] == 1]["category_id"]
        X_test = final_data[final_data["train"] == 0]["text"]
        y_test = final_data[final_data["train"] == 0]["category_id"]

        # create new dataset
        dataset = DatasetDict(
            {
                'train': Dataset.from_dict({'label': y_train.values, 'text': X_train.values}),
                'validation': Dataset.from_dict({'label': y_test.values, 'text': X_test.values})
            }
        )

        # Load pre-trained base model (not fine-tuned for any downstream task)
        model_checkpoint = config[pipeline]["FINETUNE"]["MODEL_CHECKPOINT"]
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{model_checkpoint}",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            force_download=True
        )

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{model_checkpoint}", add_prefix_space=True)
        # Add pad token if none exists
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # Create tokenize function
        def tokenize_function(examples):
            # extract text
            text = examples["text"]
            # tokenize and truncate text
            tokenizer.truncation_side = "left"
            tokenized_inputs = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=512
            )
            return tokenized_inputs

        # tokenize training and validation datasets
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # create data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        # Import accuracy evaluation metric
        accuracy = evaluate.load("accuracy")

        # define an evaluation function to pass into trainer later
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

        # Define Parameters
        # Define peft configuration
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            target_modules=[config[pipeline]["FINETUNE"]["LORA_TARGET_MODULE"]],
            r=config[pipeline]["FINETUNE"]["LORA_R"],
            lora_alpha=config[pipeline]["FINETUNE"]["LORA_ALPHA"],
            lora_dropout=config[pipeline]["FINETUNE"]["LORA_DROPOUT"]
        )
        model = get_peft_model(model, peft_config)
        # define training arguments
        training_args = TrainingArguments(
            output_dir=f"{base_files_path}/",
            learning_rate=config[pipeline]["FINETUNE"]["LORA_LR"],
            per_device_train_batch_size=config[pipeline]["FINETUNE"]["LORA_BATCH_SIZE"],
            per_device_eval_batch_size=config[pipeline]["FINETUNE"]["LORA_BATCH_SIZE"],
            num_train_epochs=config[pipeline]["FINETUNE"]["LORA_NUM_EPOCHS"],
            weight_decay=0.01,
            # evaluation_strategy="epoch",  #remove comment to see eval acc at each epoch
            save_strategy="no",
            load_best_model_at_end=False
        )

        # Create trainer object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
            compute_metrics=compute_metrics,
        )
        # Train model
        trainer.train()

        # Save model to disk
        trainer.save_model(f"{base_files_path}/{config[pipeline]['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft")

    def _is_model_present(self, pipeline: str, base_files_path: str) -> bool:
        """
        Check if a fine-tuned model is already present
        """
        model_file_path = f"{base_files_path}/{config[pipeline]['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft"
        return os.path.isdir(model_file_path)

    def _download_n_finetune_model(self, pipeline: str, base_files_path: str) -> None:
        """
        Download and finetune a local LLM from huggingface
        """
        print(f"\nPreparing pipeline {pipeline}...")
        if self._is_model_present(pipeline, base_files_path=base_files_path):
            print(f"A fine-tuned model already present for {pipeline}!")
        else:
            print(f"No model found for {pipeline}!")
            print(f"Downloading and fine-tuning a local LLM for {pipeline}...")
            self._re_ft_local_transformer(pipeline, base_files_path=base_files_path)
        print(f"Pipeline {pipeline} is now ready for usage!")

    def run(self) -> None:
        """
        Prepare or download model files for inference
        """
        # self._download_n_finetune_model("P3", "data/pipeline_3")
        # self._download_n_finetune_model("P4", "data/pipeline_4")
        self._download_n_finetune_model("P5", "data/pipeline_5")
