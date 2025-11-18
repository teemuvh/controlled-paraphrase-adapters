import json
import evaluate
import numpy as np
import torch

from datasets import load_dataset
from itertools import chain

from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    AdapterTrainer,
    EvalPrediction,
    Seq2SeqAdapterTrainer,
    AutoTokenizer,
)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", cache_dir="./tmp")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", cache_dir="./tmp")

    model.add_adapter("paraphrase-adapter", config="pfeiffer")
    model.to(device)

    data_files = {}
    data_files["train"] = "data/old_data/opusparcus-en-train-1m.json"
    data_files["eval"] = "data/dev/en-dev.pos.json"

    raw_datasets = load_dataset("json", data_files=data_files, cache_dir="./tmp")
    column_names = raw_datasets["train"].column_names
    column_names = raw_datasets["eval"].column_names
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["eval"]


    def preprocess_function(examples):
        prefix = "paraphrase this sentence: "
        model_inputs = {}

        inputs = [ex["sentence1"] for ex in examples["paraphrase"]]
        targets = [ex["sentence2"] for ex in examples["paraphrase"]]
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=100, padding="max_length", truncation=True)
        labels = tokenizer(text=targets, max_length=100, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"], "labels": model_inputs["labels"]}


    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset"
    )

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    metric = evaluate.load("sacrebleu")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds_flat = list(chain.from_iterable(preds))
        labels_flat = list(chain.from_iterable(labels))

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    model.train_adapter("paraphrase-adapter")

    training_args = Seq2SeqTrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_steps=500,
        save_steps=500,
        save_total_limit=5,
        output_dir="./training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(max_length=100, num_beams=1, metric_key_prefix="eval")

    model.save_adapter("./trained_adapter", "paraphrase-adapter")


if __name__ == "__main__":
    main()
