from datasets import load_dataset
import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_datasets as tfds
from peft import LoraConfig, get_peft_model, peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
import os
import numpy as np
from sklearn.metrics import log_loss
from torch import softmax
from datasets import load_metric
import evaluate


class LoRATuner:
    """
    Handles all the pretraining. Initialize with the base model.
    """
    def __init__(self, model_name) -> None:
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tuned_model = None
    
    def print_num_trainable_features(self):
        """
        Prints out the trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.base_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    # def print_target_modules(self):
    #     print("Possible target modules for ", self.model.base_model._get_name())
    #     for name, module in self.model.named_modules():
    #         if "Linear4bit" in str(type(module)):
    #             print(name.split(".")[-1])

    #Citation, got this from: https://212digital.medium.com/evaluating-gpt-2-language-model-a-step-by-step-guide-b451339e6a41
    #def compute_metrics(self, eval_pred):
        #logits = p.predictions
        #labels = p.label_ids
        #probabilities = softmax(logits, axis=-1)
        #loss = log_loss(labels.flatten(), probabilities.reshape(-1, probabilities.shape[-1]), labels=[i for i in range(logits.shape[-1])])
        #perplexity = np.exp(loss)
        #return {"perplexity": perplexity}
    #    perplex = load_metric('perplexity')
    #    predictions, labels = eval_pred
    #    predictions = predictions[:, 0]
    #    return perplex.compute(predictions=predictions, references=labels)


    #Citation: Got this function template from https://212digital.medium.com/fine-tuning-the-gpt-2-large-language-model-unlocking-its-full-potential-66e3a082ab9c
    def tune(self, train_file_name, val_file_name, model_id, config, eval_steps=500, logging_steps=500, save_steps = 500, epochs = 1, max_steps = -1, seed = 0):
        train_file = os.path.join("../data/", train_file_name)
        val_file = os.path.join("../data/", val_file_name)
        print(val_file)
        print(train_file)
        lora_model = get_peft_model(self.base_model, config)
        #def compute_metrics(p):
        #    metric = evaluate.load("perplexity", module_type="metric")
            #print(eval_preds)
            #logits, labels = eval_preds
            #print("logits")
            #print(logits)
            #print("labels")
            #print(labels)
            #predictions = np.argmax(logits, axis=-1)
            #return metric.compute(predictions=predictions, references=labels)
            #logits = p.predictions
            #labels = p.label_ids
            #probabilities = softmax(logits, axis=-1)
            #loss = log_loss(labels.flatten(), probabilities.reshape(-1, probabilities.shape[-1]), labels=[i for i in range(logits.shape[-1])])
            #perplexity = np.exp(loss)
            #return {"perplexity": perplexity}
        
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=128)
        
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=val_file,
            block_size=128)
        print(val_dataset)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=model_id,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=save_steps,
            save_strategy="steps",
            evaluation_strategy="steps", 
            do_eval=True,
            eval_steps=eval_steps, 
            logging_dir='./logs',
            logging_steps=logging_steps,
            max_steps=max_steps,
            eval_accumulation_steps=10, 
            seed = seed,
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True ,
            label_names = ["labels"]
        )
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer, 
            #compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience = 2)]
        )
        trainer.train()
        self.tuned_model = lora_model
        lora_model.save_pretrained(model_id)

    def create_pipeline(checkpoint_path):
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        return pipeline('text-generation', model=model, tokenizer = tokenizer)

    def get_response(pipeline, prompt, checkpoint_path):
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        pip = pipeline('text-generation', model=model, tokenizer = tokenizer)
        return pip(prompt)[0]['generated_text']