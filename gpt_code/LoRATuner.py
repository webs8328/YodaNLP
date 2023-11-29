from datasets import load_dataset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
import os
from transformers import pipeline, set_seed
import numpy as np
from datasets import load_metric
import evaluate
from transformers import set_seed
from scipy.special import softmax
from sklearn.metrics import log_loss
import sklearn
import torch
from tqdm import tqdm
from evaluate import load
import random


class LoRATuner:
    """
    Handles all the pretraining. Initialize with the base model.
    """
    def __init__(self, model_name, seed) -> None:
        set_seed(seed)
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tuned_model = None
    
    # got from https://discuss.huggingface.co/t/how-to-use-modules-command-to-get-all-the-parameters-that-pertains-to-the-uppermost-layer-of-roberta-large-model/629
    def print_potential_target_modules(self):
        modules = set()
        for name, _ in self.base_model.named_modules():
            splitstr = str(name).split('.')
            if (len(splitstr) > 3):
                mod = splitstr[3:]
                modules.add('.'.join(mod))
        
        for m in modules: print(m)

    #Citation: Got this function template from https://212digital.medium.com/fine-tuning-the-gpt-2-large-language-model-unlocking-its-full-potential-66e3a082ab9c
    def tune(self, train_file_name, val_file_name, model_id, config, eval_steps=500, logging_steps=500, save_steps = 500, epochs = 1, max_steps = -1, seed = 0):
        train_file = os.path.join("../data/", train_file_name)
        val_file = os.path.join("../data/", val_file_name)
        set_seed(seed)
        lora_model = get_peft_model(self.base_model, config)
        
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=128)
        
        val_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=val_file,
            block_size=128)

        print(os.path.join('./logs/', model_id))
        
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
            logging_dir=os.path.join('./logs/', model_id),
            logging_steps=logging_steps,
            max_steps=max_steps,
            eval_accumulation_steps=10, 
            seed = seed,
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True ,
            label_names = ["labels"],
            full_determinism = True
        )
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer, 
            callbacks=[EarlyStoppingCallback(early_stopping_patience = 2)]
        )
        trainer.train()
        self.tuned_model = lora_model
        lora_model.save_pretrained(model_id)

def create_pipeline(checkpoint_path):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    return pipeline('text-generation', model=model, tokenizer = tokenizer)

def get_response(pipeline, prompt, seed):
    set_seed(seed)
    return pipeline(prompt)[0]['generated_text']

def get_perplexity(checkpoints , test_data_file, path, seed):
    perplexity = load("perplexity", module_type="metric")
    avg_perps = []
    test_file = os.path.join("../data/", test_data_file)
    with open(test_file, 'r') as file:
        data = file.read()
    data_list = data.split("\n")
    random.seed(seed)
    random_seeds = []
    for _ in range(len(data_list)):
        random_seeds.append(random.randint(0, 3000000))
    for i in checkpoints:
        _path = path + "/checkpoint-" + str(i)
        pipe = create_pipeline(_path)
        outputs = []
        for j in range(len(data_list)):
            outputs.append(get_response(pipe, data_list[j], random_seeds[j]))
        avg_perps.append((i, perplexity.compute(predictions=outputs, model_id=_path)['mean_perplexity']))
    return avg_perps

        
        





    