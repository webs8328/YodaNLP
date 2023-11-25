from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline, set_seed
from peft import LoraConfig, get_peft_model

#Citation: Got this function template from https://212digital.medium.com/fine-tuning-the-gpt-2-large-language-model-unlocking-its-full-potential-66e3a082ab9c
def fine_tune_gpt2(model_name, train_file, output_dir, config, save_steps = 500, epochs = 1):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    lora_model = get_peft_model(model, config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=save_steps,
    )
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer= tokenizer
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def create_pipeline(checkpoint_path):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    return pipeline('text-generation', model=model, tokenizer = tokenizer)

def run_model(pipeline, prompt):
    return pipeline(prompt)[0]['generated_text']

    
    
