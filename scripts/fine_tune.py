# fine_tune.py
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Load your merged DataFrame
# Make sure to adjust the path to where your merged dataframe is stored
merged_df = pd.read_csv('/Users/anshgupta/Desktop/MindMate/data/merged_df.csv')

# Drop rows where either 'prompt' or 'response' is None
merged_df.dropna(subset=['prompt', 'response'], inplace=True)

# Initialize the tokenizer
model_name = "vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(merged_df)

def preprocess_function(examples):
    # Concatenate prompt and response with EOS token
    return {'text': [prompt + tokenizer.eos_token + response for prompt, response in zip(examples['prompt'], examples['response'])]}

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # As specified, not using fp16
    gradient_accumulation_steps=1,
)

# Additional model configuration for LoRA and 4-bit optimization
model.config.lora_r = 64
model.config.lora_alpha = 16
model.config.lora_dropout = 0.1
model.config.use_4bit = True
model.config.bnb_4bit_compute_dtype = "float16"
model.config.bnb_4bit_quant_type = "nf4"
model.config.use_nested_quant = False

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)
# Start training
trainer.train()
# Save the fine-tuned model
trainer.save_model("/Users/anshgupta/Desktop/MindMate/models")
