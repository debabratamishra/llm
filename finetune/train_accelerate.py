# Install if needed
# pip install transformers datasets peft accelerate bitsandbytes

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# Accelerator Init
accelerator = Accelerator()
device = accelerator.device

# Load model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map=None)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("squad", split="train[:1%]") # Training on just 1% of data as an example

# Preprocessing
def format_qa(example):
    prompt = f"Question: {example['question']} Context: {example['context']} Answer:"
    answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ''
    full_text = prompt + " " + answer
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenized["input_ids"].copy()
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(format_qa, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type='torch')

# DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(tokenized_dataset, shuffle=True, batch_size=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)

# Prepare with Accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Training Loop
EPOCHS = 1
MAX_TRAIN_STEPS = 50 

step = 0
model.train()
for epoch in range(EPOCHS):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        if step % 5 == 0:
            accelerator.print(f"Step {step} - Loss: {loss.item():.4f}")
        
        if step >= MAX_TRAIN_STEPS:
            break
    if step >= MAX_TRAIN_STEPS:
        break

# Save
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./finetuned_tinyllama_qa", save_function=accelerator.save)
tokenizer.save_pretrained("./finetuned_tinyllama_qa")

print("Training complete!")