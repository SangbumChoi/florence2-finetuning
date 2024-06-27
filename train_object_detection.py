import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

from data import ObjectDetectionDataset
from peft import LoraConfig, get_peft_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
revision = "refs/pr/6"
model_name = "microsoft/Florence-2-base-ft"
# Load the model and processor
# model = AutoModelForCausalLM.from_pretrained("model/Florence-2-base-ft", trust_remote_code=True).to(device)
# processor = AutoProcessor.from_pretrained("model/Florence-2-base-ft", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=True, device_map="cuda") # load the model on GPU
processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)

IGNORE_ID = -100 # Pytorch ignore index when computing loss
MAX_LENGTH = 512

apply_lora = True

def collate_fn(examples):
    prompt_texts = [example[0] for example in examples]
    label_texts = [example[1] for example in examples]
    images = [example[2] for example in examples]

    inputs = processor(
        images=images,
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
        max_length=MAX_LENGTH,
    ).to(device)

    return inputs, label_texts


# Create datasets
train_dataset = ObjectDetectionDataset("train", processor=processor, name="danelcsb/cavity", class_list=["cavity", "normal"])
val_dataset = ObjectDetectionDataset("test", processor=processor, name="danelcsb/cavity", class_list=["cavity", "normal"])

# Create DataLoader
batch_size = 2
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
)


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, label_texts = batch

            labels = processor.tokenizer(
                label_texts,
                return_tensors="pt",
                padding=True,
                max_length=MAX_LENGTH,
                return_token_type_ids=False, # no need to set this to True since BART does not use token type ids
            )["input_ids"].to(device)

            labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID # do not learn to predict pad tokens during training

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            if i % 25 == 0:
                print(loss)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )

                for generated_text, answer in zip(generated_texts, label_texts):
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task="<OD>",
                        image_size=(
                            inputs["pixel_values"].shape[-2],
                            inputs["pixel_values"].shape[-1],
                        ),
                    )
                    print("GT:", answer)
                    print("Generated Text:", generated_text)
                    print("Pred:", parsed_answer["<OD>"])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, labels = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=labels,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    model = model.half()

if apply_lora:
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision=revision
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-4)

model.push_to_hub("danelcsb/Florence-2-FT-cavity")
processor.push_to_hub("danelcsb/Florence-2-FT-cavity")