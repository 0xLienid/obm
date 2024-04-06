import os
import datetime as dt
import torch
import tiktoken
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from model import ModelArgs, Transformer
from training_utils import count_parameters, tokenize_dataset

load_dotenv()

# Training Run Config
model_name = "obm"
run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
epochs = 1
accumulation_steps = 32
batch_size = 1
lr = 3e-4
min_lr = 3e-5
eval_steps = 250
training_steps = 10000
weight_decay = 0.01
lr_decay_steps = training_steps // accumulation_steps
grad_clip = 1.0
dataset_name = "roneneldan/TinyStories"
wandb_project = "obm-experiments"
device = "cuda"
dtype = "bfloat16"

model_args = ModelArgs(
    dim=512,
    n_layers=32,
    n_regions=8,
    n_heads=32,
    vocab_size=50257,
    hidden_dim=1024,
    multiple_of=256,
    norm_eps=1e-5,
    max_seq_len=512,
    dropout=0.0
)

# Setup
os.makedirs(f"runs/{model_name}/{run_id}", exist_ok=True)

# Create model
model = Transformer(model_args)
total_params = count_parameters(model)
print(f"Total Parameters: {total_params}")
model.to(device)

# Create tokenizer
tokenizer = tiktoken.get_encoding("r50k_base")

# Load dataset
print("loading dataset...")
dataset = load_dataset(dataset_name)
train_batches = tokenize_dataset(
    dataset["train"], tokenizer, model_args.max_seq_len, batch_size, 10000)
eval_batches = tokenize_dataset(
    dataset["validation"], tokenizer, model_args.max_seq_len, batch_size, 50)

# Initialize wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(project=wandb_project, name=model_name + "--" + run_id)
print("wandb run initialized")

# Initialize optimizer and scheduler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, min_lr)
print("optimizer and scheduler initialized")

# Training loop
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch in train_batches:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        attn_masks = batch["attn_masks"].to(device)

        model(inputs, targets=targets, attn_mask=attn_masks)
        base_loss = model.last_base_loss
        total_loss = model.last_total_loss
        blocks_used = model.last_regions_used

        del inputs, targets

        total_loss = total_loss / accumulation_steps
        scaler.scale(total_loss).backward()
        epoch_loss += total_loss.item()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.zero_grad(set_to_none=True)

            print(
                f"Step: {global_step + 1}, Loss: {total_loss.item() * accumulation_steps}, Blocks used: {blocks_used}")
            try:
                run.log({"base_loss": base_loss.item(), "total_loss": total_loss.item(
                ) * accumulation_steps, "grad_norm": total_norm, "blocks_used": blocks_used})
            except:
                print(f"Failed to push training data to wandb")

        del total_loss, base_loss, blocks_used

        if (global_step + 1) % eval_steps == 0:
            model.eval()
            eval_run_base_loss = 0.0
            eval_run_total_loss = 0.0
            for i, eval_batch in enumerate(eval_batches):
                eval_inputs = eval_batch["inputs"].to(device)
                eval_targets = eval_batch["targets"].to(device)
                eval_attn_masks = eval_batch["attn_masks"].to(device)

                model(eval_inputs, targets=eval_targets,
                      attn_mask=eval_attn_masks)
                eval_base_loss = model.last_base_loss
                eval_total_loss = model.last_total_loss

                if i == 0:
                    gen_outputs = model.generate(
                        eval_inputs[0][:10].unsqueeze(0), max_new_tokens=20)
                    print(
                        f"Generated: {tokenizer.decode(gen_outputs[0].tolist())}")
                    del gen_outputs

                del eval_inputs, eval_targets

                eval_run_base_loss += eval_base_loss.item()
                eval_run_total_loss += eval_total_loss.item()
                del eval_base_loss, eval_total_loss

            eval_run_base_loss /= len(eval_batches)
            eval_run_total_loss /= len(eval_batches)
            print(f"Step: {global_step + 1}, Eval Loss: {eval_run_total_loss}")
            try:
                run.log({"eval_base_loss": eval_run_base_loss,
                        "eval_total_loss": eval_run_total_loss})
            except:
                print(f"Failed to push eval data to wandb")
            model.train()

        torch.cuda.empty_cache()

        global_step += 1

        if global_step >= training_steps:
            break

    epoch_loss /= len(train_batches)
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

    epoch += 1

# Save model
torch.save(model.state_dict(), f"runs/{model_name}/{run_id}/model.pt")
