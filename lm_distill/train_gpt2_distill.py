import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ⚡ Toggle between subset (fast) and full dataset
subset = True   # change to False for full WikiText-103 training

# 1. Load dataset (WikiText-103)
print("Loading WikiText-103 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_enc = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
test_enc = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])

# ✅ Subset mode (for quick Mac testing)
if subset:
    print("⚡ Using subset mode (50k samples)...")
    train_enc = train_enc.select(range(50000))
    test_enc = test_enc.select(range(1000))

train_enc.set_format("torch", columns=["input_ids", "attention_mask"])
test_enc.set_format("torch", columns=["input_ids", "attention_mask"])

batch_size = 4  # adjust for memory
train_loader = DataLoader(train_enc, batch_size=batch_size, shuffle=True)

# 2. Load teacher and student
print("Loading teacher and student models...")
teacher = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True).to(device)
student = AutoModelForCausalLM.from_pretrained("distilgpt2", output_hidden_states=True).to(device)

teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# 3. Losses
kl_loss = nn.KLDivLoss(reduction="batchmean")
ce_loss = nn.CrossEntropyLoss()
cosine_loss = nn.CosineEmbeddingLoss()

def distill_loss(student_out, teacher_out, labels, T=2.0, alpha=0.5, beta=0.1):
    """
    Total loss = α * CE + (1-α) * KLDiv + β * Cosine(hidden states)
    """
    student_logits = student_out.logits.view(-1, student_out.logits.size(-1))
    teacher_logits = teacher_out.logits.view(-1, teacher_out.logits.size(-1))
    labels = labels.view(-1)

    # KL divergence (soft targets)
    s_log_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
    t_prob = nn.functional.softmax(teacher_logits / T, dim=-1)
    loss_kl = kl_loss(s_log_prob, t_prob) * (T * T)

    # CE loss (hard targets)
    loss_ce = ce_loss(student_logits, labels)

    # Cosine embedding loss (hidden state alignment)
    s_hidden = student_out.hidden_states[-1].mean(dim=1)
    t_hidden = teacher_out.hidden_states[-1].mean(dim=1)
    target = torch.ones(s_hidden.size(0)).to(device)
    loss_cos = cosine_loss(s_hidden, t_hidden, target)

    return alpha * loss_ce + (1 - alpha) * loss_kl + beta * loss_cos

# 4. Optimizer
optimizer = AdamW(student.parameters(), lr=3e-5)

# 5. Training loop with checkpoints
epochs = 2 if subset else 5
save_dir = "./lm_distill/student_distilgpt2_full"
os.makedirs(save_dir, exist_ok=True)

checkpoint_interval = 1000

print(f"Starting GPT2 → DistilGPT2 distillation training ({'subset' if subset else 'full'} mode, {epochs} epochs)...")

global_step = 0
for epoch in range(epochs):
    student.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        student_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = distill_loss(student_out, teacher_out, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % checkpoint_interval == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_path, exist_ok=True)
            student.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"💾 Saved checkpoint at step {global_step} → {ckpt_path}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Avg training loss = {avg_loss:.4f}")

# 6. Save final student
student.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Final Distilled DistilGPT2 model saved at {save_dir}")