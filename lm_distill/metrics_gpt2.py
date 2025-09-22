import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Paths
base_dir = os.path.dirname(__file__)
teacher_path = os.path.join(base_dir, "teacher_gpt2")
student_path = os.path.join(base_dir, "student_distilgpt2")

# Load models & tokenizers
print("Loading teacher and student models...")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path).to(device)

student_tokenizer = AutoTokenizer.from_pretrained(student_path)
student_model = AutoModelForCausalLM.from_pretrained(student_path).to(device)

# Example text prompts
prompts = [
    "The future of artificial intelligence is",
    "Climate change is one of the most",
    "In the world of science fiction, humans and robots"
]

# Function to compute perplexity
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

# Evaluate models
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def evaluate_model(model, tokenizer, name="Model"):
    all_preds, all_refs = [], []
    ppl_scores = []

    for text in prompts:
        # Perplexity
        ppl = compute_perplexity(model, tokenizer, text)
        ppl_scores.append(ppl)

        # Generate continuation
        inputs = tokenizer(text, return_tensors="pt").to(device)
        start = time.time()
        output = model.generate(**inputs, max_length=30, num_return_sequences=1)
        end = time.time()

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        ref = text  # treating input as reference for simplicity

        all_preds.append(pred)
        all_refs.append([ref])

        print(f"\n{name} generated: {pred}")
        print(f"{name} perplexity: {ppl:.2f}, Inference time: {(end-start)*1000:.2f} ms")

    # BLEU & ROUGE
    bleu_score = bleu.compute(predictions=all_preds, references=all_refs)
    rouge_score = rouge.compute(predictions=all_preds, references=[r[0] for r in all_refs])

    return sum(ppl_scores) / len(ppl_scores), bleu_score, rouge_score

# Evaluate Teacher
print("\n📊 Evaluating Teacher GPT-2")
teacher_ppl, teacher_bleu, teacher_rouge = evaluate_model(teacher_model, teacher_tokenizer, "Teacher")

# Evaluate Student
print("\n📊 Evaluating Student DistilGPT-2")
student_ppl, student_bleu, student_rouge = evaluate_model(student_model, student_tokenizer, "Student")

# Print summary
print("\n================ Evaluation Summary ================")
print(f"Teacher GPT-2    | PPL={teacher_ppl:.2f} | BLEU={teacher_bleu['bleu']:.4f} | ROUGE-L={teacher_rouge['rougeL']:.4f}")
print(f"Student DistilGPT-2 | PPL={student_ppl:.2f} | BLEU={student_bleu['bleu']:.4f} | ROUGE-L={student_rouge['rougeL']:.4f}")
print("===================================================")