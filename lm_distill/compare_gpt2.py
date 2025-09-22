import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def model_size(path):
    size = 0
    for root, _, files in os.walk(path):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return size / (1024 * 1024)  # MB

def measure_inference(model, tokenizer, text="The movie was"):
    inputs = tokenizer(text, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_length=20)
    return (time.time() - start) * 1000  # ms

# Paths
teacher_path = "./lm_distill/teacher_gpt2"
student_path = "./lm_distill/student_distilgpt2"

# Load teacher locally
print("Loading teacher GPT-2...")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path)

# Load distilled student
print("Loading distilled student...")
student_tokenizer = AutoTokenizer.from_pretrained(student_path)
student_model = AutoModelForCausalLM.from_pretrained(student_path)

# Compare size
print(f"Teacher size: {model_size(teacher_path):.2f} MB")
print(f"Student size: {model_size(student_path):.2f} MB")

# Compare inference time
teacher_time = measure_inference(teacher_model, teacher_tokenizer)
student_time = measure_inference(student_model, student_tokenizer)

print(f"Teacher inference time: {teacher_time:.2f} ms")
print(f"Student inference time: {student_time:.2f} ms")