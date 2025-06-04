from transformers import RobertaTokenizer

# โหลด RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# ตัวอย่างข้อความ
texts = [
    "Hello world, how are you?",
    "RoBERTa tokenization example",
    "Transformers are amazing for NLP tasks!",
    "Let's see how RoBERTa handles this",
]

print("=== RoBERTa Tokenization ===\n")

for text in texts:
    print(f"ข้อความ: {text}")
    
    # แยกคำด้วย tokenize
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    print(f"จำนวน tokens: {len(tokens)}")
    print("-" * 40)