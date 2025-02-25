from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
)

model = AutoModelForTokenClassification.from_pretrained(
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
)

ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

# Input text for Named Entity Recognition
text = "Familiarity with Excel, Powerpoint, and Google Workspace products."
entities = ner_pipeline(text)

for entity in entities:
    print(
        f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}, "
        f"Start: {entity['start']}, End: {entity['end']}"
    )
