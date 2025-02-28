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
text = "Rigor and organization skills are required, whilst working to tight deadlines A commitment to quality and a thorough approach to the work Willing to go the extra mile Banking knowledge is a plus."
entities = ner_pipeline(text)

for entity in entities:
    print(
        f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}, "
        f"Start: {entity['start']}, End: {entity['end']}"
    )
