from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

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
text = "Experience with software development processes for Agile Scrum and CI CD utilizing technologies such as GIT Jenkins TerraformExperience with objectoriented functional script languages such as Python Scala C++ etc.Experience with BI client tools such as Tableau and Power BI and platform #sanfrancisco #sanjose #losangeles #sandiego #oakland #denver #miami #orlando #atlanta #chicago #boston #detroit #newyork #portland #philadelphia #dallas #houston #austin #seattle #sydney #melbourne #perth #toronto #vancouver #montreal #shanghai #beijing #shenzhen #prague"
entities = ner_pipeline(text)

for entity in entities:
    print(
        f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}"
    )
