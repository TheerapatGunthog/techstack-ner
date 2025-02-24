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
text = "Responsibilities Design and develop highly scalable, reliable, secure, and faulttolerant systems end to end using state of the art technology Work directly with Product and Technology team members to define and implement complex features Collaborate with other team members to learn and share best practices Understand and constantly optimize our products, identifying and fixing problems, improving stability and user experience Take operational responsibility for the services that are owned by your team Debug production issues across services Participate in oncall rotations as needed we support a healthy work life balance, so we invest in minimizing outofoffice interruptions and we use rotations to minimize oncall days Requirements At least 2 years of experience in software development A bachelor s degree in computer science, engineering, mathematics, or a related field or equivalent experience Strong knowledge of one or more programming languages Java, Scala, Kotlin, Groovy, Go, C C++, Rust, Python, C#, etc. and the ability to learn new programming languages quickly Strong understanding of software architecture Understanding of data systems and how to query interact with them RDBMS, NoSQL, Queues, etc."
entities = ner_pipeline(text)

for entity in entities:
    print(
        f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}, "
        f"Start: {entity['start']}, End: {entity['end']}"
    )
