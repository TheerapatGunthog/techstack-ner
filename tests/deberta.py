from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", grouped_entities=True)
ner("We are building IoT devices with C++ and FreeRTOS, using AWS IoT Core for cloud connectivity and InfluxDB for time-series data storage.")