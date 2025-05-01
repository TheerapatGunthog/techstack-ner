import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import numpy as np
from typing import List, Dict, Tuple

class TechStackNER:
    def __init__(self, model_path: str):
        """
        โหลดโมเดล NER RoBERTa ที่เทรนแล้วเพื่อตรวจจับเทคโนโลยีในข้อความ
        
        Args:
            model_path: พาธไปยังโฟลเดอร์ที่เก็บโมเดลที่เทรนแล้ว (final_model)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # โหลด tokenizer และโมเดล
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, add_prefix_space=True)
        self.model = RobertaForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # กำหนดการแปลง label จาก id เป็นข้อความ
        self.id2label = self.model.config.id2label
        
    def predict(self, text: str) -> List[Dict]:
        """
        ทำนาย NER tags สำหรับข้อความที่กำหนด
        
        Args:
            text: ข้อความที่ต้องการตรวจจับเทคโนโลยี
            
        Returns:
            List ของ Dict ที่มี token, predicted_label และ entity_group
        """
        # แบ่งข้อความเป็น tokens
        tokens = text.split()
        
        # Tokenize ด้วย RoBERTa tokenizer
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True
        )
        
        # เก็บ word_ids เพื่อ align predictions กับ tokens ต้นฉบับ
        word_ids = inputs.word_ids(batch_index=0)
        offset_mapping = inputs.pop("offset_mapping")
        
        # ย้าย inputs ไปที่อุปกรณ์ที่ใช้
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # ทำนาย
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # แปลงผลลัพธ์เป็น predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        
        # จัด predictions ให้ตรงกับ tokens ต้นฉบับ
        results = []
        prev_word_idx = None
        current_entity = None
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # เป็น special token เช่น [CLS], [SEP], etc.
                continue
                
            pred_label_id = predictions[token_idx]
            pred_label = self.id2label[pred_label_id]
            
            # ตรวจสอบว่าเป็น token แรกของคำหรือไม่
            if word_idx != prev_word_idx:
                # เป็น token ใหม่
                entity_type = None
                
                if pred_label != "O":
                    # ตรวจหาชนิดของ entity (หลังจาก B- หรือ I-)
                    entity_type = pred_label[2:]  # เช่น B-PROGRAMMINGLANG -> PROGRAMMINGLANG
                
                results.append({
                    "token": tokens[word_idx],
                    "predicted_label": pred_label,
                    "entity_group": entity_type
                })
                
            prev_word_idx = word_idx
            
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        สกัดเทคโนโลยีต่างๆ จากข้อความและจัดกลุ่มตามประเภท
        
        Args:
            text: ข้อความที่ต้องการสกัดเทคโนโลยี
            
        Returns:
            Dict ที่มีคีย์เป็นประเภทของเทคโนโลยีและค่าเป็น List ของเทคโนโลยีที่พบ
        """
        predictions = self.predict(text)
        
        entities = {}
        current_entity = []
        current_type = None
        
        for item in predictions:
            token = item["token"]
            label = item["predicted_label"]
            entity_type = item["entity_group"]
            
            if label.startswith("B-"):
                # ถ้ามี entity ที่กำลังสะสมอยู่ ให้เพิ่มเข้าไปในผลลัพธ์ก่อน
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type not in entities:
                        entities[current_type] = []
                    if entity_text not in entities[current_type]:
                        entities[current_type].append(entity_text)
                
                # เริ่ม entity ใหม่
                current_entity = [token]
                current_type = entity_type
            
            elif label.startswith("I-") and current_entity and entity_type == current_type:
                # ต่อ entity ที่กำลังสะสมอยู่
                current_entity.append(token)
            
            else:
                # "O" หรือ entity type ไม่ตรงกับ current_type
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type not in entities:
                        entities[current_type] = []
                    if entity_text not in entities[current_type]:
                        entities[current_type].append(entity_text)
                
                current_entity = []
                current_type = None
        
        # เพิ่ม entity สุดท้าย (ถ้ามี)
        if current_entity and current_type:
            entity_text = " ".join(current_entity)
            if current_type not in entities:
                entities[current_type] = []
            if entity_text not in entities[current_type]:
                entities[current_type].append(entity_text)
        
        return entities

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # เปลี่ยนพาธเป็นที่อยู่ของโมเดลที่เทรนแล้ว
    MODEL_PATH = "/home/whilebell/Code/Project/TechStack-NER/models/bootstrapping001/final_model"  # เปลี่ยนเป็นพาธที่คุณบันทึกโมเดล
    
    ner = TechStackNER(MODEL_PATH)
    
    # ตัวอย่างการใช้งาน
    sample_text = "We are building a web application using Python and Django, with React for the frontend. The data is stored in MongoDB and PostgreSQL. The application is deployed on AWS using Docker containers."
    
    # แสดงผลลัพธ์แบบละเอียด (token-by-token)
    print("=== Detailed NER Results ===")
    results = ner.predict(sample_text)
    for item in results:
        print(f"{item['token']} -> {item['predicted_label']}")
    
    # แสดงผลลัพธ์แบบจัดกลุ่มตามประเภทเทคโนโลยี
    print("\n=== Extracted Technology Stack ===")
    entities = ner.extract_entities(sample_text)
    for entity_type, items in entities.items():
        print(f"{entity_type}: {', '.join(items)}")