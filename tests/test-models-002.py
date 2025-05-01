import json
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

class TechStackEvaluator:
    def __init__(self, model_path: str):
        """
        Load the trained RoBERTa NER model for evaluation on test data.
        
        Args:
            model_path: Path to the folder containing the trained model (final_model).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, add_prefix_space=True)
        self.model = RobertaForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Map labels from id to text
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def load_test_data(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load test data from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing test data.
            
        Returns:
            List of test data.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the data is a list
        if not isinstance(data, list):
            data = [data]  # Convert to list if it's a single item
            
        return data
    
    def predict_single_example(self, tokens: List[str]) -> List[int]:
        """
        Predict NER tags for the given tokens.
        
        Args:
            tokens: List of tokens.
            
        Returns:
            List of predicted label ids.
        """
        # Tokenize using the RoBERTa tokenizer
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the appropriate device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Perform prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert outputs to predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        
        # Align predictions with the original tokens
        word_ids = self.tokenizer(tokens, is_split_into_words=True).word_ids(batch_index=0)
        aligned_predictions = []
        prev_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens like [CLS], [SEP]
                continue
            elif word_idx != prev_word_idx:
                # First token of a word
                if i < len(predictions):
                    aligned_predictions.append(predictions[i])
            prev_word_idx = word_idx
        
        # Trim predictions to match the length of tokens
        aligned_predictions = aligned_predictions[:len(tokens)]
        
        return aligned_predictions
    
    def evaluate_test_data(self, test_data: List[Dict[str, Any]], output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of test data.
            output_dir: Directory to save the results (if provided).
            
        Returns:
            Dict containing evaluation results.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_true_labels = []
        all_pred_labels = []
        all_results = []
        
        for i, example in enumerate(test_data):
            tokens = example.get("tokens", [])
            true_ner_tags = example.get("ner_tags", [])
            
            if not tokens or not true_ner_tags:
                continue
                
            # Predict labels
            pred_ner_tags = self.predict_single_example(tokens)
            
            # Trim predictions to match true labels
            pred_ner_tags = pred_ner_tags[:len(true_ner_tags)]
            
            # Add padding if necessary
            if len(pred_ner_tags) < len(true_ner_tags):
                pred_ner_tags.extend([0] * (len(true_ner_tags) - len(pred_ner_tags)))
                
            # Convert numeric tags to string labels
            true_labels_str = [self.id2label.get(label, "O") for label in true_ner_tags]
            pred_labels_str = [self.id2label.get(label, "O") for label in pred_ner_tags]
            
            # Create matching indicator
            matching = ["✅" if t == p else "❌" for t, p in zip(true_ner_tags, pred_ner_tags)]
            
            # Store results for each example
            result = {
                "id": example.get("id", i),
                "tokens": tokens,
                "true_labels": true_labels_str,
                "pred_labels": pred_labels_str,
                "matching": matching,
                "accuracy": sum(t == p for t, p in zip(true_ner_tags, pred_ner_tags)) / len(true_ner_tags) if true_ner_tags else 0
            }
            all_results.append(result)
            
            # Collect labels for classification report
            all_true_labels.extend(true_ner_tags)
            all_pred_labels.extend(pred_ner_tags)
            
        # Generate classification report
        unique_labels = sorted(set(all_true_labels + all_pred_labels))
        target_names = [self.id2label.get(label, f"Unknown-{label}") for label in unique_labels]
        
        report = classification_report(
            all_true_labels, all_pred_labels,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(
            all_true_labels, all_pred_labels,
            labels=unique_labels
        )
        
        # Convert confusion matrix to DataFrame
        cm_df = pd.DataFrame(
            cm,
            index=target_names,
            columns=target_names
        )
        
        # Create confusion matrix plot
        if output_dir:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            
            # Save classification report
            report_path = os.path.join(output_dir, "classification_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            
            # Save detailed results
            results_path = os.path.join(output_dir, "detailed_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
        
        # Calculate average accuracy
        avg_accuracy = sum(result["accuracy"] for result in all_results) / len(all_results) if all_results else 0
        
        # Summarize results
        summary = {
            "accuracy": avg_accuracy,
            "weighted_f1": report["weighted avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "number_of_examples": len(all_results),
            "number_of_tokens": len(all_true_labels)
        }
        
        if output_dir:
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        
        return {
            "summary": summary,
            "classification_report": report,
            "confusion_matrix": cm_df.to_dict(),
            "detailed_results": all_results
        }
    
    def extract_entities_from_test_data(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract technologies from test data.
        
        Args:
            test_data: List of test data.
            
        Returns:
            List of Dict containing extracted technologies.
        """
        results = []
        
        for i, example in enumerate(test_data):
            tokens = example.get("tokens", [])
            
            if not tokens:
                continue
                
            # Predict labels
            pred_ner_tags = self.predict_single_example(tokens)
            
            # Extract technologies
            entities = {}
            current_entity = []
            current_type = None
            
            for token, tag_id in zip(tokens, pred_ner_tags):
                tag = self.id2label.get(tag_id, "O")
                
                if tag.startswith("B-"):
                    # Save the previous entity (if any)
                    if current_entity and current_type:
                        entity_text = " ".join(current_entity)
                        if current_type not in entities:
                            entities[current_type] = []
                        if entity_text not in entities[current_type]:
                            entities[current_type].append(entity_text)
                    
                    # Start a new entity
                    current_entity = [token]
                    current_type = tag[2:]  # Remove "B-"
                
                elif tag.startswith("I-") and current_entity:
                    # Continue the current entity
                    entity_type = tag[2:]  # Remove "I-"
                    if entity_type == current_type:
                        current_entity.append(token)
                    else:
                        # Mismatched entity type
                        if current_entity and current_type:
                            entity_text = " ".join(current_entity)
                            if current_type not in entities:
                                entities[current_type] = []
                            if entity_text not in entities[current_type]:
                                entities[current_type].append(entity_text)
                        current_entity = [token]
                        current_type = entity_type
                
                elif tag == "O" and current_entity:
                    # Save the previous entity
                    if current_entity and current_type:
                        entity_text = " ".join(current_entity)
                        if current_type not in entities:
                            entities[current_type] = []
                        if entity_text not in entities[current_type]:
                            entities[current_type].append(entity_text)
                    current_entity = []
                    current_type = None
            
            # Save the last entity (if any)
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                if current_type not in entities:
                    entities[current_type] = []
                if entity_text not in entities[current_type]:
                    entities[current_type].append(entity_text)
            
            # Prepare the result
            result = {
                "id": example.get("id", i),
                "text": " ".join(tokens),
                "entities": entities
            }
            results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    # Paths
    MODEL_PATH = "/home/whilebell/Code/Project/TechStack-NER/models/bootstrapping001/final_model"  # Change to your model path
    TEST_DATA_PATH = "/home/whilebell/Code/Project/TechStack-NER/data/interim/bootstrapping/test-001/test_data.json"  # Change to your test JSON file path
    OUTPUT_DIR = "/home/whilebell/Code/Project/TechStack-NER/data/interim/bootstrapping/test-001"  # Directory to save results
    
    # Create evaluator
    evaluator = TechStackEvaluator(MODEL_PATH)

    # Load test data
    test_data = evaluator.load_test_data(TEST_DATA_PATH)
    print(f"Loaded {len(test_data)} test examples")

    # Evaluate the model
    print("Evaluating the model...")
    eval_results = evaluator.evaluate_test_data(test_data, OUTPUT_DIR)

    # Display evaluation summary
    print("\n=== Evaluation Summary ===")
    for key, value in eval_results["summary"].items():
        print(f"{key}: {value}")

    # Extract technologies
    print("\n=== Extracting Technologies from Test Data ===")
    extracted_entities = evaluator.extract_entities_from_test_data(test_data)

    # Display results
    for i, result in enumerate(extracted_entities):
        if i < 5:  # Display only the first 5 examples
            print(f"\nID: {result['id']}")
            print(f"Text: {result['text']}")
            print("Entities:")
            for entity_type, entities in result['entities'].items():
                print(f"  {entity_type}: {', '.join(entities)}")

    # Save extracted technology results
    entities_path = os.path.join(OUTPUT_DIR, "extracted_entities.json")
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(extracted_entities, f, indent=2)
    print(f"\nSaved extracted technology results to {entities_path}")
