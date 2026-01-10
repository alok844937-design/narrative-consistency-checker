import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

class NLIChecker: 
    def __init__(self, model_name="roberta-large-mnli"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        print(f"[INFO] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.labels = self.model.config.id2label 