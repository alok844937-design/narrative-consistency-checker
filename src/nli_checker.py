import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class NLIChecker:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Loading NLI model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.labels = ["contradiction", "neutral", "entailment"]

    def _nli_scores(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)[0]

        return {
            "contradiction": probs[0].item(),
            "neutral": probs[1].item(),
            "entailment": probs[2].item()
        }

    def verify_claim(self, claim_text, context_texts):
        """
        Verifies one claim against multiple retrieved contexts
        """
        context_results = []
        max_entailment = 0.0
        max_contradiction = 0.0
        entailments = []

        for ctx in context_texts:
            scores = self._nli_scores(ctx, claim_text)
            context_results.append({
                "context": ctx,
                "scores": scores
            })

            max_entailment = max(max_entailment, scores["entailment"])
            max_contradiction = max(max_contradiction, scores["contradiction"])
            entailments.append(scores["entailment"])

        avg_entailment = sum(entailments) / len(entailments) if entailments else 0.0

        # Verdict logic
        if max_contradiction > 0.6:
            verdict = "CONTRADICTED"
        elif max_entailment > 0.7:
            verdict = "SUPPORTED"
        elif max_entailment > 0.4:
            verdict = "PARTIALLY_SUPPORTED"
        else:
            verdict = "INSUFFICIENT_EVIDENCE"

        return {
            "verdict": verdict,
            "max_entailment_score": round(max_entailment, 4),
            "max_contradiction_score": round(max_contradiction, 4),
            "avg_entailment_score": round(avg_entailment, 4),
            "context_results": context_results
        }