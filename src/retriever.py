from sentence_transformers import SentenceTransformer, util 
import torch 
class Retriever:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def check(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)

    def retrieve(self, claim, chunks, top_k=3):
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        scores = util.cos_sim(claim_emb, chunk_embs)[0]
        top_results = torch.topk(scores, k=min(top_k, len(chunks)))

        return [chunks[idx] for idx in top_results.indices]