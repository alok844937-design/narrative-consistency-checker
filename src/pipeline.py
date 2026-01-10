from src.claim_extractor import extract_claims 
from src.chunker import chunk_text
from src.retriever import Retriever 
from src.nli_checker import NLIChecker 

def run_pipeline(novel_text, backstory):
    claims = extract_claims(backstory)
    chunks = chunk_text(novel_text)
    retriever = Retriever()
    nli = NLIChecker() 

    for claim in claims: 
        relevant_chunks = retriever.retrieve(claim, chunks) 
        max_contra = 0 
        maX_entail = 0 

        for chunk in relevant_chunks: 
            res = nli.check(chunk, claim)
            max_contra = max(max_contra, res["contradiction"])
            max_entail = max(max_entail, res["entailment"])

            if max_contra - max_entail > 0.3:
                return 0  # Inconsistent
            return 1    # Consistent
