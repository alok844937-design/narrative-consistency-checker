import re 
import nltk 

nltk.download('punkt')

def extract_claims(backstory: str):
    sentences = nltk.sent_tokenize(backstory)
    claims = [] 

    for sent in sentences: 
        parts = re.split(r'\b(and|but|because|however)\b', sent, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if len(part.split()) > 4:
                claims.append(part)

        return list(set(claims))