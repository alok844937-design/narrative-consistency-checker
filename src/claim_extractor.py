"""
Claim extraction module for identifying claims in narratives.
"""
import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ClaimExtractor:
    def __init__(self):
        """Initialize the claim extractor."""
        # Patterns that typically indicate claims
        self.claim_patterns = [
            r'\b(is|are|was|were|will be|has been|have been)\b',
            r'\b(claims?|states?|says?|reports?|indicates?|shows?|proves?)\b',
            r'\b(according to|based on|evidence suggests?)\b',
        ]
        
        # Filtering patterns for non-claims (questions, commands, etc.)
        self.non_claim_patterns = [
            r'\?$',  # Questions
            r'^(what|when|where|who|why|how)\b',  # Question words
        ]
    
    def is_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence is a claim.
        
        Args:
            sentence: Input sentence
            
        Returns:
            True if sentence is likely a claim
        """
        sentence_lower = sentence.lower().strip()
        
        # Check if it's NOT a claim
        for pattern in self.non_claim_patterns:
            if re.search(pattern, sentence_lower):
                return False
        
        # Check if it contains claim indicators
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # If sentence has subject-verb structure and makes a statement
        words = sentence_lower.split()
        if len(words) >= 4:  # Minimum length for a claim
            return True
        
        return False
    
    def extract_claims(self, text: str) -> List[Dict[str, any]]:
        """
        Extract claims from text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries containing claim information
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        claims = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if self.is_claim(sentence):
                claims.append({
                    'claim_id': len(claims),
                    'sentence_id': idx,
                    'text': sentence,
                    'length': len(sentence)
                })
        
        return claims
    
    def extract_claims_from_narrative(self, narrative: str) -> List[str]:
        """
        Extract claim texts from a narrative.
        
        Args:
            narrative: Input narrative text
            
        Returns:
            List of claim texts
        """
        claims = self.extract_claims(narrative)
        return [claim['text'] for claim in claims]
    
    def extract_key_claims(self, text: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Extract the most important claims based on heuristics.
        
        Args:
            text: Input text
            top_k: Number of top claims to return
            
        Returns:
            List of top claim dictionaries
        """
        claims = self.extract_claims(text)
        
        # Score claims based on various factors
        for claim in claims:
            score = 0
            claim_text = claim['text'].lower()
            
            # Longer claims might be more substantial
            score += min(len(claim_text) / 100, 2)
            
            # Claims with specific verbs are often important
            important_verbs = ['claims', 'states', 'proves', 'shows', 'demonstrates', 'reveals']
            for verb in important_verbs:
                if verb in claim_text:
                    score += 2
            
            # Claims with numbers/statistics
            if re.search(r'\d+', claim_text):
                score += 1
            
            # Claims with named entities (simple heuristic: capitalized words)
            capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', claim['text'])
            score += min(len(capitalized_words) * 0.5, 2)
            
            claim['importance_score'] = score
        
        # Sort by score and return top_k
        claims.sort(key=lambda x: x['importance_score'], reverse=True)
        return claims[:top_k]