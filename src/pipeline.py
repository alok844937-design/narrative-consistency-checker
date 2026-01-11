"""
Main pipeline for narrative consistency checking.
"""
import json
import os
import sys
from typing import Dict, List, Any

# Handle imports for both package and direct execution
try:
    from .chunker import TextChunker
    from .claim_extractor import ClaimExtractor
    from .retriever import Retriever
    from .nli_checker import NLIChecker
except ImportError:
    from chunker import TextChunker
    from claim_extractor import ClaimExtractor
    from retriever import Retriever
    from nli_checker import NLIChecker

class NarrativeConsistencyPipeline:
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 retriever_model: str = 'all-MiniLM-L6-v2',
                 nli_model: str = 'facebook/bart-large-mnli',
                 top_k_retrieval: int = 3):
        """
        Initialize the narrative consistency checking pipeline.
        
        Args:
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            retriever_model: Model name for retriever
            nli_model: Model name for NLI
            top_k_retrieval: Number of contexts to retrieve per claim
        """
        print("Initializing Narrative Consistency Pipeline...")
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.claim_extractor = ClaimExtractor()
        self.retriever = Retriever(model_name=retriever_model)
        self.nli_checker = NLIChecker(model_name=nli_model)
        self.top_k = top_k_retrieval
        
        # Storage for intermediate results
        self.reference_chunks = None
        self.narrative_claims = None
        
        print("Pipeline initialized successfully!")
    
    def process_reference_document(self, reference_text: str) -> List[Dict[str, Any]]:
        """
        Process the reference document by chunking and indexing.
        
        Args:
            reference_text: The reference/ground truth document
            
        Returns:
            List of chunks
        """
        print("\n=== Processing Reference Document ===")
        
        # Chunk the reference document
        print("Chunking reference document...")
        chunks = self.chunker.chunk_text(reference_text, method='sentences')
        print(f"Created {len(chunks)} chunks")
        
        # Build retrieval index
        print("Building retrieval index...")
        self.retriever.build_index(chunks)
        
        self.reference_chunks = chunks
        return chunks
    
    def extract_narrative_claims(self, narrative_text: str) -> List[Dict[str, Any]]:
        """
        Extract claims from the narrative to be checked.
        
        Args:
            narrative_text: The narrative text to check
            
        Returns:
            List of extracted claims
        """
        print("\n=== Extracting Claims from Narrative ===")
        
        claims = self.claim_extractor.extract_claims(narrative_text)
        print(f"Extracted {len(claims)} claims from narrative")
        
        self.narrative_claims = claims
        return claims
    
    def check_consistency(self, 
                         narrative_text: str, 
                         reference_text: str,
                         detailed: bool = True) -> Dict[str, Any]:
        """
        Check consistency between narrative and reference document.
        
        Args:
            narrative_text: The narrative to check
            reference_text: The reference/ground truth document
            detailed: Whether to include detailed results
            
        Returns:
            Consistency checking results
        """
        # Step 1: Process reference document
        reference_chunks = self.process_reference_document(reference_text)
        
        # Step 2: Extract claims from narrative
        narrative_claims = self.extract_narrative_claims(narrative_text)
        
        if not narrative_claims:
            return {
                'status': 'error',
                'message': 'No claims found in narrative',
                'consistency_score': 0.0
            }
        
        # Step 3: Retrieve relevant contexts for each claim
        print("\n=== Retrieving Relevant Contexts ===")
        claim_texts = [claim['text'] for claim in narrative_claims]
        retrieved_contexts = self.retriever.batch_retrieve(claim_texts, top_k=self.top_k)
        
        # Step 4: Verify each claim using NLI
        print("\n=== Verifying Claims ===")
        verification_results = []
        
        for claim, contexts in zip(narrative_claims, retrieved_contexts):
            context_texts = [ctx['text'] for ctx in contexts]
            verification = self.nli_checker.verify_claim(claim['text'], context_texts)
            
            verification_results.append({
                'claim': claim['text'],
                'claim_id': claim['claim_id'],
                'verdict': verification['verdict'],
                'max_entailment_score': verification['max_entailment_score'],
                'max_contradiction_score': verification['max_contradiction_score'],
                'avg_entailment_score': verification['avg_entailment_score'],
                'retrieved_contexts': contexts if detailed else None,
                'context_results': verification['context_results'] if detailed else None
            })
        
        # Step 5: Calculate overall consistency metrics
        print("\n=== Calculating Overall Metrics ===")
        total_claims = len(verification_results)
        supported = sum(1 for r in verification_results if r['verdict'] == 'SUPPORTED')
        contradicted = sum(1 for r in verification_results if r['verdict'] == 'CONTRADICTED')
        partial = sum(1 for r in verification_results if r['verdict'] == 'PARTIALLY_SUPPORTED')
        insufficient = sum(1 for r in verification_results if r['verdict'] == 'INSUFFICIENT_EVIDENCE')
        
        consistency_score = (supported + 0.5 * partial) / total_claims if total_claims > 0 else 0
        
        # Prepare final results
        results = {
            'status': 'success',
            'summary': {
                'total_claims': total_claims,
                'supported': supported,
                'contradicted': contradicted,
                'partially_supported': partial,
                'insufficient_evidence': insufficient,
                'consistency_score': round(consistency_score, 4),
                'consistency_percentage': round(consistency_score * 100, 2)
            },
            'claim_verifications': verification_results
        }
        
        print(f"\nConsistency Score: {results['summary']['consistency_percentage']}%")
        print(f"Supported: {supported}/{total_claims}")
        print(f"Contradicted: {contradicted}/{total_claims}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save results to a JSON file.
        
        Args:
            results: Results dictionary
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report.
        
        Args:
            results: Results dictionary
            
        Returns:
            Report string
        """
        if results['status'] != 'success':
            return f"Error: {results.get('message', 'Unknown error')}"
        
        summary = results['summary']
        
        report = f"""
{'='*60}
NARRATIVE CONSISTENCY REPORT
{'='*60}

OVERALL METRICS:
- Total Claims Analyzed: {summary['total_claims']}
- Consistency Score: {summary['consistency_percentage']}%
- Supported Claims: {summary['supported']} ({summary['supported']/summary['total_claims']*100:.1f}%)
- Contradicted Claims: {summary['contradicted']} ({summary['contradicted']/summary['total_claims']*100:.1f}%)
- Partially Supported: {summary['partially_supported']} ({summary['partially_supported']/summary['total_claims']*100:.1f}%)
- Insufficient Evidence: {summary['insufficient_evidence']} ({summary['insufficient_evidence']/summary['total_claims']*100:.1f}%)

{'='*60}
DETAILED CLAIM ANALYSIS:
{'='*60}
"""
        
        for i, claim_result in enumerate(results['claim_verifications'], 1):
            verdict_symbol = {
                'SUPPORTED': '✓',
                'CONTRADICTED': '✗',
                'PARTIALLY_SUPPORTED': '~',
                'INSUFFICIENT_EVIDENCE': '?'
            }.get(claim_result['verdict'], '?')
            
            report += f"""
Claim {i}: {verdict_symbol} {claim_result['verdict']}
Text: "{claim_result['claim']}"
Scores:
  - Entailment: {claim_result['max_entailment_score']:.3f}
  - Contradiction: {claim_result['max_contradiction_score']:.3f}
  - Average Support: {claim_result['avg_entailment_score']:.3f}
{'-'*60}
"""
        
        return report