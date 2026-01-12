# Narrative Consistency Checker

**IIT Kharagpur Data Science Hackathon 2026**

A comprehensive pipeline for checking the consistency of narratives against reference documents using Natural Language Inference (NLI) and semantic retrieval.

## 📋 Overview

This system analyzes narratives to extract claims and verifies their consistency against a reference/ground truth document. It uses state-of-the-art NLP techniques including:

- **Text Chunking**: Intelligently splits documents into manageable pieces
- **Claim Extraction**: Identifies factual claims from narratives
- **Semantic Retrieval**: Finds relevant context using sentence embeddings
- **NLI Verification**: Verifies claims using Natural Language Inference

## 🏗️ Project Structure

```
narrative-consistency-checker/
│
├── data/
│   └── sample_input.csv                          # Input data file
│
├── src/
│   ├── __init__.py
│   ├── chunker.py                                # Text Chunking module
│   ├── claim_extractor.py                        # Claim extraction module
│   ├── retriever.py                              # Semantic retrieval module 
│   ├── nli_checker.py                            # NLI verification module
│   ├── pipeline.py                               # Main pipeline orchestration 
│   └── optimized_pipeline.py 
│
├── output/                                       # Output directory (auto created)
├── cache/                     
│
├── requirements.txt                              # Python dependencies
├── run.py                                        # Main execution script 
├── run_optimized.py
├── setup.sh
├── setup.bat
├── README.md                                     # This file
├── OPTIMIZATION_GUIDE.md
├── PERFORMANCE_COMPARISON.md
└── QUICK_START.md
```

## Project Repository 

The full project, icluding tthe presentation slides, is available at:
[GitHub Repository](https://github.com/alok844937-design/narrative-consistency-checker)

## 🚀 Installation

### 1. Clone the repository or navigate to project directory

```bash
cd narrative-consistency-checker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (will auto-download on first run, but you can do it manually)

```python
import nltk
nltk.download('punkt')
```

## 📊 Input Format

The input CSV file should have at least two columns:
- **narrative**: The narrative text to check
- **reference**: The reference/ground truth document

Example `data/sample_input.csv`:

```csv
narrative,reference
"The company reported a 25% increase in revenue. The CEO announced expansion plans to Asia.","The company's Q4 results showed a 25% year-over-year revenue growth. The board approved a strategic expansion into Asian markets, scheduled for Q2 next year."
```

## 💻 Usage

### Basic Usage

```bash
python run.py
```

### Advanced Usage

```bash
python run.py \
    --input data/sample_input.csv \
    --output output/results.json \
    --report output/report.txt \
    --chunk-size 512 \
    --top-k 3 \
    --detailed
```

### Command Line Arguments

- `--input`: Path to input CSV file (default: `data/sample_input.csv`)
- `--output`: Path to output JSON file (default: `output/results.json`)
- `--report`: Path to output report file (default: `output/report.txt`)
- `--chunk-size`: Size of text chunks in characters (default: 512)
- `--top-k`: Number of contexts to retrieve per claim (default: 3)
- `--detailed`: Include detailed context and NLI results in output

## 📈 Output

### JSON Output

The system generates a detailed JSON file with:

```json
{
  "status": "success",
  "summary": {
    "total_claims": 10,
    "supported": 7,
    "contradicted": 1,
    "partially_supported": 2,
    "insufficient_evidence": 0,
    "consistency_score": 0.8,
    "consistency_percentage": 80.0
  },
  "claim_verifications": [
    {
      "claim": "The company reported a 25% increase in revenue.",
      "verdict": "SUPPORTED",
      "max_entailment_score": 0.95,
      "max_contradiction_score": 0.02,
      "avg_entailment_score": 0.87
    }
  ]
}
```

### Text Report

A human-readable report with:
- Overall consistency metrics
- Per-claim verification results
- Confidence scores

## 🔧 Module Details

### 1. Chunker (`src/chunker.py`)

Splits large documents into smaller, overlapping chunks for efficient processing.

**Features:**
- Sentence-based chunking
- Paragraph-based chunking
- Configurable overlap

### 2. Claim Extractor (`src/claim_extractor.py`)

Extracts factual claims from narrative text using linguistic patterns.

**Features:**
- Identifies assertive statements
- Filters out questions and non-claims
- Ranks claims by importance

### 3. Retriever (`src/retriever.py`)

Finds relevant context passages for each claim using semantic similarity.

**Features:**
- Uses sentence transformers for embeddings
- FAISS indexing for fast retrieval
- Batch processing support

### 4. NLI Checker (`src/nli_checker.py`)

Verifies claims against context using Natural Language Inference.

**Features:**
- Pre-trained BART-MNLI model
- Three-way classification (entailment, neutral, contradiction)
- Confidence scoring

### 5. Pipeline (`src/pipeline.py`)

Orchestrates the entire workflow from input to output.

**Features:**
- End-to-end processing
- Result aggregation
- Report generation

## 🎯 How It Works

1. **Load Data**: Read narrative and reference texts from CSV
2. **Chunk Reference**: Split reference document into searchable chunks
3. **Extract Claims**: Identify claims in the narrative
4. **Retrieve Context**: Find relevant chunks for each claim
5. **Verify Claims**: Use NLI to check consistency
6. **Generate Report**: Create summary and detailed results

## 📝 Customization

### Changing Models

Edit the model names in `run.py` or pass them to the pipeline:

```python
pipeline = NarrativeConsistencyPipeline(
    retriever_model='all-mpnet-base-v2',  # Different sentence transformer
    nli_model='microsoft/deberta-v3-large-mnli'  # Different NLI model
)
```

### Adjusting Thresholds

Modify thresholds in `src/nli_checker.py`:

```python
if max_entailment > 0.7:  # Adjust this threshold
    verdict = 'SUPPORTED'
```

## 🐛 Troubleshooting

### Out of Memory

- Reduce `chunk_size`
- Process fewer claims at once
- Use CPU instead of GPU (automatic fallback)

### Slow Processing

- Reduce `top_k` retrieval
- Use smaller models
- Process in batches

### Poor Results

- Increase `chunk_size` for more context
- Increase `top_k` for more evidence
- Try different NLI models

## 📚 Dependencies

- `pandas`: Data handling
- `numpy`: Numerical operations
- `transformers`: NLI models
- `sentence-transformers`: Semantic embeddings
- `torch`: Deep learning backend
- `faiss-cpu`: Fast similarity search
- `nltk`: Text processing

## 🎓 For Hackathon Submission

Make sure to:
1. Include sample input data in `data/sample_input.csv`
2. Test with your specific dataset format
3. Adjust column names in `run.py` if needed
4. Document any custom modifications
5. Include output examples in your submission


## 📧 Support

For hackathon-specific questions, contact the organizers.

## 🏆 Good Luck!

Best wishes for the IIT Kharagpur Data Science Hackathon 2026!

---

**Note**: This system requires ~2-3GB of disk space for models and ~4GB RAM for processing. First run will download models automatically.