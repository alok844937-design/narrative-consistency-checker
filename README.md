# Narrative Consistency Checker 
This repository presents a long-context NLP system developed for the **Kharagpur Data Science Hackathon 2026**.
This task is to verify whether a hypothetical **character backstory** is logically consistent with a **full-length narraative (novel)** by detecting factual and causal contradictions.

## 🧠Problem Statement
Given: 
- A long narrative text (100k+ words)
- A hypothetical backstory for a central character

Goal: 
- Classify whether the backstory is **consistent** or **inconsistent** with the narrative.
  or
This problem requires **long-context reasoning**, not surface-level text classification.

## 🔍Key Challenges
- Extremely long input documents
- Implicit character facts and events
- Logical and casual contradictions
- Evidence localization within narratives

## 🚀Proposed Approach 
We address the problem using an **evidence-aware, claim-level verification pipeline**:
### 1. Claim-level Decomposition 
The backstory is decomposed into atomic semantic claims using rule-based sentence and conjuction splitting.

### 2. Evidence Retrieval 
Each claim is embedded independently and matched against chunked narrative text to retrieve the most relevant evidence.

### 3. Contradiction Mining via NLI 
A pretrained Natural Language Inference (NLI) model is used to detect entailment and contradiction between each claim and retrieved evidence.

### 4. Conservative Decision Strategy 
If any claim exhibits a high-confidence contradiction with the narrative, the backstory is classified as **inconsistent**.

This conservative strategy minimizes false positives for consistency. 

## 🧩Architecture
Backstory → Claim Extraction → Claim-wise Retrieval → NLI-based Verification → Confidence-Aware Aggregation → Final Consistency Decision

## Installation 
```bash
pip install -r requirements.txt
```
<br>
## Author
Alok
IIT Patna
