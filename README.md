# Adversarial-Robust Intrusion Detector with AI Triage

An end-to-end Network Intrusion Detection System (NIDS) built on the NSL-KDD benchmark, stress-tested with a black-box adversarial evasion attack, hardened via adversarial retraining under a strict zero-leakage protocol, and coupled with an LLM-assisted incident triage layer that explains flagged threats in plain English.

---

## The Problem
Most student and tutorial intrusion detection systems stop at *"trained a classifier, achieved X% accuracy on test data."* In real security operations, two critical challenges make simple accuracy misleading:
1. **Adversarial Evasion**: Attackers actively modify controllable connection properties to sneak malicious traffic past static machine learning boundaries.
2. **Alert Fatigue**: Security Operations Center (SOC) analysts are overwhelmed by raw binary alerts without contextual explanation or threat categorization.

This project addresses both: we build an initial classifier, design an evasion attack against it, measure how easily it is fooled on completely unseen attacks, harden it with adversarial retraining, and translate every detected alert into a structured, plain-English incident note.

---

## Architecture & Zero-Leakage Evaluation Protocol

```text
NSL-KDD Dataset (KDDTrain+, KDDTest+ / 41 Connection Features)
                         │
        preprocess.py (One-hot encoding, column schema alignment)
                         │
        Baseline Random Forest Classifier (200 trees)
                         │
        ┌────────────────┴────────────────────────┐
        │ Test Set Split (50% / 50%)              │
        ▼                                         ▼
Adversarial Crafting Split (X_atk)       Held-Out Test Set (X_holdout)
(Used solely to generate evasion         (Strictly unseen; never touched
 attacks to augment training)             during training or hardening)
        │                                         │
153 Evasion Samples Generated                     │
        │                                         │
Adversarial Retraining                            │
(Augment X_train with evasions)                   │
        │                                         │
Hardened Model (200 trees)                        │
        │                                         │
        └────────────────┬────────────────────────┘
                         ▼
        Zero-Leakage Evaluation on X_holdout
        - Baseline Evasion Rate:  56.67% (170/300 evaded)
        - Hardened Evasion Rate:  16.00% (48/300 evaded)
        - Vulnerability Drop:     40.67% reduction
        - Clean Test Recall:      67.69% preserved
                         │
        Flagged Intrusions (Verdict == ATTACK)
                         │
        LLM Incident Triage Layer
        (Attack Type, MITRE ATT&CK, Severity, Concise Analyst Summary)
                         │
        Interactive Gradio Web Application (app.py)
```

---

## Real Measured Experimental Results (Leak-Free Protocol)

All numbers below were empirically measured on the held-out partition (`X_holdout`, 11,272 connections) of the official NSL-KDD test set (`KDDTest+.txt`):

### 1. Classification Performance

| Metric | Baseline Classifier | Adversarially Hardened Model | Delta |
|---|---|---|---|
| **Clean Test Accuracy** | 77.32% | **80.59%** | `+3.26%` |
| **Clean Test Precision** | 97.19% | **97.42%** | `+0.23%` |
| **Attack Catch Rate (Recall)** | 61.95% | **67.69%** | `+5.74%` |
| **Clean Test F1-Score** | 75.67% | **79.88%** | `+4.21%` |

### 2. Adversarial Evasion Attack & Hardening (Strictly Unseen Attacks)

* **Threat Model**: Black-box greedy random search. The adversary observes only model output (`normal` vs `attack`) and does not know tree splits, weights, or hyperparameters.
* **Attacker-Controllable Features (5)**: `["duration", "src_bytes", "dst_bytes", "count", "srv_count"]` (features an adversary can plausibly tune during connection establishment).
* **Sample Size**: 300 unseen test attacks sampled from the held-out partition (`X_holdout`).

| Model Variant | Evasion Success Rate | Evaded Connections | Detection Retention |
|---|---|---|---|
| **Baseline Model** | **56.67%** | 170 / 300 | 43.33% |
| **Hardened Model** | **16.00%** | 48 / 300 | **84.00%** |

### Key Takeaways
1. **Severe Baseline Fragility**: Simply perturbing 5 connection features allowed **56.67%** of unseen baseline-detected attacks to slip past the detector as "normal" traffic.
2. **Generalizable Hardening**: Adding successful evasion examples from the attack split back into the training distribution reduced the evasion success rate on *completely unseen* attacks from **56.67% down to 16.00%** (an absolute vulnerability reduction of **40.67%**).
3. **Leakage-Free Validation**: Because `X_holdout` was strictly sequestered from the adversarial crafting split (`X_atk`), the observed robustness gains prove real generalization against adversarial manipulation rather than memorization of perturbed test rows.

---

## LLM Triage Layer

Whenever traffic is flagged as an attack, the triage layer (`triage.py`) analyzes connection metrics and generates a structured incident summary:
* **Likely Attack Type**: e.g. *SYN Flood / Denial of Service*, *Port Scan / Reconnaissance*, *Smurf Amplification*.
* **MITRE ATT&CK Mapping**: e.g. `T1498` (*Network Denial of Service*), `T1046` (*Network Service Discovery*).
* **Severity**: `LOW`, `MEDIUM`, or `HIGH`.
* **Analyst Rationale**: 1-2 concise sentences explaining the primary telemetry anomalies (e.g. *Abnormally high connection count (250) with high SYN error rate (100%) targeting HTTP*).

Supports Groq, Google Gemini, Anthropic, or OpenAI REST APIs via `.env`, with zero heavy SDK dependencies.

> **Design Note (Cost & Latency Optimization)**: The LLM triage engine is invoked exclusively on connections flagged as `[ATTACK]`. Benign connections (`[NORMAL]`) are verified without dispatching an API call, preventing token waste, minimizing latency, and keeping pipeline throughput scalable.

---

## Project Structure

```text
ids-adversarial-triage/
├── data/                       # Local cache for KDDTrain+.txt & KDDTest+.txt (git-ignored)
├── preprocess.py               # Canonical schema & encoding (single source of truth)
├── train.py                    # Zero-leakage training, evasion, and hardening pipeline
├── triage.py                   # LLM incident triage engine (direct REST via requests)
├── app.py                      # Interactive Gradio web application
├── sample_test.csv             # Curated sample network traffic for instant testing
├── baseline_model.joblib       # Saved baseline Random Forest model
├── model.joblib                # Saved hardened Random Forest model
├── feature_columns.joblib      # Serialized feature column schema
├── results.json                # Raw measured benchmark metrics
├── results.md                  # Human-readable experimental report
├── requirements.txt            # Project dependencies
├── .env.example                # API key template
├── .gitignore                  # Git ignore rules
└── notes.md                    # Private interview prep & resume bullet points
```

---

## Quickstart & Installation

### 1. Setup Environment
```bash
git clone https://github.com/sanyambedi/adversarial-nids-ai-triage.git
cd adversarial-nids-ai-triage
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure LLM API Key (Optional for Triage)
Copy the example environment file and insert your API key:
```bash
cp .env.example .env
```
Supported keys: `GROQ_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`. (If no key is configured, an automated heuristic analyst note is generated for offline local testing).

### 3. Run Pipeline or Launch Web App
To re-run the entire leak-free training, evasion testing, and hardening pipeline:
```bash
python train.py
```

To launch the interactive Gradio dashboard:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser. Click **Test Sample Data (sample_test.csv)** for an instant demo or upload any NSL-KDD formatted CSV.

---

## Limitations
* **Dataset Age**: NSL-KDD is derived from 1999 simulated network traffic. It does not reflect modern protocol stacks, cloud VPC flow logs, or encrypted HTTPS/TLS payload dynamics. It is used here strictly as a reproducible benchmark for comparative adversarial robustness.
* **Threat Model Scope**: The evasion search assumes an attacker controls connection duration, byte volumes, and connection rates (`duration, src_bytes, dst_bytes, count, srv_count`). Real-world attacker capabilities depend on network topology and protocol constraints.
* **Triage Heuristics**: The LLM triage layer performs inference over connection-level statistical aggregates rather than full packet captures (PCAP). It is designed as an analyst prioritization aid, not definitive forensic attribution.
* **Binary Classification**: The underlying classifier predicts binary verdicts (`normal` vs `attack`); attack typing is synthesized at the triage stage.

---

## Ethics
This project is conducted strictly for defensive security research. The evasion algorithms were built to stress-test and harden self-hosted models, not against any unauthorized or production third-party systems.
