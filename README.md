# Adversarial-Robust Intrusion Detector with AI Triage

An end-to-end Network Intrusion Detection System (NIDS) built on the NSL-KDD benchmark, stress-tested with a black-box adversarial evasion attack, hardened via adversarial retraining, and coupled with an LLM-assisted incident triage layer that explains flagged threats in plain English.

---

## The Problem
Most student and tutorial intrusion detection systems stop at *"trained a classifier, achieved X% accuracy on test data."* In real security operations, two critical challenges make simple accuracy misleading:
1. **Adversarial Evasion**: Attackers actively modify controllable connection properties to sneak malicious traffic past static machine learning boundaries.
2. **Alert Fatigue**: Security Operations Center (SOC) analysts are overwhelmed by raw binary alerts without contextual explanation or threat categorization.

This project addresses both: we build an initial classifier, design an evasion attack against it, measure how easily it is fooled, harden it with adversarial retraining, and translate every detected alert into a structured, plain-English incident note.

---

## Architecture

```text
NSL-KDD Dataset (KDDTrain+, KDDTest+ / 41 Connection Features)
                         │
        preprocess.py (One-hot encoding, column schema alignment)
                         │
        Baseline Random Forest Classifier (150 trees)
                         │
        ┌────────────────┴────────────────────────┐
        │                                         │
Clean Test Evaluation                    Black-Box Evasion Attack
(Baseline recall: 60.90%)               (Perturb 5 controllable features
                                         toward normal distribution)
                                                  │
                                         41.00% of attacks evaded
                                                  │
                                         Adversarial Retraining
                                         (Augment training set with
                                         evasions, retrain model)
                                                  │
                                         Hardened Model Evaluation
                                         (Evasion rate drops to 4.33%
                                          Clean recall: 66.84%)
                                                  │
                                         Flagged Intrusions (Verdict == ATTACK)
                                                  │
                                         LLM Incident Triage Layer
                                         (Attack Type, MITRE ATT&CK, Severity,
                                          Concise Analyst Summary)
                                                  │
                                         Interactive Gradio Demo (app.py)
```

---

## Real Measured Experimental Results

All numbers below were empirically measured on the official NSL-KDD test set (`KDDTest+.txt`):

### 1. Classification Performance

| Metric | Baseline Classifier | Adversarially Hardened Model | Delta |
|---|---|---|---|
| **Clean Test Accuracy** | 76.56% | **79.96%** | `+3.40%` |
| **Clean Test Precision** | 96.71% | **97.03%** | `+0.32%` |
| **Attack Catch Rate (Recall)** | 60.90% | **66.84%** | `+5.94%` |
| **Clean Test F1-Score** | 74.73% | **79.15%** | `+4.42%` |

### 2. Adversarial Evasion Attack & Hardening

* **Threat Model**: Black-box greedy random search. The adversary observes only model output (`normal` vs `attack`) and does not know tree splits, weights, or hyperparameters.
* **Attacker-Controllable Features (5)**: `["duration", "src_bytes", "dst_bytes", "count", "srv_count"]` (features an adversary can plausibly tune during connection establishment).
* **Sample Size**: 300 correctly caught test attack connections.

| Model Variant | Evasion Success Rate | Evaded Connections | Detection Retention |
|---|---|---|---|
| **Baseline Model** | **41.00%** | 123 / 300 | 59.00% |
| **Hardened Model** | **4.33%** | 13 / 300 | **95.67%** |

### Key Takeaways
1. **High Baseline Fragility**: Simply perturbing 5 connection features allowed **41.00%** of previously detected attacks to slip past the baseline detector as "normal" traffic.
2. **Substantial Hardening**: Adding successful evasion examples back into the training distribution reduced the evasion success rate from **41.00% down to 4.33%** (an absolute vulnerability reduction of **36.67%**).
3. **No Trade-Off Degradation**: Hardening did not degrade performance on normal traffic; in fact, clean test recall improved from **60.90% to 66.84%** due to improved boundary generalization.

---

## LLM Triage Layer

Whenever traffic is flagged as an attack, the triage layer (`triage.py`) analyzes connection metrics and generates a structured incident summary:
* **Likely Attack Type**: e.g. *SYN Flood / Denial of Service*, *Port Scan / Reconnaissance*, *Smurf Amplification*.
* **MITRE ATT&CK Mapping**: e.g. `T1498` (*Network Denial of Service*), `T1046` (*Network Service Discovery*).
* **Severity**: `LOW`, `MEDIUM`, or `HIGH`.
* **Analyst Rationale**: 1-2 concise sentences explaining the primary telemetry anomalies (e.g. *Abnormally high connection count (250) with high SYN error rate (100%) targeting HTTP*).

Supports Groq, Google Gemini, Anthropic, or OpenAI REST APIs via `.env`, with zero heavy SDK dependencies.

---

## Project Structure

```text
ids-adversarial-triage/
├── data/                       # Local cache for KDDTrain+.txt & KDDTest+.txt (git-ignored)
├── preprocess.py               # Canonical schema & encoding (single source of truth)
├── train.py                    # Training, black-box evasion, and hardening pipeline
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
git clone <your-repo-url>
cd ids-adversarial-triage
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
To re-run the entire training, evasion testing, and hardening pipeline:
```bash
python train.py
```

To launch the interactive Gradio dashboard:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser. You can click **Test Sample Data (sample_test.csv)** for an instant demo or upload any NSL-KDD formatted CSV.

---

## Limitations
* **Dataset Age**: NSL-KDD is derived from 1999 simulated network traffic. It does not reflect modern protocol stacks, cloud VPC flow logs, or encrypted HTTPS/TLS payload dynamics. It is used here strictly as a reproducible benchmark for comparative adversarial robustness.
* **Threat Model Scope**: The evasion search assumes an attacker controls connection duration, byte volumes, and connection rates (`duration, src_bytes, dst_bytes, count, srv_count`). Real-world attacker capabilities depend on network topology and protocol constraints.
* **Triage Heuristics**: The LLM triage layer performs inference over connection-level statistical aggregates rather than full packet captures (PCAP). It is designed as an analyst prioritization aid, not definitive forensic attribution.
* **Binary Classification**: The underlying classifier predicts binary verdicts (`normal` vs `attack`); attack typing is synthesized at the triage stage.

---

## Ethics
This project is conducted strictly for defensive security research. The evasion algorithms were built to stress-test and harden self-hosted models, not against any unauthorized or production third-party systems.
