# Experimental Results: Adversarial Robustness and Intrusion Detection

Measured numbers from the NSL-KDD benchmark training and adversarial evaluation.

## 1. Baseline Performance vs. Hardened Performance

| Metric | Baseline Random Forest | Adversarially Hardened Model | Delta |
|---|---|---|---|
| **Clean Test Accuracy** | 76.56% | 79.96% | +3.40% |
| **Clean Test Precision** | 96.71% | 97.03% | +0.32% |
| **Clean Test Recall (Attack Catch Rate)** | 60.90% | 66.84% | +5.95% |
| **Clean Test F1-Score** | 74.73% | 79.15% | +4.42% |

## 2. Adversarial Evasion Test Results

- **Threat Model**: Black-box greedy random search targeting normal empirical distributions.
- **Attacker-Controllable Features (5)**: `duration, src_bytes, dst_bytes, count, srv_count`
- **Evaluated Attack Connections**: 300 previously caught test attacks.

| Model Variant | Evasion Success Rate | Evaded Connections | Detection Retention |
|---|---|---|---|
| **Baseline Model** | **41.00%** | 123 / 300 | 59.00% |
| **Hardened Model (Post-Adversarial Training)** | **4.33%** | 13 / 300 | 95.67% |

### Key Observations
1. **Adversarial Vulnerability**: The baseline model was susceptible to evasion: **41.00%** of caught attacks were disguised as normal traffic simply by perturbing connection duration, bytes transferred, and connection counters.
2. **Hardening Recovery**: Adversarial training reduced evasion vulnerability from **41.00%** down to **4.33%** (an absolute reduction of **36.67%**).
3. **Preservation of Clean Recall**: Clean test set recall was preserved (66.84% vs 60.90%), proving that hardening against adversarial evasions did not cause a catastrophic collapse in general attack detection.
