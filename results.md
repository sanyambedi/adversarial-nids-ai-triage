# Experimental Results: Leak-Free Adversarial Robustness and Intrusion Detection

Empirical benchmark evaluation conducted under a strict zero-leakage protocol (50% attack-crafting split / 50% held-out evaluation split). Both models evaluated with Random Forest (n_estimators=200).

## 1. Baseline Performance vs. Hardened Performance (Held-Out Test Traffic)

| Metric | Baseline Random Forest | Adversarially Hardened Model | Delta |
|---|---|---|---|
| **Clean Test Accuracy** | 77.32% | 80.59% | +3.26% |
| **Clean Test Precision** | 97.19% | 97.42% | +0.23% |
| **Clean Test Recall (Attack Catch Rate)** | 61.95% | 67.69% | +5.74% |
| **Clean Test F1-Score** | 75.67% | 79.88% | +4.21% |

## 2. Adversarial Evasion Test Results (Strictly Unseen Attacks)

- **Threat Model**: Black-box greedy random search targeting normal empirical distributions.
- **Attacker-Controllable Features (5)**: `duration, src_bytes, dst_bytes, count, srv_count`
- **Evaluated Attack Connections**: 300 unseen attack connections from the held-out split (`X_holdout`).
- **Leakage Prevention**: Zero samples or perturbations evaluated here were seen during training or adversarial retraining.

| Model Variant | Evasion Success Rate | Evaded Connections | Detection Retention |
|---|---|---|---|
| **Baseline Model** | **56.67%** | 170 / 300 | 43.33% |
| **Hardened Model (Post-Adversarial Training)** | **16.00%** | 48 / 300 | 84.00% |

### Key Observations
1. **Adversarial Vulnerability on Unseen Attacks**: The baseline model allowed **56.67%** of unseen attacks in the holdout split to slip past as normal traffic via black-box feature manipulation.
2. **Generalizable Hardening**: Even on completely held-out, unseen attack connections, adversarial training reduced the evasion rate from **56.67%** to **16.00%** (an absolute vulnerability reduction of **40.67%**).
3. **Honest Robustness / Accuracy Balance**: Evaluated strictly on held-out clean data, clean recall is 67.69% (vs 61.95% baseline) and accuracy is 80.59% (vs 77.32%), confirming robust decision boundaries without artificial test leakage.
