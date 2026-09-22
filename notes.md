# Interview Preparation & Resume Bullets (Private Reference)

---

## 1. Resume Bullet Points (Ready to Copy-Paste)

### Option A (Full Impact & Adversarial Security Focus)
* Built a network intrusion detection pipeline on the NSL-KDD benchmark (122 encoded features, Random Forest with 200 trees) and designed a black-box adversarial evasion attack that bypassed 56.67% of baseline detections by perturbing 5 attacker-controllable features.
* Hardened the detector via adversarial training under a zero-leakage 50/50 test split protocol, slashing evasion vulnerability on strictly unseen holdout attacks from 56.67% to 16.00% (a 40.67% absolute reduction) while preserving 67.69% clean attack recall.
* Implemented an automated LLM triage engine (direct REST API via `requests`) that translates raw alert telemetry into structured incident notes with MITRE ATT&CK categorization, paired with an interactive Gradio web application.

### Option B (Compact 2-Bullet Version)
* Developed an adversarial-robust network intrusion detector (Random Forest, NSL-KDD); engineered a black-box evasion attack (56.67% bypass rate) and mitigated vulnerability to 16.00% via adversarial retraining on a strictly held-out evaluation set.
* Architected an AI-assisted SOC triage layer and Gradio dashboard that analyzes flagged connections to provide plain-English threat rationale and MITRE ATT&CK mappings.

---

## 2. Technical Interview Questions & Defenses

### Q1: How did you prevent test-set leakage during adversarial training?
> **Answer**: "A common flaw in adversarial training projects is generating evasions from the test set and re-evaluating the hardened model on that same test set, which leaks near-duplicate rows into training. To guarantee zero leakage, I split the NSL-KDD test set into two strict 50/50 partitions (`X_atk` and `X_holdout`). The evasion samples used for adversarial retraining were drawn solely from `X_atk`. All final evaluations—both baseline vs. hardened evasion rates and clean test recall—were measured exclusively on the unseen `X_holdout` partition, proving true generalization against adversarial attacks."

### Q2: Why did you use NSL-KDD if it is known to be dated (1999-era traffic)?
> **Answer**: "NSL-KDD is an established, standardized benchmark in academic and defensive security research. Because it has a fixed train/test split with known distribution shifts, it allows for reproducible, apples-to-apples comparison of adversarial attack and defense strategies without data leakage. In my project README, I explicitly state that NSL-KDD does not reflect modern encrypted TLS traffic or cloud architectures—the project's core contribution is the adversarial evasion testing and triage pipeline, not claiming production deployment on modern enterprise backbones."

### Q3: Why did you use a black-box random search instead of gradient-based attacks like FGSM or PGD?
> **Answer**: "Two fundamental reasons:
> 1. **Model Architecture**: Random Forests are non-differentiable step functions. You cannot directly calculate loss gradients with respect to input features like you can with neural networks.
> 2. **Threat Model Realism**: A real-world adversary attacking an external or edge IDS cannot see internal decision thresholds, tree topologies, or training weights. A black-box attack where the attacker only observes binary feedback (`normal` vs `flagged`) is much more representative of a realistic threat actor."

### Q4: How did you select the attacker-controllable features (`duration`, `src_bytes`, `dst_bytes`, `count`, `srv_count`)?
> **Answer**: "Feature controllability is central to realistic adversarial ML in cyber security. An attacker can directly control how long they keep a socket open (`duration`), how much data they pad into payloads (`src_bytes`, `dst_bytes`), and the transmission rate or connection burstiness (`count`, `srv_count`). In contrast, an attacker cannot arbitrarily manipulate host error rates or server-side TCP stack flags (`serror_rate`, `rerror_rate`) without breaking the underlying protocol. Restricting perturbations to plausible features ensures the generated adversarial traffic remains functionally viable."

### Q5: Did adversarial training hurt clean traffic detection or introduce false positives?
> **Answer**: "Evaluated strictly on held-out clean data (`X_holdout`), clean recall held strong at 67.69% (vs 61.95% baseline), and overall clean test accuracy was 80.59% (vs 77.32% baseline) with precision remaining high at 97.42%. Because the perturbed adversarial samples occupied ambiguous boundary regions between normal traffic distributions and attack traffic, retraining helped the ensemble define tighter decision boundaries without catastrophic degradation."

### Q6: What are the limitations of the LLM triage layer?
> **Answer**: "The LLM generates triage narratives based on connection summary statistics (bytes, duration, error flags) rather than raw packet payload inspections (PCAP). Therefore, the LLM provides an operational heuristic to prioritize analyst review and associate observed metrics with MITRE ATT&CK techniques—it is an analyst aid, not definitive forensic proof of attribution."

### Q7: What would you do with more engineering time?
> **Answer**:
> 1. Extend from binary classification to multi-class hierarchical classification (DoS, Probe, R2L, U2R).
> 2. Implement an adaptive adversary (e.g., genetic algorithms or Bayesian optimization) to test if the hardened model can be broken again.
> 3. Benchmark on a modern flow dataset like CICIDS2017 or UNSW-NB15 with contemporary DDoS and exploit payloads.
