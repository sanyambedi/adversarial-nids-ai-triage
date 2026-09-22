# Interview Preparation & Resume Bullets (Private Reference)

---

## 1. Resume Bullet Points (Ready to Copy-Paste)

### Option A (Full Impact & Adversarial Security Focus)
* Built a network intrusion detection pipeline on the NSL-KDD benchmark (122 encoded features) and designed a black-box adversarial evasion attack that bypassed 41.00% of baseline detections by perturbing 5 attacker-controllable features.
* Hardened the classifier via adversarial training, cutting evasion vulnerability from 41.00% down to 4.33% (a 36.67% absolute reduction) while preserving 66.84% recall on clean test traffic.
* Implemented an automated LLM triage engine (via direct REST API) that converts raw connection alerts into structured SOC incident notes with MITRE ATT&CK categorization, paired with an interactive Gradio web application.

### Option B (Compact 2-Bullet Version)
* Developed an adversarial-robust network intrusion detector (Random Forest, NSL-KDD); proved baseline fragility by crafting a black-box evasion attack (41.00% bypass rate) and mitigated the vulnerability to 4.33% via adversarial retraining.
* Architected an AI-assisted SOC triage layer and Gradio dashboard that analyzes flagged connections to provide plain-English threat rationale and MITRE ATT&CK mappings.

---

## 2. Technical Interview Questions & Defenses

### Q1: Why did you use NSL-KDD if it is known to be dated (1999-era traffic)?
> **Answer**: "NSL-KDD is an established, standardized benchmark in academic and defensive security research. Because it has a fixed train/test split with known distribution shifts, it allows for reproducible, apples-to-apples comparison of adversarial attack and defense strategies without data leakage. In my project README, I explicitly state that NSL-KDD does not reflect modern encrypted TLS traffic or cloud architectures—the project's core contribution is the adversarial evasion testing and triage pipeline, not claiming production deployment on modern enterprise backbones."

### Q2: Why did you use a black-box random search instead of gradient-based attacks like FGSM or PGD?
> **Answer**: "Two fundamental reasons:
> 1. **Model Architecture**: Random Forests are non-differentiable step functions. You cannot directly calculate loss gradients with respect to input features like you can with neural networks.
> 2. **Threat Model Realism**: A real-world adversary attacking an external or edge IDS cannot see internal decision thresholds, tree topologies, or training weights. A black-box attack where the attacker only observes binary feedback (`normal` vs `flagged`) is much more representative of a realistic threat actor."

### Q3: How did you select the attacker-controllable features (`duration`, `src_bytes`, `dst_bytes`, `count`, `srv_count`)?
> **Answer**: "Feature controllability is central to realistic adversarial ML in cyber security. An attacker can directly control how long they keep a socket open (`duration`), how much data they pad into payloads (`src_bytes`, `dst_bytes`), and the transmission rate or connection burstiness (`count`, `srv_count`). In contrast, an attacker cannot arbitrarily manipulate host error rates or server-side TCP stack flags (`serror_rate`, `rerror_rate`) without breaking the underlying protocol. Restricting perturbations to plausible features ensures the generated adversarial traffic remains functionally viable."

### Q4: Did adversarial training hurt clean traffic detection or introduce false positives?
> **Answer**: "In our empirical measurements, clean test recall actually improved from 60.90% to 66.84%, and overall clean test accuracy rose from 76.56% to 79.96% with precision holding strong at 97.03%. Because the perturbed adversarial samples occupied ambiguous boundary regions between normal traffic distributions and attack traffic, retraining helped the ensemble define tighter decision boundaries without catastrophic degradation."

### Q5: What are the limitations of the LLM triage layer?
> **Answer**: "The LLM generates triage narratives based on connection summary statistics (bytes, duration, error flags) rather than raw packet payload inspections (PCAP). Therefore, the LLM provides an operational heuristic to prioritize analyst review and associate observed metrics with MITRE ATT&CK techniques—it is an analyst aid, not definitive forensic proof of attribution."

### Q6: What would you do with more engineering time?
> **Answer**:
> 1. Extend from binary classification to multi-class hierarchical classification (DoS, Probe, R2L, U2R).
> 2. Implement an adaptive adversary (e.g., genetic algorithms or Bayesian optimization) to test if the hardened model can be broken again.
> 3. Benchmark on a modern flow dataset like CICIDS2017 or UNSW-NB15 with contemporary DDoS and exploit payloads.
