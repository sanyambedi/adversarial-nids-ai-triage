"""app.py - Gradio Web Application for Adversarial-Robust Intrusion Detection with AI Triage.

Provides an interactive dashboard to:
1. Upload network traffic logs (or load sample_test.csv).
2. Predict benign vs. attack traffic using the adversarially-hardened model.
3. Generate plain-English AI triage notes and MITRE ATT&CK mappings for flagged attacks.
"""

import os
import json
import joblib
import pandas as pd
import gradio as gr

from preprocess import prepare_inference_data, FEATURE_COLUMNS_FILE
from triage import explain_flagged_row

MODEL_FILE = "model.joblib"
SAMPLE_CSV_FILE = "sample_test.csv"
RESULTS_FILE = "results.json"

# Load model and column schema
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("model.joblib not found. Please run train.py first!")

model = joblib.load(MODEL_FILE)
feature_cols = joblib.load(FEATURE_COLUMNS_FILE)

# Load measured benchmark results if available
metrics_summary = ""
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        r = json.load(f)
    metrics_summary = f"""
### Model Robustness Benchmark
* **Baseline Evasion Vulnerability:** `{r['baseline']['evasion_rate']:.2%}` ({r['baseline']['evaded_count']}/{r['sample_size_evasion']} attacks evaded)
* **Hardened Evasion Vulnerability:** `{r['hardened']['evasion_rate']:.2%}` ({r['hardened']['evaded_count']}/{r['sample_size_evasion']} attacks evaded)
* **Robustness Gain:** `{r['baseline']['evasion_rate'] - r['hardened']['evasion_rate']:.2%}` evasion reduction via adversarial training
* **Clean Test Attack Catch Rate (Recall):** `{r['hardened']['recall']:.2%}` preserved
"""


def analyze_traffic(file_obj, max_rows=30):
    """Processes uploaded CSV file through the inference pipeline and LLM triage layer."""
    if file_obj is None:
        if os.path.exists(SAMPLE_CSV_FILE):
            filepath = SAMPLE_CSV_FILE
        else:
            return pd.DataFrame(), "Please upload a CSV file or ensure sample_test.csv is present."
    else:
        filepath = file_obj.name

    df = pd.read_csv(filepath)
    if len(df) > max_rows:
        df_subset = df.head(max_rows).copy()
        row_note = f"Showing first {max_rows} of {len(df)} connections."
    else:
        df_subset = df.copy()
        row_note = f"Processed {len(df_subset)} connections."

    # Align columns using preprocess.py (single source of truth)
    X_inference = prepare_inference_data(df_subset, feature_cols)
    preds = model.predict(X_inference)

    results = []
    attack_count = 0

    for idx, pred in enumerate(preds):
        raw_row = df_subset.iloc[idx].to_dict()
        if pred == 1:
            attack_count += 1
            triage_info = explain_flagged_row(raw_row)
            results.append({
                "Row": idx + 1,
                "Verdict": "[ATTACK]",
                "Protocol": str(raw_row.get("protocol_type", "unknown")),
                "Service": str(raw_row.get("service", "unknown")),
                "Likely Attack Type": triage_info.get("likely_attack_type", "Unknown Attack"),
                "MITRE Category": triage_info.get("mitre_category", "Unknown"),
                "Severity": str(triage_info.get("severity", "medium")).upper(),
                "AI Analyst Summary": triage_info.get("one_line_reason", "Flagged by detector.")
            })
        else:
            results.append({
                "Row": idx + 1,
                "Verdict": "[NORMAL]",
                "Protocol": str(raw_row.get("protocol_type", "unknown")),
                "Service": str(raw_row.get("service", "unknown")),
                "Likely Attack Type": "-",
                "MITRE Category": "-",
                "Severity": "LOW",
                "AI Analyst Summary": "Benign network connection verified."
            })

    result_df = pd.DataFrame(results)
    total = len(df_subset)
    normal_count = total - attack_count
    status_summary = f"""
### Triage Summary
* **Total Connections Analyzed:** {total} ({row_note})
* **Flagged Intrusions:** `{attack_count}` ({attack_count/total:.1%})
* **Benign Connections:** `{normal_count}` ({normal_count/total:.1%})
"""
    return result_df, status_summary


def load_sample():
    """Helper to analyze the built-in sample_test.csv directly."""
    return analyze_traffic(None)


# Gradio Interface
with gr.Blocks(title="Adversarial-Robust Intrusion Detector with AI Triage") as demo:
    gr.Markdown("""
    # Adversarial-Robust Intrusion Detector with AI Triage
    A network intrusion detection system trained on NSL-KDD, stress-tested against a black-box evasion attack,
    hardened with adversarial training, and equipped with an AI triage narrative for every detected threat.
    """)

    if metrics_summary:
        gr.Markdown(metrics_summary)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Network Traffic CSV (NSL-KDD Format)",
                file_types=[".csv", ".txt"]
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze Uploaded File", variant="primary")
                sample_btn = gr.Button("Test Sample Data (sample_test.csv)", variant="secondary")

            summary_output = gr.Markdown(value="Upload a file or click **Test Sample Data** to begin triage.")

        with gr.Column(scale=2):
            table_output = gr.Dataframe(
                headers=["Row", "Verdict", "Protocol", "Service", "Likely Attack Type", "MITRE Category", "Severity", "AI Analyst Summary"],
                datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                label="Connection Inspection & Triage Table",
                wrap=True
            )

    analyze_btn.click(
        fn=analyze_traffic,
        inputs=[file_input],
        outputs=[table_output, summary_output]
    )

    sample_btn.click(
        fn=load_sample,
        inputs=[],
        outputs=[table_output, summary_output]
    )

    gr.Markdown("""
    ---
    **Notice & Ethics:** Prototype built for defensive security research and analyst workload triage.
    NSL-KDD is a standard academic benchmark; LLM narrative represents automated heuristic interpretation, not forensic ground truth.
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
