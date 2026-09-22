"""triage.py - Plain-English incident triage layer using LLM REST API.

Converts flagged network connections into structured SOC incident notes
containing attack type estimation, MITRE ATT&CK categorization, severity,
and an analyst explanation. Uses direct `requests` calls without heavy SDKs.
"""

import os
import json
import requests
def _load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

_load_env()

# Check for configured API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _call_llm_rest(prompt: str) -> str:
    """Invokes LLM provider REST API directly using requests."""
    # 1. Groq REST endpoint (Fast, free tier)
    if GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        # Candidate models supported by Groq accounts
        candidate_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        last_err = None
        for model_name in candidate_models:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 250
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                last_err = f"{resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                last_err = str(e)
        raise RuntimeError(f"Groq call failed across models: {last_err}")

    # 2. Google Gemini REST endpoint
    elif GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 250,
                "responseMimeType": "application/json"
            }
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # 3. Anthropic REST endpoint
    elif ANTHROPIC_API_KEY:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 250,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    # 4. OpenAI REST endpoint
    elif OPENAI_API_KEY:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 250,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise ValueError(
        "No LLM API key configured. Please set GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env."
    )


def _dev_heuristic_fallback(row: dict) -> dict:
    """Internal offline fallback when no API key is provided during local dev testing."""
    service = str(row.get("service", "")).lower()
    flag = str(row.get("flag", "")).lower()
    count = float(row.get("count", 0))
    serror = float(row.get("serror_rate", 0))
    src_bytes = float(row.get("src_bytes", 0))
    dst_bytes = float(row.get("dst_bytes", 0))

    if count > 100 and (serror > 0.5 or flag in ["s0", "s1"]):
        return {
            "likely_attack_type": "SYN Flood / DoS",
            "mitre_category": "T1498 (Network Denial of Service)",
            "severity": "high",
            "one_line_reason": f"Abnormally high connection count ({int(count)}) with high SYN error rate ({serror:.1%}) targeting {service or 'service'}."
        }
    elif count > 50 and dst_bytes == 0:
        return {
            "likely_attack_type": "Port Scan / Reconnaissance",
            "mitre_category": "T1046 (Network Service Discovery)",
            "severity": "medium",
            "one_line_reason": f"Rapid repeated connection attempts with 0 return bytes indicating host or port reconnaissance."
        }
    elif src_bytes > 50000:
        return {
            "likely_attack_type": "Data Exfiltration / Smurf DoS",
            "mitre_category": "T1048 (Exfiltration Over Alternative Protocol)",
            "severity": "high",
            "one_line_reason": f"Excessive outbound payload size ({int(src_bytes)} bytes) anomalous for typical {service or 'traffic'}."
        }
    else:
        return {
            "likely_attack_type": "Suspicious Anomalous Traffic",
            "mitre_category": "T1190 (Exploit Public-Facing Application)",
            "severity": "medium",
            "one_line_reason": f"Statistical connection anomaly detected on service '{service}' with flag state '{flag}'."
        }


def explain_flagged_row(row: dict) -> dict:
    """Analyzes a flagged connection row and returns a structured incident explanation."""
    # Filter to most salient network traffic features to optimize tokens
    salient_keys = [
        "protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes",
        "count", "srv_count", "serror_rate", "rerror_rate", "same_srv_rate",
        "diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_serror_rate"
    ]
    compact_row = {k: row[k] for k in salient_keys if k in row}
    if not compact_row:
        compact_row = {k: v for k, v in list(row.items())[:15]}

    prompt = f"""You are a senior SOC analyst assistant. A network intrusion detection system flagged this connection as an attack.
Analyze the connection metrics and return ONLY a valid JSON object matching this schema:
{{
  "likely_attack_type": "<e.g. SYN Flood, Port Scan, Buffer Overflow, Ping of Death>",
  "mitre_category": "<e.g. T1498 Network DoS, T1046 Network Service Discovery>",
  "severity": "<low|medium|high>",
  "one_line_reason": "<1-2 sentence concise technical rationale for analyst>"
}}

Connection metrics:
{json.dumps(compact_row, default=str)}
"""

    try:
        raw_text = _call_llm_rest(prompt)
        # Strip markdown json codeblocks if returned
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
        # Ensure all required keys are present
        for req in ["likely_attack_type", "mitre_category", "severity", "one_line_reason"]:
            if req not in result:
                result[req] = "unknown"
        return result
    except Exception as e:
        # If API key missing or network call failed, provide fallback
        fallback = _dev_heuristic_fallback(row)
        fallback["note"] = f"(Generated via heuristic: {str(e)[:60]})" if "No LLM API key" not in str(e) else "(Heuristic fallback - configure LLM API key in .env)"
        return fallback


if __name__ == "__main__":
    # Test on a simulated SYN flood connection
    test_conn = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
        "count": 250,
        "srv_count": 250,
        "serror_rate": 1.0,
        "srv_serror_rate": 1.0,
        "same_srv_rate": 1.0
    }
    print("Testing triage layer on sample attack row:")
    explanation = explain_flagged_row(test_conn)
    print(json.dumps(explanation, indent=2))
