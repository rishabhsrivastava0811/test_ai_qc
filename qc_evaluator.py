import json, re, yaml, math, time
from typing import Dict, Any, List, Tuple, Optional

def load_rubric(yaml_text: str) -> Dict[str, Any]:
    return yaml.safe_load(yaml_text)

def build_system_prompt(rubric: Dict[str, Any]) -> str:
    rules = rubric.get("global_rules", [])
    rules_block = "\n".join([f"- {r}" for r in rules])
    return f"""You are a strict Call Quality Analyst for a food-delivery marketplace.
Follow the rubric faithfully. Never invent quotes or timestamps.
Global rules:
{rules_block}
Output ONLY valid JSON matching the schema I will provide."""

def build_user_prompt(transcript: str, rubric: Dict[str, Any]) -> str:
    metrics = rubric.get("metrics", [])
    verdict = rubric.get("verdict_thresholds", {"pass": 80, "needs_review": 60})
    metrics_block = []
    for m in metrics:
        metrics_block.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "weight": m.get("weight", 0.1),
            "description": m.get("description",""),
            "rubric": m.get("rubric",{}),
            "min_quotes": m.get("min_quotes",1),
            "must_flag": m.get("must_flag", []),
        })
    payload = {
        "verdict_thresholds": verdict,
        "metrics": metrics_block,
        "transcript": transcript,
    }
    return json.dumps(payload, ensure_ascii=False)

def target_json_schema() -> Dict[str, Any]:
    # A minimal JSON schema for Responses API; but we'll also validate manually after parsing
    return {
        "name": "qc_result",
        "schema": {
            "type": "object",
            "properties": {
                "overall_score": {"type": "number"},
                "verdict": {"type": "string", "enum": ["PASS", "NEEDS_REVIEW", "FAIL"]},
                "summary": {"type": "string"},
                "red_flags": {"type": "array", "items": {"type": "string"}},
                "per_metric": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                            "score": {"type": "number"},
                            "rationale": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "quote": {"type": "string"},
                                        "start_time": {"type": "string"},
                                        "end_time": {"type": "string"}
                                    },
                                    "required": ["quote"]
                                }
                            }
                        },
                        "required": ["id", "score"]
                    }
                }
            },
            "required": ["overall_score", "verdict", "per_metric"]
        }
    }

def safe_json_extract(text: str) -> str:
    # Try to extract the first JSON object in text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)

def compute_overall(per_metric: List[Dict[str, Any]]) -> float:
    total_w = sum([m.get("weight", 0.0) for m in per_metric]) or 1.0
    s = 0.0
    for m in per_metric:
        s += (m.get("score", 0.0) * m.get("weight", 0.0))
    return round(s/total_w, 2)

def verdict_from_score(score: float, thresholds: Dict[str, Any]) -> str:
    if score >= thresholds.get("pass", 80):
        return "PASS"
    if score >= thresholds.get("needs_review", 60):
        return "NEEDS_REVIEW"
    return "FAIL"

def normalize_result(parsed: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    # Fill missing fields and recompute overall if needed
    pm = parsed.get("per_metric", [])
    overall = parsed.get("overall_score")
    if overall is None and pm:
        overall = compute_overall(pm)
    v = parsed.get("verdict") or verdict_from_score(overall or 0, thresholds)
    parsed["overall_score"] = float(overall or 0)
    parsed["verdict"] = v
    parsed.setdefault("summary", "")
    parsed.setdefault("red_flags", [])
    return parsed

def evaluate_with_openai(client, model: str, transcript: str, rubric_yaml: str, temperature: float=0.0, use_responses_api: bool=True) -> Dict[str, Any]:
    rubric = load_rubric(rubric_yaml)
    sys_prompt = build_system_prompt(rubric)
    user_payload = build_user_prompt(transcript, rubric)
    schema = target_json_schema()

    if use_responses_api:
        # Prefer Responses API with JSON schema if available
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role":"system", "content": sys_prompt},
                    {"role":"user", "content": f"""Using this JSON payload (rubric and transcript):\n```json\n{user_payload}\n```\nReturn a JSON that matches this schema name {schema["name"]}."""}
                ],
                temperature=temperature,
                response_format={
                    "type":"json_schema",
                    "json_schema": schema
                }
            )
            text = resp.output_text
        except Exception as e:
            # Fallback to Chat Completions
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": sys_prompt},
                    {"role":"user","content": f"Rubric+Transcript JSON:\n{user_payload}\nReturn ONLY JSON matching the schema: {json.dumps(schema)}"}
                ],
                temperature=temperature,
                response_format={"type":"json_object"}
            )
            text = chat.choices[0].message.content
    else:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content": sys_prompt},
                {"role":"user","content": f"Rubric+Transcript JSON:\n{user_payload}\nReturn ONLY JSON matching the schema: {json.dumps(schema)}"}
            ],
            temperature=temperature,
            response_format={"type":"json_object"}
        )
        text = chat.choices[0].message.content

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = json.loads(safe_json_extract(text))

    parsed = normalize_result(parsed, load_rubric(rubric_yaml).get("verdict_thresholds", {"pass":80,"needs_review":60}))
    return parsed
