import streamlit as st
import pandas as pd
import json, io, os, yaml
from qc_evaluator import evaluate_with_openai, load_rubric
from openai import OpenAI

st.set_page_config(page_title="GPT Call QC", layout="wide")
st.title("📞 GPT-based Call QC (Configurable Rubrics)")

with st.sidebar:
    st.header("🔐 OpenAI Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your key is kept in memory for this session only.")
    model = st.text_input("Model", value="gpt-4o-mini", help="Use any chat-capable model available to your account.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    use_responses = st.toggle("Use Responses API (json_schema)", value=True)
    st.markdown("---")
    st.caption("Tip: Create an `.env` file and set OPENAI_API_KEY to avoid pasting every time.")

tab1, tab2, tab3 = st.tabs(["Single Call", "Batch", "Settings"])

default_yaml = open(os.path.join("config","qc_metrics.example.yaml"), "r", encoding="utf-8").read()

with tab1:
    st.subheader("Single Call Evaluation")
    transcript = st.text_area("Paste transcript here", height=220, placeholder="Agent: ...\nOwner: ...")
    audio_file = st.file_uploader("Or upload audio (mp3/wav/m4a) to transcribe", type=["mp3","wav","m4a"], accept_multiple_files=False)
    colA, colB = st.columns(2)
    with colA:
        use_yaml = st.text_area("QC Metrics YAML", value=default_yaml, height=420)
    with colB:
        st.write("Preview")
        try:
            rb = load_rubric(use_yaml)
            st.json(rb)
        except Exception as e:
            st.error(f"Invalid YAML: {e}")

    if st.button("📝 Transcribe Audio", disabled=not audio_file or not api_key):
        try:
            client = OpenAI(api_key=api_key)
            with st.spinner("Transcribing..."):
                # Try new transcription model first, fallback to whisper-1
                try:
                    audio_bytes = audio_file.read()
                    tmp_path = f"/tmp/{audio_file.name}"
                    with open(tmp_path, "wb") as f: f.write(audio_bytes)
                    tr = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=open(tmp_path, "rb"))
                except Exception:
                    audio_bytes = audio_file.read()
                    tmp_path = f"/tmp/{audio_file.name}"
                    with open(tmp_path, "wb") as f: f.write(audio_bytes)
                    tr = client.audio.transcriptions.create(model="whisper-1", file=open(tmp_path, "rb"))
            transcript = tr.text
            st.success("Transcription complete. Inserted into the transcript box.")
            st.session_state["transcript_text"] = transcript
        except Exception as e:
            st.error(f"Transcription failed: {e}")

    if st.session_state.get("transcript_text") and not transcript:
        transcript = st.session_state["transcript_text"]

    if st.button("⚖️ Evaluate QC", type="primary", disabled=not api_key or not (transcript or audio_file)):
        if not transcript:
            st.warning("No transcript found. Please paste a transcript or run transcription.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("Scoring with GPT..."):
                    result = evaluate_with_openai(client, model, transcript, use_yaml, temperature, use_responses_api=use_responses)
                st.success("Done")
                st.subheader("Overall")
                st.metric("Score", result.get("overall_score", 0))
                st.metric("Verdict", result.get("verdict","N/A"))
                st.write("Summary")
                st.write(result.get("summary",""))
                st.write("Red Flags")
                st.write(result.get("red_flags", []))

                st.subheader("Per-metric Details")
                pm = result.get("per_metric", [])
                if pm:
                    df = pd.DataFrame([{
                        "id": m.get("id"),
                        "name": m.get("name"),
                        "weight": m.get("weight"),
                        "score": m.get("score"),
                        "rationale": m.get("rationale")
                    } for m in pm])
                    st.dataframe(df, use_container_width=True)
                st.subheader("Raw JSON")
                st.code(json.dumps(result, ensure_ascii=False, indent=2))

                # Downloads
                st.download_button("⬇️ Download JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="qc_result.json", mime="application/json")
                if pm:
                    csv_df = pd.DataFrame(pm)
                    st.download_button("⬇️ Download CSV (per-metric)", data=csv_df.to_csv(index=False), file_name="qc_per_metric.csv", mime="text/csv")
            except Exception as e:
                st.error(f"QC failed: {e}")

with tab2:
    st.subheader("Batch Evaluation")
    batch_yaml = st.text_area("QC Metrics YAML (batch)", value=default_yaml, height=200)
    uploaded_csv = st.file_uploader("Upload CSV with columns: call_id, transcript", type=["csv"])
    if st.button("Run Batch", disabled=not(api_key and uploaded_csv)):
        try:
            client = OpenAI(api_key=api_key)
            df = pd.read_csv(uploaded_csv)
            out_rows = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                call_id = row.get("call_id", f"row_{i}")
                tx = str(row.get("transcript",""))
                res = evaluate_with_openai(client, model, tx, batch_yaml, 0.0, use_responses_api=use_responses)
                out_rows.append({
                    "call_id": call_id,
                    "overall_score": res.get("overall_score"),
                    "verdict": res.get("verdict"),
                    "summary": res.get("summary",""),
                    "per_metric_json": json.dumps(res.get("per_metric", []), ensure_ascii=False)
                })
                progress.progress((i+1)/len(df))
            out_df = pd.DataFrame(out_rows)
            st.success("Batch complete")
            st.dataframe(out_df, use_container_width=True)
            st.download_button("⬇️ Download Batch CSV", data=out_df.to_csv(index=False), file_name="qc_batch_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch failed: {e}")

with tab3:
    st.subheader("Settings & Notes")
    st.markdown("""
- **Privacy:** The API key is kept in memory only for this session unless you change the code.
- **Latency & Cost:** Longer transcripts increase tokens. Consider summarizing or chunking.
- **JSON Enforcement:** If your account supports the **Responses API** with `json_schema`, keep it on for rock-solid JSON outputs.
- **Rubrics:** Tune weights and `verdict_thresholds` to match your policy.
- **Safety:** Always human-review red flags before actions.
""")
