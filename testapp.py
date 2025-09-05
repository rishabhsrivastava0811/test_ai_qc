import streamlit as st
import pandas as pd
import json, os, tempfile, requests
from qc_evaluator import evaluate_with_openai, load_rubric
from openai import OpenAI

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="GPT Call QC", layout="wide")
st.title("📞 GPT-based Call QC (Configurable Rubrics + Mixed Bilingual Transcription)")

# ---------------------------
# Sidebar settings
# ---------------------------
with st.sidebar:
    st.header("🔐 OpenAI Settings")
    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Your key is kept in memory for this session only.")
    model = st.text_input("QC Model", value="gpt-4o",
                          help="Model for QC evaluation (e.g., gpt-4o, gpt-4o-mini).")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    use_responses = st.toggle("Use Responses API (json_schema)", value=True)
    st.markdown("---")

# ---------------------------
# Discover YAML rubrics
# ---------------------------
config_dir = "config"
yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
if not yaml_files:
    st.error("No YAML rubric files found in the `config/` directory.")
    st.stop()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Single Call", "Batch", "Settings"])


# ---------------------------
# Helper: Transcribe + QC-format audio
# ---------------------------
def transcribe_and_format(client, audio_file):
    """
    Step 1: Whisper transcription (Hindi + English auto-detect).
    Step 2: GPT-4o formatting into Devanagari + Roman with QC annotations.
    """
    try:
        # Save temp file
        audio_bytes = audio_file.read()
        tmp_path = f"/tmp/{audio_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # Step 1: Whisper transcription
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=None,  # auto-detect Hindi/English
                response_format="json"
            )
        raw_text = tr["text"]

        # Step 2: GPT formatting + QC
        system_prompt = """You are a transcription formatter and call quality analyst.

Rules:
- Keep Hindi words in Devanagari (नमस्ते).
- Keep English words in Roman (hello).
- Do NOT write English words in Devanagari phonetics.
- Do NOT translate; preserve the original language.
- Annotate each segment with:
  • pronunciation (correct / incorrect)
  • tone (polite, harsh, neutral, enthusiastic)
  • pace (slow, fast, normal)

Output must be JSON with top-level object { "segments": [...] }"""

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "TranscriptQC",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "segment": {"type": "string"},
                                        "text": {"type": "string"},
                                        "pronunciation": {"type": "string"},
                                        "tone": {"type": "string"},
                                        "pace": {"type": "string"}
                                    },
                                    "required": ["segment", "text", "pronunciation", "tone", "pace"]
                                }
                            }
                        },
                        "required": ["segments"]
                    }
                }
            }
        )

        try:
            parsed = json.loads(response.choices[0].message.content)
            segments = parsed.get("segments", [])
        except Exception as e:
            st.error(f"⚠️ Error parsing GPT output: {e}")
            segments = [{"segment": "0", "text": raw_text,
                         "pronunciation": "N/A", "tone": "N/A", "pace": "N/A"}]

        return raw_text, segments

    except Exception as e:
        st.error(f"Transcription+QC failed: {e}")
        return "", []


# ---------------------------
# Tab 1 - Single Call
# ---------------------------
with tab1:
    st.subheader("Single Call Evaluation")

    transcript_raw = st.session_state.get("transcript_raw", "")
    transcript_segments = st.session_state.get("transcript_segments", [])

    audio_file = st.file_uploader(
        "Upload audio (mp3/wav/m4a)",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=False
    )

    # Rubric selection
    selected_yaml = st.selectbox("Choose QC Rubric", options=yaml_files, index=0)
    yaml_path = os.path.join(config_dir, selected_yaml)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            use_yaml = f.read()
        rb = load_rubric(use_yaml)
    except Exception as e:
        st.error(f"Error loading YAML: {e}")
        use_yaml = None

    # --- Transcribe Button ---
    if st.button("📝 Transcribe Audio", disabled=not audio_file):
        if not api_key:
            st.warning("Please enter your OpenAI API Key.")
        else:
            client = OpenAI(api_key=api_key)
            with st.spinner("Transcribing + formatting..."):
                raw_text, segments = transcribe_and_format(client, audio_file)
                transcript_raw = raw_text
                transcript_segments = segments
                st.session_state["transcript_raw"] = raw_text
                st.session_state["transcript_segments"] = segments

            st.success("Transcription complete ✅")
            st.subheader("Raw Transcript (Whisper)")
            st.text_area("Raw", transcript_raw, height=150)

            st.subheader("Formatted Transcript with QC")
            st.json(transcript_segments)

    # --- Evaluate Button ---
    if st.button("⚖️ Evaluate QC", type="primary",
                 disabled=not (transcript_segments and use_yaml)):
        client = OpenAI(api_key=api_key)
        transcript_text = " ".join([seg["text"] for seg in transcript_segments])
        with st.spinner("Scoring with GPT..."):
            result = evaluate_with_openai(
                client, model, transcript_text, use_yaml,
                temperature, use_responses_api=use_responses
            )

        st.success("QC Evaluation Done ✅")
        st.subheader("Transcript Used for QC")
        st.text_area("Transcript", transcript_text, height=200)

        st.subheader("Overall")
        st.metric("Score", result.get("overall_score", 0))
        st.metric("Verdict", result.get("verdict", "N/A"))

        st.write("Summary")
        st.write(result.get("summary", ""))
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


# ---------------------------
# Tab 2 - Batch
# ---------------------------
with tab2:
    st.subheader("Batch Evaluation")

    selected_batch_yaml = st.selectbox("Choose QC Rubric (Batch)", options=yaml_files, index=0, key="batch_yaml")
    yaml_path = os.path.join(config_dir, selected_batch_yaml)
    with open(yaml_path, "r", encoding="utf-8") as f:
        batch_yaml = f.read()

    uploaded_csv = st.file_uploader("Upload CSV with one column: 'link'", type=["csv"])

    if st.button("Run Batch", disabled=not(api_key and uploaded_csv)):
        client = OpenAI(api_key=api_key)
        df = pd.read_csv(uploaded_csv)

        if df.shape[1] != 1 or "link" not in df.columns:
            st.error("CSV must have exactly one column named 'link'")
        else:
            out_rows = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                link = str(row["link"]).strip()
                if not link:
                    continue
                try:
                    # Download
                    response = requests.get(link)
                    response.raise_for_status()
                    tmp_path = tempfile.mktemp(suffix=".mp3")
                    with open(tmp_path, "wb") as f:
                        f.write(response.content)

                    # Transcribe + QC
                    with open(tmp_path, "rb") as f:
                        audio_file = f
                        raw_text, segments = transcribe_and_format(client, audio_file)

                    transcript_text = " ".join([seg["text"] for seg in segments])

                    # QC Eval
                    res = evaluate_with_openai(
                        client, model, transcript_text, batch_yaml,
                        0.0, use_responses_api=use_responses
                    )
                    analysis_text = json.dumps(res, ensure_ascii=False, indent=2)

                    out_rows.append({
                        "link": link,
                        "raw_transcript": raw_text,
                        "formatted_transcript": json.dumps(segments, ensure_ascii=False),
                        "analysis": analysis_text
                    })

                except Exception as e:
                    out_rows.append({
                        "link": link,
                        "raw_transcript": "",
                        "formatted_transcript": "",
                        "analysis": f"Error: {e}"
                    })

                progress.progress((i + 1) / len(df))

            out_df = pd.DataFrame(out_rows)
            st.success("Batch complete 🎉")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "⬇️ Download Results CSV",
                data=out_df.to_csv(index=False),
                file_name="qc_batch_results.csv",
                mime="text/csv"
            )


# ---------------------------
# Tab 3 - Settings & Notes
# ---------------------------
with tab3:
    st.subheader("Settings & Notes")
    st.markdown("""
- **Mixed transcription:** Hindi → Devanagari, English → Roman.
- **Annotations:** Pronunciation, tone, pace.
- **QC:** Config-driven via YAML in `config/`.
- **Batch mode:** Processes links from CSV.
- **Privacy:** API key only lives in session memory.
""")
