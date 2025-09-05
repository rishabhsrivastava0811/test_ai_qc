import streamlit as st
import pandas as pd
import json, os
from qc_evaluator import evaluate_with_openai, load_rubric
from openai import OpenAI

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="AI Moderated QC", layout="wide")
st.title("AI QC")

# ---------------------------
# Sidebar settings
# ---------------------------
with st.sidebar:
    st.header("🔐 OpenAI Settings")
    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Your key is kept in memory for this session only.")
    model = st.text_input("Model", value="gpt-4o-mini",
                          help="Use any chat-capable model available to your account.")
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
# Helper: Transcribe audio
# ---------------------------
def transcribe_audio(client, audio_file):
    try:
        audio_bytes = audio_file.read()
        tmp_path = f"/tmp/{audio_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        try:
            # Try GPT-4o transcription
            tr = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=open(tmp_path, "rb")
            )
        except Exception:
            # Fallback to whisper-1
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_path, "rb")
            )

        return tr.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ""


# ---------------------------
# Tab 1 - Single Call
# ---------------------------
with tab1:
    st.subheader("Single Call Evaluation")

    # Always initialize transcript safely
    transcript = st.session_state.get("transcript_text", "")

    audio_file = st.file_uploader(
        "Upload audio (mp3/wav/m4a) to transcribe",
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
        # ❌ Removed: st.json(rb)
    except Exception as e:
        st.error(f"Error loading YAML: {e}")
        use_yaml = None

    # Helper function: transcription
    def transcribe_audio(file, client):
        audio_bytes = file.read()
        tmp_path = f"/tmp/{file.name}"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        try:
            tr = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=open(tmp_path, "rb")
            )
        except Exception:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(tmp_path, "rb")
            )
        return tr.text

    # --- Transcribe Button ---
    if st.button("📝 Transcribe Audio", disabled=not audio_file):
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("Transcribing..."):
                    transcript = transcribe_audio(audio_file, client)
                st.session_state["transcript_text"] = transcript

                st.success("Transcription complete.")
                st.subheader("Transcript")
                st.text_area("Transcript", transcript, height=200)

            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # --- Evaluate Button ---
    if st.button("⚖️ Evaluate QC", type="primary", disabled= not (audio_file or transcript)):
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        elif not use_yaml:
            st.warning("No rubric YAML selected.")
        else:
            try:
                client = OpenAI(api_key=api_key)

                # If transcript not available, auto-generate from audio
                if not transcript and audio_file:
                    with st.spinner("Transcribing before QC..."):
                        transcript = transcribe_audio(audio_file, client)
                        st.session_state["transcript_text"] = transcript
                        st.success("Auto-transcription complete ✅")

                if not transcript:
                    st.warning("No transcript found. Please upload an audio file.")
                else:
                    with st.spinner("Scoring with GPT..."):
                        result = evaluate_with_openai(
                            client, model, transcript, use_yaml,
                            temperature, use_responses_api=use_responses
                        )

                    st.success("Done ✅")
                    st.subheader("Transcript Used for QC")
                    st.text_area("Transcript", transcript, height=200)

                    # Results
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

                    # st.subheader("Raw JSON")
                    # st.code(json.dumps(result, ensure_ascii=False, indent=2))

            except Exception as e:
                st.error(f"QC failed: {e}")


# ---------------------------
# Tab 2 - Batch
# ---------------------------
with tab2:
    st.subheader("Batch Evaluation")

    # --- Pick YAML Rubric ---
    selected_batch_yaml = st.selectbox("Choose QC Rubric (Batch)", options=yaml_files, index=0, key="batch_yaml")
    yaml_path = os.path.join(config_dir, selected_batch_yaml)
    with open(yaml_path, "r", encoding="utf-8") as f:
        batch_yaml = f.read()

    # --- Upload CSV ---
    uploaded_csv = st.file_uploader("Upload CSV with one column: 'link'", type=["csv"])

    if st.button("Run Batch", disabled=not(api_key and uploaded_csv)):
        try:
            import requests, tempfile

            client = OpenAI(api_key=api_key)
            df = pd.read_csv(uploaded_csv)

            # Validate CSV
            if df.shape[1] != 1:
                st.error("CSV must have exactly one column with header 'link'")
            elif "link" not in df.columns:
                st.error("First column header must be 'link'")
            else:
                out_rows = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    link = str(row["link"]).strip()
                    if not link:
                        continue

                    try:
                        # --- Step 1: Download audio file ---
                        response = requests.get(link)
                        response.raise_for_status()

                        tmp_path = tempfile.mktemp(suffix=".mp3")
                        with open(tmp_path, "wb") as f:
                            f.write(response.content)

                        # --- Step 2: Transcribe ---
                        try:
                            tr = client.audio.transcriptions.create(
                                model="gpt-4o-transcribe",
                                file=open(tmp_path, "rb")
                            )
                        except Exception:
                            tr = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=open(tmp_path, "rb")
                            )
                        transcript = tr.text

                        # --- Step 3: QC Analysis using YAML rubric ---
                        res = evaluate_with_openai(
                            client, model, transcript, batch_yaml,
                            0.0, use_responses_api=use_responses
                        )
                        analysis_text = json.dumps(res, ensure_ascii=False, indent=2)

                        out_rows.append({
                            "link": link,
                            "transcript": transcript,
                            "analysis": analysis_text
                        })

                    except Exception as e:
                        out_rows.append({
                            "link": link,
                            "transcript": "",
                            "analysis": f"Error: {e}"
                        })

                    progress.progress((i + 1) / len(df))

                # --- Step 4: Show + Export Results ---
                out_df = pd.DataFrame(out_rows)
                st.success("Batch complete 🎉")
                st.dataframe(out_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=out_df.to_csv(index=False),
                    file_name="qc_batch_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Batch failed: {e}")


# ---------------------------
# Tab 3 - Settings & Notes
# ---------------------------
with tab3:
    st.subheader("Settings & Notes")
    st.markdown("""
- **Privacy:** The API key is kept in memory only for this session unless you change the code.  
- **Latency & Cost:** Longer transcripts increase tokens. Consider summarizing or chunking.  
- **JSON Enforcement:** If your account supports the **Responses API** with `json_schema`, keep it on for rock-solid JSON outputs.  
- **Rubrics:** Just drop new `.yaml` files into the `config/` folder to make them available in the UI.  
- **Safety:** Always human-review red flags before actions.  
""")
