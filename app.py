import streamlit as st
import tempfile
import os
from predict import predict_audio

st.set_page_config(page_title="Deepfake Audio Detection", page_icon="🛡️", layout="centered")

# Corporate Styling via CSS
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
        color: #1A1A1A;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 4px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #002244;
        color: white;
    }
    h1, h2, h3 {
        color: #003366;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Audio Integrity Verification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 1.1em; margin-bottom: 2em;'>Upload audio files to detect AI-generated content and verify authenticity.</p>", unsafe_allow_html=True)

# Upload Section
st.markdown("### 📎 Upload File")
uploaded_file = st.file_uploader("Select a .wav, .flac, or .mp3 file to analyze", type=['wav', 'flac', 'mp3', 'ogg'], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("Start Analysis", use_container_width=True)
        
    if analyze_btn:
        with st.spinner("Processing audio signals..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            try:
                if not os.path.exists('deepfake_audio_model.pth'):
                    st.error("Model file 'deepfake_audio_model.pth' not found. Please train the model first.")
                else:
                    label, confidence = predict_audio(tmp_file_path, 'deepfake_audio_model.pth')
                    
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Report")
                    
                    # Display metrics in columns
                    metric_col1, metric_col2 = st.columns(2)
                    
                    is_genuine = "Genuine" in label
                    color = "normal" if is_genuine else "inverse"
                    
                    metric_col1.metric("Classification", "Human (Genuine)" if is_genuine else "AI (Deepfake)")
                    metric_col2.metric("Confidence Score", f"{confidence * 100:.2f}%")
                    
                    if is_genuine:
                        st.success("✅ **Verified**: The audio analysis indicates human speech.")
                    else:
                        st.error("⚠️ **Alert**: The audio analysis indicates synthetic/AI-generated speech.")
                        
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
