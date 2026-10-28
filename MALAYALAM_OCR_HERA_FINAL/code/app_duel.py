import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time
import re

# Add paths for imports
project_root = "/mnt/Data_volume/Dhanya/project"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import HERA modules
try:
    import segmentation_v2 as seg_v2
    from char_embeddings import CharacterRecognizer, viterbi_decode, load_malayalam_lm
    import config
except ImportError as e:
    st.error(f"Import Error: {e}")

# Page Setup
st.set_page_config(page_title="HERA: Malayalam OCR Demo", layout="wide")

st.title("🛡️ HERA: Hybrid Error-Preserving Retrieval Architecture")
st.markdown("### The Final Evolution: Fine-tuned Vision Transformers + Viterbi Re-ranking")

# --- Initialize Resources ---
@st.cache_resource
def get_lm():
    return load_malayalam_lm()

@st.cache_resource
def get_recognizer(model_name):
    return CharacterRecognizer(model_name=model_name)

lm = get_lm()
fine_tuned_path = "/mnt/Data_volume/Dhanya/project/siglip_malayalam_finetuned/merged_model"
base_model_path = "google/siglip2-base-patch16-naflex"

recognizer_hera = get_recognizer(fine_tuned_path)
recognizer_base = get_recognizer(base_model_path)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload Word Image", type=["jpg", "jpeg", "png"])
manual_path = st.sidebar.text_input("OR Manual File Path", "")

if st.sidebar.button("🧹 Reset System Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared!")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Retrieval Settings")
n_candidates = st.sidebar.slider("Viterbi Candidates (K)", 1, 10, 5)
show_debug = st.sidebar.checkbox("Show Segmentation Debug", value=False)

# --- Image Loading Logic ---
final_img = None
if uploaded_file is not None:
    final_img = Image.open(uploaded_file).convert("RGB")
elif manual_path != "":
    if os.path.exists(manual_path):
        final_img = Image.open(manual_path).convert("RGB")
    else:
        st.sidebar.error("File path does not exist!")

# --- Main App Execution ---
if final_img is not None:
    st.image(final_img, caption="Target Image", width=400)
    
    # 1. Segmentation (HERA Standard: Water-fill Reservoir V2)
    with st.spinner("Executing Hierarchical Segmentation..."):
        # Default high-performance config
        cfg = seg_v2.SegmentationConfig(
            min_gap_fraction=0.015,
            threshold_fraction=0.15,
            merge_bridge_threshold=0.25
        )
        chunks, debug = seg_v2.segment_word_v2(final_img, config=cfg, return_debug=True)
    
    col1, col2 = st.columns(2)
    
    # --- Side A: Base Model (Zero-Shot) ---
    with col1:
        st.subheader("⚪ Base SigLIP (Zero-Shot)")
        st.info("Raw Vision-Language embeddings without domain fine-tuning or LM re-ranking.")
        
        start_time = time.time()
        pred_base = ""
        seg_cols = st.columns(min(len(chunks), 4))
        
        for i, chunk in enumerate(chunks):
            results = recognizer_base.recognize(chunk, n_results=1)
            char = "?"
            if results:
                char = re.sub(r'\(.*\)', '', results[0]['char_class'])
                pred_base += char
            with seg_cols[i % 4]:
                st.image(chunk, use_container_width=True)
                st.caption(f"'{char}'")
        
        duration = time.time() - start_time
        st.metric("Base Accuracy", f"{len(chunks)} chars", f"{duration:.2f}s")
        st.markdown(f"**Raw Output:** `{pred_base}`")

    # --- Side B: HERA FULL PIPELINE (Optimized) ---
    with col2:
        st.subheader("🔥 HERA Optimized Pipeline")
        st.success("Fine-tuned SigLIP-2 + Hybrid HOG + Viterbi Re-ranking")
        
        start_time = time.time()
        word_results = []
        for chunk in chunks:
            # Get top candidates for Viterbi
            results = recognizer_hera.recognize(chunk, n_results=n_candidates)
            word_results.append({'recognitions': results})
            
        # Viterbi Decoding
        final_chars, _ = viterbi_decode(word_results, lm)
        pred_hera = "".join(final_chars)
        pred_hera = re.sub(r'\(.*\)', '', pred_hera) # Clean labels
        
        seg_cols = st.columns(min(len(chunks), 4))
        for i, (chunk, char) in enumerate(zip(chunks, final_chars)):
            clean_char = re.sub(r'\(.*\)', '', char)
            with seg_cols[i % 4]:
                st.image(chunk, use_container_width=True)
                st.caption(f"'{clean_char}'")
                
        duration = time.time() - start_time
        st.metric("HERA Performance", "OPTIMIZED", f"{duration:.2f}s")
        st.markdown(f"**Final Integrated Output:**")
        st.code(pred_hera, language="")

    # --- Debug Information ---
    if show_debug:
        with st.expander("🔬 View HERA Segmentation Debug"):
            vis = seg_v2.visualise_segmentation(final_img, segments=debug['final_segments'], config=cfg)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Segmentation Boundaries")
            
            binary = debug['binary']
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
            st.image(heatmap, caption="Water-fill Pressure Map")

else:
    st.info("💡 Select a sample from the Grand Gold Standard below to see HERA in action!")
    
    # Show examples from Gold Standard
    st.markdown("---")
    st.markdown("### 🌟 Grand Gold Standard 200 (Elite Samples)")
    gold_dir = "/mnt/Data_volume/Dhanya/project/GRAND_GOLD_STANDARD_200"
    if os.path.exists(gold_dir):
        files = [f for f in os.listdir(gold_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:10]
        cols = st.columns(5)
        for i, f in enumerate(files):
            with cols[i % 5]:
                if st.button(f"Load {f}", key=f):
                    st.session_state['manual_path'] = os.path.join(gold_dir, f)
                st.image(os.path.join(gold_dir, f), use_container_width=True)

st.markdown("---")
st.caption("Developed by Dhanya et al. | HERA: A Hybrid Error-Preserving Retrieval Architecture")
