"""
Streamlit web application for image processing using main.py logic
Supports uploading images and displaying processing results
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from main import (
    find_corner_markers, warp, analyze_grid, visualize,
    detect_grid_points,
    TARGET_W, TARGET_H
)

# Configure Streamlit page
st.set_page_config(
    page_title="Answer Sheet Recognition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Answer Sheet Recognition")
st.markdown("Upload an answer sheet image to analyze bubbles and extract answers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    debug_mode = st.checkbox("Debug Mode (Show intermediate steps)", value=True)
    fill_threshold = st.slider("Fill Threshold (Phan 1-3)", 0.5, 1.0, 0.70, 0.05)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes answer sheets by:
    1. Detecting corner markers
    2. Applying perspective transformation
    3. Analyzing grid sections
    4. Detecting filled bubbles
    5. Extracting answers
    """)

# Main content
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Process image
    st.subheader("⏳ Processing...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    grid_debug = None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Find corners
            status_text.text("Step 1/5: Detecting corner markers...")
            progress_bar.progress(20)
            corners = find_corner_markers(image, debug_out=f"{temp_dir}/corners.jpg" if debug_mode else None)
            
            # Step 2: Warp perspective
            status_text.text("Step 2/5: Applying perspective transformation...")
            progress_bar.progress(40)
            warped = warp(image, corners)
            
            # Step 3: Analyze grid
            status_text.text("Step 3/5: Detecting grid lines & analyzing sections...")
            progress_bar.progress(60)
            grid_debug = detect_grid_points(warped, debug_prefix=f"{temp_dir}/grid" if debug_mode else None)
            grid_data = analyze_grid(warped)
            
            # Step 4: Visualize results
            status_text.text("Step 4/5: Generating visualization...")
            progress_bar.progress(80)
            output_path = f"{temp_dir}/result.jpg"
            visualize(warped, grid_data, output_path)
            result_image = cv2.imread(output_path)
            
            # Step 5: Complete
            status_text.text("Step 5/5: Complete!")
            progress_bar.progress(100)
            
            # Display results
            with col2:
                st.subheader("📊 Analysis Result")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("---")
            
            # Extract structured answers
            def extract_answers(grid_data):
                """Extract answers from grid data in structured format"""
                # Phần I - Multiple Choice (0=A, 1=B, 2=C, 3=D)
                fc = {str(i): [] for i in range(1, 41)}
                for item in grid_data.get("phan1", []):
                    q = item.get("q", 0)
                    if item.get("filled", False):
                        choice = item.get("choice", "")
                        choice_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(choice, -1)
                        if choice_idx >= 0:
                            fc[str(q)].append(choice_idx)
                
                # Phần II - True/False with row mapping
                # Map câu+row to question index: 1a=1, 1b=2, 1c=3, 1d=4, 2a=5, 2b=6, etc.
                # 0 = Sai (Wrong), 1 = Đúng (Correct)
                tf = {}
                phan2_data = grid_data.get("phan2", [])
                
                # Build mapping of (câu, row) to question index
                question_idx = 1
                for cau_num in range(1, 9):  # 8 questions
                    for row_char in ["a", "b", "c", "d"]:
                        for item in phan2_data:
                            if item.get("cau") == cau_num and item.get("row") == row_char and item.get("filled", False):
                                col = item.get("col", "")
                                choice_idx = 0 if col == "Sai" else 1  # 0=Sai, 1=Đúng
                                tf[str(question_idx)] = [choice_idx]
                                break
                        question_idx += 1
                
                # Phần III - Digits (special characters: -, ,, 0-9)
                # Output digits in order of columns (left to right), not by row label
                dg = {str(i): "" for i in range(1, 7)}
                phan3_data = grid_data.get("phan3", [])
                
                for cau_num in range(1, 7):
                    # Group filled items by col (column position)
                    filled_by_col = {}
                    for item in phan3_data:
                        if item.get("cau") == cau_num and item.get("filled", False):
                            col = item.get("col", 0)
                            row = item.get("row", "")
                            if col not in filled_by_col:
                                filled_by_col[col] = row
                    
                    # Sort by column and concatenate in order
                    digits = []
                    for col in sorted(filled_by_col.keys()):
                        digits.append(filled_by_col[col])
                    dg[str(cau_num)] = "".join(digits) if digits else ""
                
                # SBD - Student ID
                sbd = ""
                sbd_data = grid_data.get("sobaodanh", [])
                if sbd_data:
                    digits_by_col = {}
                    for item in sbd_data:
                        col = item.get("col", 0)
                        if item.get("filled", False):
                            if col not in digits_by_col:
                                digits_by_col[col] = item.get("digit", "")
                    sbd = "".join([digits_by_col.get(i, "") for i in sorted(digits_by_col.keys())])
                
                # Mã Đề - Exam Code
                mdt = ""
                made_data = grid_data.get("made", [])
                if made_data:
                    digits_by_col = {}
                    for item in made_data:
                        col = item.get("col", 0)
                        if item.get("filled", False):
                            if col not in digits_by_col:
                                digits_by_col[col] = item.get("digit", "")
                    mdt = "".join([digits_by_col.get(i, "") for i in sorted(digits_by_col.keys())])
                
                return {
                    "fc": fc,
                    "tf": tf,
                    "dg": dg,
                    "sbd": sbd,
                    "mdt": mdt
                }
            
            extracted = extract_answers(grid_data)
            
            # Display extracted answers
            st.subheader("📝 Extracted Answers")
            
            # Create tabs for each section
            tab1, tab2, tab3, tab4, tab5, tab_json = st.tabs([
                "SBD (ID)", "Mã Đề", "Phần I", "Phần II", "Phần III", "JSON"
            ])
            
            with tab1:
                st.markdown("### Số Báo Danh (Student ID)")
                if extracted["sbd"]:
                    st.success(f"**SBD: `{extracted['sbd']}`**")
                    sbd_data = grid_data.get("sobaodanh", [])
                    sbd_df = []
                    for item in sbd_data:
                        sbd_df.append({
                            "Digit": item.get("digit", ""),
                            "Column": item.get("col", ""),
                            "Filled": "✓" if item.get("filled", False) else "✗",
                            "Darkness": f"{item.get('mean_darkness', 0):.1f}"
                        })
                    st.dataframe(sbd_df, use_container_width=True)
                else:
                    st.info("No SBD data found")
            
            with tab2:
                st.markdown("### Mã Đề (Exam Code)")
                if extracted["mdt"]:
                    st.success(f"**Mã Đề: `{extracted['mdt']}`**")
                    made_data = grid_data.get("made", [])
                    made_df = []
                    for item in made_data:
                        made_df.append({
                            "Digit": item.get("digit", ""),
                            "Column": item.get("col", ""),
                            "Filled": "✓" if item.get("filled", False) else "✗",
                            "Darkness": f"{item.get('mean_darkness', 0):.1f}"
                        })
                    st.dataframe(made_df, use_container_width=True)
                else:
                    st.info("No Mã Đề data found")
            
            with tab3:
                st.markdown("### Phần I - Multiple Choice (40 questions)")
                fc_data = extracted["fc"]
                
                if any(fc_data.values()):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                    
                    for q_num in range(1, 41):
                        q_str = str(q_num)
                        answer_idx = fc_data[q_str][0] if fc_data[q_str] else -1
                        answer_letter = choice_map.get(answer_idx, "-")
                        
                        col_idx = (q_num - 1) % 4
                        cols = [col_a, col_b, col_c, col_d]
                        with cols[col_idx]:
                            st.metric(f"Q{q_num}", answer_letter)
                    
                    total_answers = sum(1 for v in fc_data.values() if v)
                    st.info(f"Total answers: {total_answers}/40")
                else:
                    st.info("No answers detected in Phần I")
            
            with tab4:
                st.markdown("### Phần II - True/False (32 questions)")
                tf_data = extracted["tf"]
                choice_map = {0: "Sai", 1: "Đúng"}
                
                if any(tf_data.values()):
                    # Display in 4 columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for q_num in range(1, 33):
                        q_str = str(q_num)
                        answer_idx = tf_data.get(q_str, [None])[0] if tf_data.get(q_str) else -1
                        answer_text = choice_map.get(answer_idx, "-")
                        
                        col_idx = (q_num - 1) % 4
                        cols = [col1, col2, col3, col4]
                        with cols[col_idx]:
                            st.metric(f"Q{q_num}", answer_text)
                    
                    total_answers = sum(1 for v in tf_data.values() if v)
                    st.info(f"Total answers: {total_answers}/32")
                else:
                    st.info("No answers detected in Phần II")
            
            with tab5:
                st.markdown("### Phần III - Number Input (6 questions)")
                dg_data = extracted["dg"]
                
                if any(dg_data.values()):
                    col1, col2, col3 = st.columns(3)
                    
                    for cau_num in range(1, 7):
                        cau_str = str(cau_num)
                        answer = dg_data[cau_str]
                        
                        col_idx = (cau_num - 1) % 3
                        cols = [col1, col2, col3]
                        with cols[col_idx]:
                            st.metric(f"Câu {cau_num}", answer if answer else "-")
                    
                    total_answers = sum(1 for v in dg_data.values() if v)
                    st.info(f"Total answers: {total_answers}/6")
                else:
                    st.info("No answers detected in Phần III")
            
            with tab_json:
                st.markdown("### JSON Output")
                import json
                output_json = {
                    "res": extracted,
                    "sbd": extracted["sbd"],
                    "mdt": extracted["mdt"]
                }
                st.json(output_json)
                
                # Code block for easy copying
                st.markdown("**Raw JSON:**")
                st.code(json.dumps([output_json], indent=2), language="json")
            
            # Display debug information if enabled
            if debug_mode:
                st.markdown("---")
                st.subheader("🔍 Debug Information")
                
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    # Warped image
                    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    st.markdown("**Warped Image:**")
                    st.image(warped_rgb, use_container_width=True)
                
                with debug_col2:
                    # Corner detection
                    if os.path.exists(f"{temp_dir}/corners.jpg"):
                        corners_img = cv2.imread(f"{temp_dir}/corners.jpg")
                        st.markdown("**Corner Detection:**")
                        st.image(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Detailed statistics
                st.markdown("**Statistics:**")
                total_filled = sum(1 for section in grid_data.values() for item in section if item.get("filled"))
                total_cells = sum(len(section) for section in grid_data.values())
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Total Cells", total_cells)
                with stats_col2:
                    st.metric("Filled Cells", total_filled)
                with stats_col3:
                    st.metric("Fill Rate", f"{(total_filled/max(total_cells, 1)*100):.1f}%")
                with stats_col4:
                    grid_point_count = len(grid_debug["points"]) if grid_debug else 0
                    st.metric("Grid Points", grid_point_count)

                st.markdown("**Morphological Grid Detection:**")
                if grid_debug:
                    grid_row1 = st.columns(3)
                    with grid_row1[0]:
                        st.caption("Threshold")
                        st.image(grid_debug["binary"], clamp=True, use_container_width=True)
                    with grid_row1[1]:
                        st.caption("Vertical Lines")
                        st.image(grid_debug["vertical"], clamp=True, use_container_width=True)
                    with grid_row1[2]:
                        st.caption("Horizontal Lines")
                        st.image(grid_debug["horizontal"], clamp=True, use_container_width=True)

                    grid_row2 = st.columns(2)
                    with grid_row2[0]:
                        st.caption("Intersections")
                        st.image(grid_debug["intersections"], clamp=True, use_container_width=True)
                    with grid_row2[1]:
                        st.caption("Grid Points Overlay")
                        points_rgb = cv2.cvtColor(grid_debug["points_overlay"], cv2.COLOR_BGR2RGB)
                        st.image(points_rgb, use_container_width=True)
                else:
                    st.info("Grid morphology data unavailable.")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please make sure the image is a valid answer sheet.")
else:
    st.info("👆 Upload an image to get started")
    
    # Example section
    st.markdown("---")
    st.subheader("📖 How to use:")
    st.markdown("""
    1. **Upload** an image of an answer sheet (JPG, PNG, BMP)
    2. **Configure** settings in the sidebar if needed
    3. **View** the analysis results and extracted answers
    4. **Check** debug information for detailed diagnostics
    
    The tool will:
    - Detect corner markers automatically
    - Apply perspective correction
    - Analyze each grid section
    - Extract filled bubble answers
    - Display results in an interactive dashboard
    """)
