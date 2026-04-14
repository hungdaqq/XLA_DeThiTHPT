"""
Streamlit web application for image processing using main_dynamic_threshold.py logic
Supports uploading images and displaying processing results
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from main_dynamic_threshold import (
    find_corner_markers, warp, detect_grid_points, analyze_grid, visualize, TARGET_W, TARGET_H
)

# Configure Streamlit page
st.set_page_config(
    page_title="Nhận dạng phiếu trả lời",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Nhận dạng phiếu trả lời THPT Quốc gia")
st.markdown("Tải lên hình ảnh phiếu trả lời để phân tích các ô tích và trích xuất câu trả lời")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    debug_mode = st.checkbox("Chế Độ Debug (Hiển thị các bước trung gian)", value=True)
    
    st.markdown("---")
    st.markdown("### Ngưỡng Giá Trị")
    fill_ratio_phan1 = st.slider("PHẦN I", 0.05, 0.30, 0.10, 0.01)
    fill_ratio_phan2 = st.slider("PHẦN II", 0.05, 0.30, 0.10, 0.01)
    fill_ratio_phan3 = st.slider("PHẦN III", 0.05, 0.30, 0.15, 0.01)
    
    st.markdown("---")
    st.markdown("### Về Ứng Dụng")
    st.markdown("""
    Công cụ này phân tích phiếu trả lời bằng cách sử dụng hình ảnh hình thái học nhị phân:
    1. Phát hiện các điểm góc
    2. Áp dụng biến đổi phối cảnh
    3. Phát hiện các đường lưới thông qua hình thái học
    4. Phân tích các phần lưới với hình ảnh nhị phân
    5. Phát hiện các ô được tích
    6. Trích xuất câu trả lời
    """)

# Main content
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Tải Lên Hình Ảnh")
    uploaded_file = st.file_uploader("Chọn một tệp hình ảnh", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with col1:
        st.subheader("📸 Hình Ảnh Gốc")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Process image
    st.subheader("⏳ Đang Xử Lý...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Find corners
            status_text.text("Bước 1/5: Phát hiện các điểm góc...")
            progress_bar.progress(20)
            corners, corners_valid = find_corner_markers(image, debug_out=f"{temp_dir}/corners.jpg" if debug_mode else None)
            
            # Validate corner detection
            if not corners_valid:
                st.error("❌ **Lỗi: Không tìm thấy đủ điểm góc!** \n\nVui lòng tải lên một hình ảnh phiếu trả lời THPT Quốc gia hợp lệ.")
                st.stop()
            
            # Step 2: Warp perspective
            status_text.text("Bước 2/5: Áp dụng biến đổi phối cảnh...")
            progress_bar.progress(40)
            warped = warp(image, corners)
            
            # Step 3: Analyze grid
            status_text.text("Bước 3/5: Phân tích các phần lưới...")
            progress_bar.progress(60)
            grid_debug = detect_grid_points(warped, debug_prefix=f"{temp_dir}/grid" if debug_mode else None)
            grid_data = analyze_grid(warped, grid_debug)
            
            # Step 4: Visualize results
            status_text.text("Bước 4/5: Tạo hình ảnh trực quan...")
            progress_bar.progress(80)
            output_path = f"{temp_dir}/result.jpg"
            visualize(warped, grid_data, output_path)
            result_image = cv2.imread(output_path)
            
            # Step 5: Complete
            status_text.text("Bước 5/5: Hoàn thành!")
            progress_bar.progress(100)
            
            # Display results
            with col2:
                st.subheader("📊 Kết Quả Phân Tích")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("---")
            
            # Extract structured answers
            def trích_xuất_câu_trả_lời(grid_data):
                """Trích xuất câu trả lời từ dữ liệu lưới ở định dạng có cấu trúc"""
                # PHẦN I - Trắc nghiệm (0=A, 1=B, 2=C, 3=D)
                # Nếu có 2+ đáp án => "X" (invalid)
                fc = {str(i): [] for i in range(1, 41)}
                fc_invalid = set()  # Track invalid questions with multiple answers
                for item in grid_data.get("phan1", []):
                    q = item.get("q", 0)
                    if item.get("filled", False):
                        choice = item.get("choice", "")
                        choice_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(choice, -1)
                        if choice_idx >= 0:
                            fc[str(q)].append(choice_idx)
                
                # Mark questions with multiple answers as invalid
                for q_str in fc:
                    if len(fc[q_str]) > 1:
                        fc_invalid.add(q_str)
                        fc[q_str] = [-2]  # -2 means invalid/multiple answers
                
                # PHẦN II - Đúng/Sai với ánh xạ hàng
                # Ánh xạ câu+hàng sang chỉ số câu hỏi: 1a=1, 1b=2, 1c=3, 1d=4, 2a=5, 2b=6, vv.
                # 0 = Sai, 1 = Đúng
                tf = {}
                tf_invalid = set()  # Track invalid questions
                phan2_data = grid_data.get("phan2", [])
                
                # Count filled cells per question
                filled_per_question = {}
                for item in phan2_data:
                    cau_num = item.get("cau", 0)
                    row_char = item.get("row", "")
                    if item.get("filled", False):
                        key = (cau_num, row_char)
                        if key not in filled_per_question:
                            filled_per_question[key] = 0
                        filled_per_question[key] += 1
                
                # Build question index mapping
                question_idx = 1
                for cau_num in range(1, 9):  # 8 questions (2 cau per group)
                    for row_char in ["a", "b", "c", "d"]:
                        answers_for_question = []
                        for item in phan2_data:
                            if item.get("cau") == cau_num and item.get("row") == row_char and item.get("filled", False):
                                col = item.get("col", "")
                                choice_idx = 0 if col == "Sai" else 1  # 0=Sai, 1=Đúng
                                answers_for_question.append(choice_idx)
                        
                        if len(answers_for_question) > 1:
                            # Multiple answers for this row - mark as invalid
                            tf[str(question_idx)] = [-2]  # -2 = invalid
                            tf_invalid.add(str(question_idx))
                        elif len(answers_for_question) == 1:
                            tf[str(question_idx)] = answers_for_question
                        
                        question_idx += 1
                
                # PHẦN III - Chữ số (các ký tự đặc biệt: -, ,, 0-9)
                # Xuất chữ số theo thứ tự các cột (trái sang phải), không theo nhãn hàng
                dg = {str(i): "" for i in range(1, 7)}
                dg_invalid = set()  # Track invalid questions with multiple answers per position
                phan3_data = grid_data.get("phan3", [])
                
                for cau_num in range(1, 7):
                    # Group filled items by col (column position)
                    filled_by_col = {}
                    for item in phan3_data:
                        if item.get("cau") == cau_num and item.get("filled", False):
                            col = item.get("col", 0)
                            row = item.get("row", "")
                            if col not in filled_by_col:
                                filled_by_col[col] = []
                            filled_by_col[col].append(row)
                    
                    # Check if any column has multiple answers
                    has_invalid = any(len(v) > 1 for v in filled_by_col.values())
                    if has_invalid:
                        dg[str(cau_num)] = "X"  # Mark as invalid
                        dg_invalid.add(str(cau_num))
                    else:
                        # Sort by column and concatenate in order
                        digits = []
                        for col in sorted(filled_by_col.keys()):
                            if filled_by_col[col]:
                                digits.append(filled_by_col[col][0])
                        dg[str(cau_num)] = "".join(digits) if digits else ""
                
                # SBD - Student ID
                sbd = ""
                sbd_invalid = False
                sbd_data = grid_data.get("sobaodanh", [])
                if sbd_data:
                    digits_by_col = {}
                    for item in sbd_data:
                        col = item.get("col", 0)
                        if item.get("filled", False):
                            if col not in digits_by_col:
                                digits_by_col[col] = []
                            digits_by_col[col].append(item.get("digit", ""))
                    
                    # Check if any column has multiple digits
                    if any(len(v) > 1 for v in digits_by_col.values()):
                        sbd = "X"  # Mark as invalid
                        sbd_invalid = True
                    else:
                        sbd = "".join([digits_by_col.get(i, [None])[0] or "" for i in sorted(digits_by_col.keys())])
                
                # Mã Đề - Exam Code
                mdt = ""
                mdt_invalid = False
                made_data = grid_data.get("made", [])
                if made_data:
                    digits_by_col = {}
                    for item in made_data:
                        col = item.get("col", 0)
                        if item.get("filled", False):
                            if col not in digits_by_col:
                                digits_by_col[col] = []
                            digits_by_col[col].append(item.get("digit", ""))
                    
                    # Check if any column has multiple digits
                    if any(len(v) > 1 for v in digits_by_col.values()):
                        mdt = "X"  # Mark as invalid
                        mdt_invalid = True
                    else:
                        mdt = "".join([digits_by_col.get(i, [None])[0] or "" for i in sorted(digits_by_col.keys())])
                
                return {
                    "fc": fc,
                    "fc_invalid": fc_invalid,
                    "tf": tf,
                    "tf_invalid": tf_invalid,
                    "dg": dg,
                    "dg_invalid": dg_invalid,
                    "sbd": sbd,
                    "sbd_invalid": sbd_invalid,
                    "mdt": mdt,
                    "mdt_invalid": mdt_invalid
                }
            
            extracted = trích_xuất_câu_trả_lời(grid_data)
            
            # Display extracted answers
            st.subheader("📝 Câu Trả Lời Được Trích Xuất")
            
            # Create tabs for each section
            tab1, tab2, tab3, tab4, tab5, tab_json = st.tabs([
                "SBD (ID)", "Mã Đề", "Phần I", "Phần II", "Phần III", "JSON"
            ])
            
            with tab1:
                st.markdown("### Số Báo Danh (Student ID)")
                if extracted["sbd"]:
                    if extracted["sbd_invalid"]:
                        st.error(f"❌ **SBD: `{extracted['sbd']}`** ⚠️ (Có cột được khoanh 2 chữ số hoặc hơn - Không hợp lệ)")
                    else:
                        st.success(f"**SBD: `{extracted['sbd']}`** ✅")
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
                    st.info("Không tìm thấy dữ liệu SBD")
            
            with tab2:
                st.markdown("### Mã Đề (Exam Code)")
                if extracted["mdt"]:
                    if extracted["mdt_invalid"]:
                        st.error(f"❌ **Mã Đề: `{extracted['mdt']}`** ⚠️ (Có cột được khoanh 2 chữ số hoặc hơn - Không hợp lệ)")
                    else:
                        st.success(f"**Mã Đề: `{extracted['mdt']}`** ✅")
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
                    st.info("Không tìm thấy dữ liệu Mã Đề")
            
            with tab3:
                st.markdown("### Phần I - Trắc Nghiệm (40 câu)")
                fc_data = extracted["fc"]
                fc_invalid = extracted["fc_invalid"]
                
                if any(fc_data.values()):
                    # Show warning if there are invalid questions
                    if fc_invalid:
                        st.warning(f"⚠️ **{len(fc_invalid)} câu được khoanh 2 đáp án hoặc hơn (Không hợp lệ):** {', '.join(sorted(fc_invalid, key=int))}")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    choice_map = {0: "A", 1: "B", 2: "C", 3: "D", -2: "X"}
                    
                    for q_num in range(1, 41):
                        q_str = str(q_num)
                        answer_idx = fc_data[q_str][0] if fc_data[q_str] else -1
                        answer_letter = choice_map.get(answer_idx, "-")
                        
                        col_idx = (q_num - 1) % 4
                        cols = [col_a, col_b, col_c, col_d]
                        with cols[col_idx]:
                            if q_str in fc_invalid:
                                st.metric(f"Q{q_num}", answer_letter, "⚠️")
                            else:
                                st.metric(f"Q{q_num}", answer_letter)
                    
                    total_answers = sum(1 for v in fc_data.values() if v and v[0] >= 0)
                    st.info(f"Tổng cộng câu trả lời hợp lệ: {total_answers}/40")
                else:
                    st.info("Không phát hiện câu trả lời nào trong Phần I")
            
            with tab4:
                st.markdown("### Phần II - Đúng/Sai (32 câu)")
                tf_data = extracted["tf"]
                tf_invalid = extracted["tf_invalid"]
                choice_map = {0: "Sai", 1: "Đúng", -2: "X"}
                
                if any(tf_data.values()):
                    # Show warning if there are invalid questions
                    if tf_invalid:
                        st.warning(f"⚠️ **{len(tf_invalid)} câu được khoanh 2 hoặc hơn (Không hợp lệ):** {', '.join(sorted(tf_invalid, key=int))}")
                    
                    # Display in 4 columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for q_num in range(1, 33):
                        q_str = str(q_num)
                        answer_idx = tf_data.get(q_str, [None])[0] if tf_data.get(q_str) else -1
                        answer_text = choice_map.get(answer_idx, "-")
                        
                        col_idx = (q_num - 1) % 4
                        cols = [col1, col2, col3, col4]
                        with cols[col_idx]:
                            if q_str in tf_invalid:
                                st.metric(f"Q{q_num}", answer_text, "⚠️")
                            else:
                                st.metric(f"Q{q_num}", answer_text)
                    
                    total_answers = sum(1 for v in tf_data.values() if v and v[0] >= 0)
                    st.info(f"Tổng cộng câu trả lời hợp lệ: {total_answers}/32")
                else:
                    st.info("Không phát hiện câu trả lời nào trong Phần II")
            
            with tab5:
                st.markdown("### Phần III - Nhập Số (6 câu)")
                dg_data = extracted["dg"]
                dg_invalid = extracted["dg_invalid"]
                
                if any(dg_data.values()):
                    # Show warning if there are invalid questions
                    if dg_invalid:
                        st.warning(f"⚠️ **{len(dg_invalid)} câu có nhiều chữ số ở cùng vị trí (Không hợp lệ):** {', '.join(sorted(dg_invalid, key=int))}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for cau_num in range(1, 7):
                        cau_str = str(cau_num)
                        answer = dg_data[cau_str]
                        
                        col_idx = (cau_num - 1) % 3
                        cols = [col1, col2, col3]
                        with cols[col_idx]:
                            if cau_str in dg_invalid:
                                st.metric(f"Câu {cau_num}", answer if answer else "-", "⚠️")
                            else:
                                st.metric(f"Câu {cau_num}", answer if answer else "-")
                    
                    total_answers = sum(1 for v in dg_data.values() if v and v != "X")
                    st.info(f"Tổng cộng câu trả lời hợp lệ: {total_answers}/6")
                else:
                    st.info("Không phát hiện câu trả lời nào trong Phần III")
            
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
                st.subheader("🔍 Thông Tin Debug")
                
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    # Warped image
                    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    st.markdown("**Hình Ảnh Biến Đổi:**")
                    st.image(warped_rgb, use_container_width=True)
                
                with debug_col2:
                    # Corner detection
                    if os.path.exists(f"{temp_dir}/corners.jpg"):
                        corners_img = cv2.imread(f"{temp_dir}/corners.jpg")
                        st.markdown("**Phát Hiện Góc:**")
                        st.image(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Detailed statistics
                st.markdown("**Thống Kê:**")
                total_filled = sum(1 for section in grid_data.values() for item in section if item.get("filled"))
                total_cells = sum(len(section) for section in grid_data.values())
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Tổng Cộng Ô", total_cells)
                with stats_col2:
                    st.metric("Ô Được Tích", total_filled)
                with stats_col3:
                    st.metric("Tỷ Lệ Điền", f"{(total_filled/max(total_cells, 1)*100):.1f}%")
    
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý hình ảnh: {str(e)}")
        st.info("Vui lòng đảm bảo hình ảnh là một phiếu trả lời hợp lệ.")
else:
    st.info("👆 Tải lên một hình ảnh để bắt đầu")
    
    # Example section
    st.markdown("---")
    st.subheader("📖 Cách sử dụng:")
    st.markdown("""
    1. **Tải lên** một hình ảnh phiếu trả lời (JPG, PNG, BMP)
    2. **Cấu hình** các cài đặt trong thanh bên nếu cần
    3. **Xem** kết quả phân tích và các câu trả lời được trích xuất
    4. **Kiểm tra** thông tin gỡ lỗi để chẩn đoán chi tiết
    
    Công cụ sẽ:
    - Tự động phát hiện các điểm góc
    - Áp dụng sửa chữa phối cảnh
    - Phân tích từng phần lưới
    - Trích xuất các câu trả lời ở các ô đã tích
    - Hiển thị kết quả trong bảng điều khiển tương tác
    """)
