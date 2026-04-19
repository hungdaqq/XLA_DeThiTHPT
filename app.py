"""
Streamlit web application using the same detection/decoding logic as datetime.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

import detect as core


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _process_image_with_main_logic(
    image: np.ndarray,
    fill_ratio_phan1: float,
    fill_ratio_phan2: float,
    fill_ratio_phan3: float,
    debug_prefix: Optional[str],
) -> Dict[str, object]:
    def _run_detection_pipeline(
        src_img: np.ndarray,
        prefix: Optional[str],
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        data_local = core.detect_boxes_from_morph_lines(
            src_img,
            vertical_scale=0.015,
            horizontal_scale=0.015,
            min_line_length=50,
            align_vertical_rows=True,
            vertical_row_tolerance=10,
            block_size=35,
            block_offset=7,
            min_box_area=200,
            min_box_width=15,
            min_box_height=15,
            close_kernel_size=3,
            debug_prefix=prefix,
        )
        parts_local = core.group_boxes_into_parts(data_local["boxes"], row_tolerance=30)
        return data_local, parts_local

    def _preprocess_clahe(src_img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _parts_score(parts_local: Dict[str, object], page_h: int) -> float:
        p1 = len(parts_local["part_i"])
        p2 = len(parts_local["part_ii"])
        p3 = len(parts_local["part_iii"])
        score = 0.0
        score += 3.0 if p1 == 4 else (p1 / 4.0)
        score += 4.0 * min(1.0, p2 / 8.0)
        score += 3.0 * min(1.0, p3 / 6.0)

        if p3 and page_h > 0:
            y_vals = [cv2.boundingRect(b)[1] for b in parts_local["part_iii"]]
            p3_ratio = float(np.mean(y_vals)) / float(page_h)
            score += max(0.0, min(1.0, (p3_ratio - 0.55) / 0.20))
        return score

    data, parts = _run_detection_pipeline(image, debug_prefix)

    preprocess_mode = "base"
    page_h = image.shape[0] if image is not None else 0

    if len(parts["part_ii"]) < 8 or len(parts["part_iii"]) < 6:
        img_clahe = _preprocess_clahe(image)
        data_clahe, parts_clahe = _run_detection_pipeline(img_clahe, None)

        base_score = _parts_score(parts, page_h)
        clahe_score = _parts_score(parts_clahe, page_h)
        if clahe_score > base_score + 0.05:
            data = data_clahe
            parts = parts_clahe
            preprocess_mode = "clahe"

    part_box_set = set(id(box) for box in parts["all_parts"])
    remaining_boxes = [box for box in data["boxes"] if id(box) not in part_box_set]

    remaining_for_upper = core._split_merged_boxes_for_grouping(
        remaining_boxes,
        split_wide=True,
        split_tall=False,
    )

    sbd_candidates, ma_de_candidates, split_x = core._separate_upper_id_boxes(
        remaining_for_upper,
        parts["part_i"],
    )

    sobao_danh = core.detect_sobao_danh_boxes(
        sbd_candidates,
        boxes_per_row=6,
        max_rows=10,
        row_tolerance=30,
        size_tolerance_ratio=0.35,
        debug=False,
    )

    remaining_for_ma_de = core._split_merged_boxes_for_grouping(
        ma_de_candidates,
        split_wide=False,
        split_tall=True,
    )
    ma_de = core.detect_ma_de_boxes(
        remaining_for_ma_de,
        boxes_per_row=3,
        max_rows=10,
        row_tolerance=20,
        size_tolerance_ratio=0.3,
        debug=False,
    )

    # Keep the same MaDe completion behavior as detect.py.
    if ma_de["row_count"] < 10 and ma_de["ma_de_rows"] and sobao_danh["sobao_danh_rows"]:
        ref_positions = []
        for row in sobao_danh["sobao_danh_rows"][:10]:
            ys = [cv2.boundingRect(box)[1] for box in row]
            if ys:
                ref_positions.append(int(round(float(np.mean(ys)))))

        ma_de_rect_rows = []
        for row in ma_de["ma_de_rows"]:
            rects = [cv2.boundingRect(box) for box in row]
            rects = sorted(rects, key=lambda r: r[0])
            if len(rects) == 3:
                ma_de_rect_rows.append(rects)

        if len(ref_positions) == 10 and ma_de_rect_rows:
            col_x = [int(round(float(np.median([rects[c][0] for rects in ma_de_rect_rows])))) for c in range(3)]
            col_w = [int(round(float(np.median([rects[c][2] for rects in ma_de_rect_rows])))) for c in range(3)]
            row_h = int(round(float(np.median([rects[0][3] for rects in ma_de_rect_rows]))))
            row_h = max(1, row_h)

            detected_rows_y = [int(round(float(np.mean([r[1] for r in rects])))) for rects in ma_de_rect_rows]
            align_tolerance = max(35, int(round(row_h * 1.2)))
            aligned_rows: List[Optional[List[np.ndarray]]] = [None] * 10

            used_ref_indices = set()
            for rects, row_y in sorted(zip(ma_de_rect_rows, detected_rows_y), key=lambda t: t[1]):
                candidates = sorted(range(10), key=lambda idx: abs(ref_positions[idx] - row_y))
                chosen_idx = None
                for idx in candidates:
                    if idx in used_ref_indices:
                        continue
                    if abs(ref_positions[idx] - row_y) <= align_tolerance:
                        chosen_idx = idx
                        break
                if chosen_idx is None:
                    continue

                row_polys: List[np.ndarray] = []
                for rx, ry, rw, rh in rects:
                    row_polys.append(
                        np.array(
                            [
                                [[rx, ry]],
                                [[rx + rw, ry]],
                                [[rx + rw, ry + rh]],
                                [[rx, ry + rh]],
                            ],
                            dtype=np.int32,
                        )
                    )
                aligned_rows[chosen_idx] = row_polys
                used_ref_indices.add(chosen_idx)

            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx]
                synthetic_row: List[np.ndarray] = []
                for c in range(3):
                    x = col_x[c]
                    w = max(1, col_w[c])
                    synthetic_row.append(
                        np.array(
                            [
                                [[x, y_ref]],
                                [[x + w, y_ref]],
                                [[x + w, y_ref + row_h]],
                                [[x, y_ref + row_h]],
                            ],
                            dtype=np.int32,
                        )
                    )
                aligned_rows[idx] = synthetic_row

            ma_de_completed_rows = [row for row in aligned_rows if row is not None]
            if len(ma_de_completed_rows) == 10:
                ma_de["ma_de_rows"] = ma_de_completed_rows
                ma_de["ma_de"] = [box for row in ma_de_completed_rows for box in row]
                ma_de["row_count"] = 10

    combined_results = {
        "sobao_danh_rows": sobao_danh["sobao_danh_rows"],
        "ma_de_rows": ma_de["ma_de_rows"],
    }
    extrapolated = core.extrapolate_missing_rows(combined_results, target_rows=10, debug=False)

    gray_for_digits = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobao_rows_aligned = extrapolated.get("sobao_danh_rows_aligned", sobao_danh["sobao_danh_rows"])
    ma_de_rows_aligned = extrapolated.get("ma_de_rows_aligned", ma_de["ma_de_rows"])

    sbd_digits = core.evaluate_digit_rows_mean_darkness(
        gray_for_digits,
        sobao_rows_aligned,
        expected_cols=6,
    )
    made_digits = core.evaluate_digit_rows_mean_darkness(
        gray_for_digits,
        ma_de_rows_aligned,
        expected_cols=3,
    )

    overlay = image.copy()
    part_configs = [
        ("part_i", "Part I (4)", (0, 255, 0)),
        ("part_ii", "Part II (8)", (0, 165, 255)),
        ("part_iii", "Part III (6)", (255, 0, 0)),
    ]

    for part_key, label, color in part_configs:
        for poly in parts[part_key]:
            cv2.polylines(overlay, [poly], True, color, 3)

        if parts[part_key]:
            min_y = min(cv2.boundingRect(p)[1] for p in parts[part_key])
            cv2.putText(
                overlay,
                label,
                (50, min_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                2,
            )

    sobao_color = (255, 128, 0)
    for row in sobao_danh["sobao_danh_rows"]:
        for poly in row:
            cv2.polylines(overlay, [poly], True, sobao_color, 2)

    ma_de_color = (255, 255, 0)
    for row in ma_de["ma_de_rows"]:
        for poly in row:
            cv2.polylines(overlay, [poly], True, ma_de_color, 2)

    combined_grid_image = image.copy()
    binary_threshold = data.get("binary")

    part_i_evals: List[Dict[str, object]] = []
    if parts["part_i"]:
        grid_result = core.extract_grid_from_boxes(
            combined_grid_image,
            boxes=parts["part_i"],
            grid_cols=4,
            grid_rows=10,
            start_offset_ratio_x=0.2,
            start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.015,
            end_offset_ratio_y=0.015,
            grid_color=(0, 255, 0),
            grid_thickness=1,
        )
        combined_grid_image = grid_result["image_with_grid"]

        if binary_threshold is not None:
            part_i_evals = core.evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result["grid_info"],
                fill_ratio_thresh=float(fill_ratio_phan1),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    part_ii_evals: List[Dict[str, object]] = []
    if parts["part_ii"]:
        offset_ratios = []
        end_offset_x = []
        end_offset_y = []
        for box_idx in range(len(parts["part_ii"])):
            if box_idx % 2 == 0:
                offset_ratios.append((0.3, 0.33))
            else:
                offset_ratios.append((0.0, 0.33))
            end_offset_x.append(0.0)
            end_offset_y.append(0.03)

        grid_result_ii = core.extract_grid_from_boxes_variable_offsets(
            combined_grid_image,
            boxes=parts["part_ii"],
            grid_cols=2,
            grid_rows=4,
            start_offset_ratios=offset_ratios,
            end_offset_ratios_x=end_offset_x,
            end_offset_ratios_y=end_offset_y,
            grid_color=(0, 165, 255),
            grid_thickness=1,
        )
        combined_grid_image = grid_result_ii["image_with_grid"]

        if binary_threshold is not None:
            part_ii_evals = core.evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result_ii["grid_info"],
                fill_ratio_thresh=float(fill_ratio_phan2),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    part_iii_evals: List[Dict[str, object]] = []
    if parts["part_iii"]:
        custom_pattern = [[0], [1, 2]] + [[0, 1, 2, 3] for _ in range(10)]
        grid_result_iii = core.extract_grid_from_boxes_custom_pattern(
            combined_grid_image,
            boxes=parts["part_iii"],
            grid_cols=4,
            grid_rows=12,
            start_offset_ratio_x=0.22,
            start_offset_ratio_y=0.16,
            end_offset_ratio_x=0.1,
            end_offset_ratio_y=0.015,
            grid_color=(255, 0, 0),
            grid_thickness=1,
            row_col_patterns=custom_pattern,
        )
        combined_grid_image = grid_result_iii["image_with_grid"]

        if binary_threshold is not None:
            part_iii_evals = core.evaluate_grid_fill_from_binary(
                binary_image=binary_threshold,
                grid_info=grid_result_iii["grid_info"],
                fill_ratio_thresh=float(fill_ratio_phan3),
                inner_margin_ratio=0.05,
                mask_mode="hough-circle",
                circle_radius_scale=0.5,
                circle_border_exclude_ratio=0.0,
            )

    if sobao_danh["sobao_danh_rows"]:
        for row in sobao_danh["sobao_danh_rows"]:
            for poly in row:
                cv2.polylines(combined_grid_image, [poly], True, (255, 128, 0), 1)

    if ma_de["ma_de_rows"]:
        for row in ma_de["ma_de_rows"]:
            for poly in row:
                cv2.polylines(combined_grid_image, [poly], True, (255, 255, 0), 1)

    combined_grid_image = core.draw_digit_darkness_overlay(
        combined_grid_image,
        sbd_digits,
        color=(0, 220, 255),
        alpha=0.40,
    )
    combined_grid_image = core.draw_digit_darkness_overlay(
        combined_grid_image,
        made_digits,
        color=(0, 255, 255),
        alpha=0.40,
    )

    all_evals = part_i_evals + part_ii_evals + part_iii_evals
    if all_evals:
        combined_grid_image = core.draw_filled_cells_overlay(
            combined_grid_image,
            all_evals,
            color=(0, 255, 0),
            alpha=0.35,
        )

    return {
        "preprocess_mode": preprocess_mode,
        "data": data,
        "parts": parts,
        "sobao_danh": sobao_danh,
        "ma_de": ma_de,
        "split_x": split_x,
        "extrapolated": extrapolated,
        "sbd_digits": sbd_digits,
        "made_digits": made_digits,
        "part_i_evals": part_i_evals,
        "part_ii_evals": part_ii_evals,
        "part_iii_evals": part_iii_evals,
        "parts_overlay": overlay,
        "result_image": combined_grid_image,
        "binary_threshold": binary_threshold,
    }


def _build_structured_answers(results: Dict[str, object]) -> Dict[str, object]:
    part_i_evals = results["part_i_evals"]
    part_ii_evals = results["part_ii_evals"]
    part_iii_evals = results["part_iii_evals"]
    sbd_digits = results["sbd_digits"]
    made_digits = results["made_digits"]

    fc: Dict[str, List[int]] = {str(i): [] for i in range(1, 41)}
    fc_invalid = set()
    for item in part_i_evals:
        if not bool(item.get("filled", False)):
            continue
        q = (int(item.get("box_idx", -1)) * 10) + int(item.get("row", -1)) + 1
        if 1 <= q <= 40:
            fc[str(q)].append(int(item.get("col", -1)))

    for q_str, choices in fc.items():
        if len(choices) > 1:
            fc_invalid.add(q_str)
            fc[q_str] = [-2]

    tf: Dict[str, List[int]] = {str(i): [] for i in range(1, 33)}
    tf_invalid = set()
    for item in part_ii_evals:
        if not bool(item.get("filled", False)):
            continue
        box_idx = int(item.get("box_idx", -1))
        row = int(item.get("row", -1))
        col = int(item.get("col", -1))
        q = (box_idx * 4) + row + 1
        if 1 <= q <= 32 and col in (0, 1):
            tf[str(q)].append(col)

    for q_str, answers in tf.items():
        if len(answers) > 1:
            tf_invalid.add(q_str)
            tf[q_str] = [-2]

    row_labels = ["-", ","] + [str(i) for i in range(10)]
    dg: Dict[str, str] = {str(i): "" for i in range(1, 7)}
    dg_invalid = set()

    for cau_num in range(1, 7):
        filled_by_col: Dict[int, List[str]] = {}
        for item in part_iii_evals:
            if not bool(item.get("filled", False)):
                continue
            if int(item.get("box_idx", -1)) + 1 != cau_num:
                continue

            col = int(item.get("col", -1))
            row = int(item.get("row", -1))
            if col < 0 or row < 0 or row >= len(row_labels):
                continue
            filled_by_col.setdefault(col, []).append(row_labels[row])

        has_invalid = any(len(v) > 1 for v in filled_by_col.values())
        if has_invalid:
            dg[str(cau_num)] = "X"
            dg_invalid.add(str(cau_num))
            continue

        digits = []
        for col in sorted(filled_by_col.keys()):
            if filled_by_col[col]:
                digits.append(filled_by_col[col][0])
        dg[str(cau_num)] = "".join(digits) if digits else ""

    sbd = str(sbd_digits.get("decoded", ""))
    mdt = str(made_digits.get("decoded", ""))
    sbd_invalid = "?" in sbd
    mdt_invalid = "?" in mdt

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
        "mdt_invalid": mdt_invalid,
    }


def _digit_eval_table(result: Dict[str, object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    evaluations = result.get("evaluations", [])
    if not isinstance(evaluations, list):
        return out

    for item in sorted(evaluations, key=lambda x: (int(x.get("col", -1)), int(x.get("row", -1)))):
        out.append(
            {
                "Digit": str(item.get("row", "")),
                "Column": int(item.get("col", -1)),
                "Filled": "✓" if bool(item.get("filled", False)) else "✗",
                "Darkness": f"{float(item.get('mean_darkness', 0.0)):.1f}",
            }
        )
    return out


def _build_json_payload(extracted: Dict[str, object]) -> Dict[str, object]:
    return {
        "res": {
            "fc": extracted["fc"],
            "fc_invalid": sorted(extracted["fc_invalid"], key=int),
            "tf": extracted["tf"],
            "tf_invalid": sorted(extracted["tf_invalid"], key=int),
            "dg": extracted["dg"],
            "dg_invalid": sorted(extracted["dg_invalid"], key=int),
            "sbd": extracted["sbd"],
            "sbd_invalid": extracted["sbd_invalid"],
            "mdt": extracted["mdt"],
            "mdt_invalid": extracted["mdt_invalid"],
        },
        "sbd": extracted["sbd"],
        "mdt": extracted["mdt"],
    }


def _build_batch_summary_row(
    file_name: str,
    status: str,
    results: Optional[Dict[str, object]],
    extracted: Optional[Dict[str, object]],
    error_message: str = "",
) -> Dict[str, object]:
    if status != "OK" or results is None or extracted is None:
        return {
            "File": file_name,
            "Status": status,
            "SBD": "",
            "Mã đề": "",
            "Part I": "",
            "Part II": "",
            "Part III": "",
            "Boxes": "",
            "Preprocess": "",
            "Error": error_message,
        }

    fc_valid = sum(1 for v in extracted["fc"].values() if v and v[0] >= 0)
    tf_valid = sum(1 for v in extracted["tf"].values() if v and v[0] >= 0)
    dg_valid = sum(1 for v in extracted["dg"].values() if v and v != "X")

    return {
        "File": file_name,
        "Status": status,
        "SBD": extracted["sbd"],
        "Mã đề": extracted["mdt"],
        "Part I": f"{fc_valid}/40",
        "Part II": f"{tf_valid}/32",
        "Part III": f"{dg_valid}/6",
        "Boxes": len(results["data"]["boxes"]),
        "Preprocess": results["preprocess_mode"],
        "Error": "",
    }


def _render_detailed_result(
    results: Dict[str, object],
    extracted: Dict[str, object],
    debug_mode: bool,
) -> None:
    st.markdown("---")
    st.subheader("📝 Câu Trả Lời Được Trích Xuất")

    tab1, tab2, tab3, tab4, tab5, tab_json = st.tabs(
        ["SBD (ID)", "Mã Đề", "Phần I", "Phần II", "Phần III", "JSON"]
    )

    with tab1:
        st.markdown("### Số Báo Danh")
        if extracted["sbd"]:
            if extracted["sbd_invalid"]:
                st.warning(f"SBD: {extracted['sbd']} (chưa chắc chắn hoàn toàn)")
            else:
                st.success(f"SBD: {extracted['sbd']}")
            st.dataframe(_digit_eval_table(results["sbd_digits"]), use_container_width=True)
        else:
            st.info("Không tìm thấy dữ liệu SBD")

    with tab2:
        st.markdown("### Mã Đề")
        if extracted["mdt"]:
            if extracted["mdt_invalid"]:
                st.warning(f"Mã Đề: {extracted['mdt']} (chưa chắc chắn hoàn toàn)")
            else:
                st.success(f"Mã Đề: {extracted['mdt']}")
            st.dataframe(_digit_eval_table(results["made_digits"]), use_container_width=True)
        else:
            st.info("Không tìm thấy dữ liệu Mã Đề")

    with tab3:
        st.markdown("### Phần I - Trắc Nghiệm (40 câu)")
        fc_data = extracted["fc"]
        fc_invalid = extracted["fc_invalid"]
        choice_map = {0: "A", 1: "B", 2: "C", 3: "D", -2: "X"}

        if any(fc_data.values()):
            if fc_invalid:
                st.warning(
                    f"⚠️ {len(fc_invalid)} câu có nhiều đáp án: {', '.join(sorted(fc_invalid, key=int))}"
                )

            cols = st.columns(4)
            for q_num in range(1, 41):
                q_str = str(q_num)
                answer_idx = fc_data[q_str][0] if fc_data[q_str] else -1
                answer_letter = choice_map.get(answer_idx, "-")
                col_idx = (q_num - 1) % 4
                with cols[col_idx]:
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
            if tf_invalid:
                st.warning(
                    f"⚠️ {len(tf_invalid)} câu có nhiều đáp án: {', '.join(sorted(tf_invalid, key=int))}"
                )

            cols = st.columns(4)
            for q_num in range(1, 33):
                q_str = str(q_num)
                answer_idx = tf_data[q_str][0] if tf_data[q_str] else -1
                answer_text = choice_map.get(answer_idx, "-")
                col_idx = (q_num - 1) % 4
                with cols[col_idx]:
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
            if dg_invalid:
                st.warning(
                    f"⚠️ {len(dg_invalid)} câu không hợp lệ: {', '.join(sorted(dg_invalid, key=int))}"
                )

            cols = st.columns(3)
            for cau_num in range(1, 7):
                cau_str = str(cau_num)
                answer = dg_data[cau_str]
                col_idx = (cau_num - 1) % 3
                with cols[col_idx]:
                    st.metric(f"Câu {cau_num}", answer if answer else "-")

            total_answers = sum(1 for v in dg_data.values() if v and v != "X")
            st.info(f"Tổng cộng câu trả lời hợp lệ: {total_answers}/6")
        else:
            st.info("Không phát hiện câu trả lời nào trong Phần III")

    with tab_json:
        st.markdown("### JSON Output")
        json_payload = _build_json_payload(extracted)
        st.json(json_payload)
        st.code(json.dumps([json_payload], indent=2, ensure_ascii=False), language="json")

    if debug_mode:
        st.markdown("---")
        st.subheader("🔍 Thông Tin Debug")

        debug_col1, debug_col2 = st.columns(2)
        with debug_col1:
            st.markdown("**Overlay Parts**")
            st.image(_to_rgb(results["parts_overlay"]), use_container_width=True)
        with debug_col2:
            if results["binary_threshold"] is not None:
                st.markdown("**Binary Threshold**")
                st.image(_to_rgb(results["binary_threshold"]), use_container_width=True)
            else:
                st.markdown("**Boxes Overlay**")
                st.image(_to_rgb(results["data"]["boxes_overlay"]), use_container_width=True)

        part_i_evals = results["part_i_evals"]
        part_ii_evals = results["part_ii_evals"]
        part_iii_evals = results["part_iii_evals"]
        total_cells = len(part_i_evals) + len(part_ii_evals) + len(part_iii_evals)
        total_filled = sum(1 for e in (part_i_evals + part_ii_evals + part_iii_evals) if e.get("filled", False))

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Detected Boxes", len(results["data"]["boxes"]))
        with stats_col2:
            st.metric("Tổng Cộng Ô", total_cells)
        with stats_col3:
            pct = (total_filled / max(1, total_cells)) * 100.0
            st.metric("Tỷ Lệ Điền", f"{pct:.1f}%")

        st.caption(
            f"Preprocess: {results['preprocess_mode']} | "
            f"Part I/II/III: {len(results['parts']['part_i'])}/"
            f"{len(results['parts']['part_ii'])}/{len(results['parts']['part_iii'])} | "
            f"Upper split X: {float(results['split_x']):.1f}"
        )


st.set_page_config(
    page_title="Nhận dạng phiếu trả lời",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Nhận dạng phiếu trả lời THPT Quốc gia")
st.markdown("Ứng dụng này chạy cùng pipeline nhận dạng/giải mã như detect.py")

with st.sidebar:
    st.header("⚙️ Cấu Hình")
    debug_mode = st.checkbox("Chế Độ Debug", value=True)

    st.markdown("---")
    st.markdown("### Ngưỡng Fill Ratio")
    fill_ratio_phan1 = st.slider("PHẦN I", 0.30, 0.95, 0.55, 0.01)
    fill_ratio_phan2 = st.slider("PHẦN II", 0.30, 0.95, 0.55, 0.01)
    fill_ratio_phan3 = st.slider("PHẦN III", 0.30, 0.95, 0.55, 0.01)

    st.markdown("---")
    st.markdown("### Ghi chú")
    st.caption("Luồng xử lý: detect box -> group part -> decode SBD/Mã đề -> grid fill ratio -> trích xuất đáp án")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Tải Lên Hình Ảnh")
    uploaded_files = st.file_uploader(
        "Chọn 1 hoặc nhiều ảnh phiếu",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    process_clicked = st.button(
        "🚀 Bắt đầu xử lý",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

BATCH_RESULTS_KEY = "batch_results"
BATCH_SIGNATURE_KEY = "batch_signature"

if BATCH_RESULTS_KEY not in st.session_state:
    st.session_state[BATCH_RESULTS_KEY] = []
if BATCH_SIGNATURE_KEY not in st.session_state:
    st.session_state[BATCH_SIGNATURE_KEY] = ()

current_signature: Tuple[Tuple[str, int], ...] = ()
if uploaded_files:
    current_signature = tuple((f.name, int(getattr(f, "size", 0))) for f in uploaded_files)

if process_clicked and uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_results: List[Dict[str, object]] = []
    total_files = len(uploaded_files)

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, uploaded in enumerate(uploaded_files, start=1):
            file_name = uploaded.name
            status_text.text(f"Đang xử lý {idx}/{total_files}: {file_name}")
            progress_bar.progress(int(((idx - 1) / max(1, total_files)) * 100))

            try:
                file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Không đọc được ảnh")

                debug_prefix = None
                if debug_mode:
                    stem = Path(file_name).stem
                    debug_prefix = str(Path(temp_dir) / f"grid_{stem}")

                results = _process_image_with_main_logic(
                    image,
                    fill_ratio_phan1=fill_ratio_phan1,
                    fill_ratio_phan2=fill_ratio_phan2,
                    fill_ratio_phan3=fill_ratio_phan3,
                    debug_prefix=debug_prefix,
                )
                extracted = _build_structured_answers(results)

                batch_results.append(
                    {
                        "file_name": file_name,
                        "status": "OK",
                        "error": "",
                        "image": image,
                        "results": results,
                        "extracted": extracted,
                        "summary": _build_batch_summary_row(file_name, "OK", results, extracted),
                    }
                )
            except Exception as err:
                batch_results.append(
                    {
                        "file_name": file_name,
                        "status": "ERROR",
                        "error": str(err),
                        "image": None,
                        "results": None,
                        "extracted": None,
                        "summary": _build_batch_summary_row(
                            file_name,
                            "ERROR",
                            None,
                            None,
                            error_message=str(err),
                        ),
                    }
                )

    progress_bar.progress(100)
    status_text.text(f"Hoàn thành xử lý {len(uploaded_files)} ảnh")
    st.session_state[BATCH_RESULTS_KEY] = batch_results
    st.session_state[BATCH_SIGNATURE_KEY] = current_signature

has_batch_results = (
    bool(uploaded_files)
    and bool(st.session_state[BATCH_RESULTS_KEY])
    and st.session_state[BATCH_SIGNATURE_KEY] == current_signature
)

if has_batch_results:
    batch_results = st.session_state[BATCH_RESULTS_KEY]
    summary_rows = [item["summary"] for item in batch_results]
    success_items = [item for item in batch_results if item["status"] == "OK"]
    error_items = [item for item in batch_results if item["status"] != "OK"]

    st.markdown("---")
    st.subheader("📋 Kết Quả Tổng Hợp")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Tổng số ảnh", len(batch_results))
    with metric_col2:
        st.metric("Thành công", len(success_items))
    with metric_col3:
        st.metric("Lỗi", len(error_items))

    st.dataframe(summary_rows, use_container_width=True)

    combined_payload = []
    for item in success_items:
        payload = _build_json_payload(item["extracted"])
        payload["file_name"] = item["file_name"]
        combined_payload.append(payload)

    st.download_button(
        label="⬇️ Tải JSON tổng hợp",
        data=json.dumps(combined_payload, indent=2, ensure_ascii=False),
        file_name="ket_qua_tong_hop.json",
        mime="application/json",
        use_container_width=True,
        disabled=not combined_payload,
    )

    if success_items:
        selected_file_name = st.selectbox(
            "Chọn ảnh để xem chi tiết",
            options=[item["file_name"] for item in success_items],
        )
        selected_item = next(item for item in success_items if item["file_name"] == selected_file_name)

        with col1:
            st.subheader("📸 Hình Ảnh Gốc")
            st.image(_to_rgb(selected_item["image"]), use_container_width=True)

        with col2:
            st.subheader("📊 Kết Quả Phân Tích")
            st.image(_to_rgb(selected_item["results"]["result_image"]), use_container_width=True)

        _render_detailed_result(
            selected_item["results"],
            selected_item["extracted"],
            debug_mode=debug_mode,
        )
    else:
        st.error("Không có ảnh nào xử lý thành công để hiển thị chi tiết.")
elif uploaded_files:
    st.info("Nhấn '🚀 Bắt đầu xử lý' để chạy batch và xem tiến độ/kết quả tổng hợp.")
else:
    st.info("👆 Tải lên một hình ảnh để bắt đầu")
    st.markdown("---")
    st.subheader("📖 Cách sử dụng")
    st.markdown(
        """
        1. Tải lên một hoặc nhiều ảnh phiếu trả lời.
        2. Chỉnh ngưỡng fill ratio ở thanh bên nếu cần.
        3. Nhấn nút xử lý để theo dõi tiến độ batch.
        4. Xem bảng tổng hợp và chọn từng ảnh để xem chi tiết.
        5. Bật Debug để xem thêm thông tin nội bộ của pipeline.
        """
    )
