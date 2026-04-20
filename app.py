"""
Streamlit web application using the same detection/decoding logic as datetime.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

import detect as core


PROCESS_CACHE_KEY = "process_cache"
PROCESS_CACHE_VERSION = "v7-separate-id-topview"
MAX_PROCESS_CACHE_ITEMS = 32
DISPLAY_PREVIEW_MAX_SIDE = 1400


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _build_processing_cache_key(
    file_bytes: bytes,
    fill_ratio_phan1: float,
    fill_ratio_phan2: float,
    fill_ratio_phan3: float,
    debug_mode: bool,
) -> str:
    digest = hashlib.sha1(file_bytes).hexdigest()
    return (
        f"{PROCESS_CACHE_VERSION}:{digest}:"
        f"{fill_ratio_phan1:.4f}:{fill_ratio_phan2:.4f}:{fill_ratio_phan3:.4f}:"
        f"debug={int(bool(debug_mode))}"
    )


def _ensure_process_cache() -> Dict[str, Dict[str, object]]:
    if PROCESS_CACHE_KEY not in st.session_state:
        st.session_state[PROCESS_CACHE_KEY] = {}
    cache_obj = st.session_state[PROCESS_CACHE_KEY]
    if not isinstance(cache_obj, dict):
        st.session_state[PROCESS_CACHE_KEY] = {}
        cache_obj = st.session_state[PROCESS_CACHE_KEY]

    # Dọn cache cũ khác version để tránh giữ dữ liệu lớn không còn dùng.
    expected_prefix = f"{PROCESS_CACHE_VERSION}:"
    for key in list(cache_obj.keys()):
        if not str(key).startswith(expected_prefix):
            cache_obj.pop(key, None)

    return st.session_state[PROCESS_CACHE_KEY]


def _save_process_cache_entry(key: str, value: Dict[str, object]) -> None:
    cache_obj = _ensure_process_cache()
    cache_obj[key] = value
    if len(cache_obj) <= MAX_PROCESS_CACHE_ITEMS:
        return

    # Loại bớt entry cũ theo thứ tự chèn để tránh giữ quá nhiều dữ liệu ảnh lớn trong RAM.
    for old_key in list(cache_obj.keys())[: len(cache_obj) - MAX_PROCESS_CACHE_ITEMS]:
        cache_obj.pop(old_key, None)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_for_preview(image: np.ndarray, max_side: int = DISPLAY_PREVIEW_MAX_SIDE) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_image_for_state(image: Optional[np.ndarray]) -> Optional[bytes]:
    if image is None or image.size == 0:
        return None
    preview = _resize_for_preview(image)
    ok, buffer = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not ok:
        return None
    return buffer.tobytes()


def _decode_image_from_state(image_bytes: object) -> Optional[np.ndarray]:
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        return None
    arr = np.frombuffer(bytes(image_bytes), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _get_boxes_count(results: Optional[Dict[str, object]]) -> int:
    if not isinstance(results, dict):
        return 0
    data = results.get("data", {})
    if not isinstance(data, dict):
        return 0
    if "boxes_count" in data:
        return int(data.get("boxes_count", 0) or 0)
    boxes = data.get("boxes", [])
    return len(boxes) if isinstance(boxes, list) else 0


def _get_part_counts(results: Optional[Dict[str, object]]) -> Tuple[int, int, int]:
    if not isinstance(results, dict):
        return (0, 0, 0)
    parts = results.get("parts", {})
    if not isinstance(parts, dict):
        return (0, 0, 0)

    if "part_i_count" in parts:
        return (
            int(parts.get("part_i_count", 0) or 0),
            int(parts.get("part_ii_count", 0) or 0),
            int(parts.get("part_iii_count", 0) or 0),
        )

    part_i = parts.get("part_i", [])
    part_ii = parts.get("part_ii", [])
    part_iii = parts.get("part_iii", [])
    return (
        len(part_i) if isinstance(part_i, list) else 0,
        len(part_ii) if isinstance(part_ii, list) else 0,
        len(part_iii) if isinstance(part_iii, list) else 0,
    )


def _compact_results_for_storage(results: Dict[str, object], keep_debug: bool) -> Dict[str, object]:
    data = results.get("data", {})
    parts = results.get("parts", {})

    compact_data: Dict[str, object] = {
        "boxes_count": len(data.get("boxes", [])) if isinstance(data, dict) else 0,
    }
    if keep_debug and isinstance(data, dict):
        compact_data["boxes_overlay_jpg"] = _encode_image_for_state(data.get("boxes_overlay"))

    compact_parts: Dict[str, object] = {
        "part_i_count": len(parts.get("part_i", [])) if isinstance(parts, dict) else 0,
        "part_ii_count": len(parts.get("part_ii", [])) if isinstance(parts, dict) else 0,
        "part_iii_count": len(parts.get("part_iii", [])) if isinstance(parts, dict) else 0,
    }

    return {
        "resize_scale": float(results.get("resize_scale", 1.0)),
        "preprocess_mode": str(results.get("preprocess_mode", "base")),
        "data": compact_data,
        "parts": compact_parts,
        "split_x": float(results.get("split_x", 0.0)),
        "sbd_digits": results.get("sbd_digits", {}),
        "made_digits": results.get("made_digits", {}),
        "part_i_evals": results.get("part_i_evals", []),
        "part_ii_evals": results.get("part_ii_evals", []),
        "part_iii_evals": results.get("part_iii_evals", []),
        "result_image_jpg": _encode_image_for_state(results.get("result_image")),
        "parts_overlay_jpg": _encode_image_for_state(results.get("parts_overlay")) if keep_debug else None,
        "binary_threshold_jpg": _encode_image_for_state(results.get("binary_threshold")) if keep_debug else None,
    }


def _process_image_with_main_logic(
    image: np.ndarray,
    fill_ratio_phan1: float,
    fill_ratio_phan2: float,
    fill_ratio_phan3: float,
    debug_prefix: Optional[str],
) -> Dict[str, object]:
    image, resize_scale = core.resize_image_for_inference(image, max_side=2200)

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
    topview_applied = False
    page_h = image.shape[0] if image is not None else 0
    corner_markers = core.detect_black_corner_markers(image, debug_prefix=None)

    if len(parts["part_ii"]) < 8 or len(parts["part_iii"]) < 6:
        img_clahe = _preprocess_clahe(image)
        data_clahe, parts_clahe = _run_detection_pipeline(img_clahe, None)

        base_score = _parts_score(parts, page_h)
        clahe_score = _parts_score(parts_clahe, page_h)
        if clahe_score > base_score + 0.05:
            data = data_clahe
            parts = parts_clahe
            preprocess_mode = "clahe"

    def _detect_upper_id_from_layout(
        data_local: Dict[str, object],
        parts_local: Dict[str, object],
    ) -> Dict[str, object]:
        part_box_set_local = set(id(box) for box in parts_local["all_parts"])
        remaining_boxes_local = [
            box for box in data_local["boxes"]
            if id(box) not in part_box_set_local
        ]

        remaining_for_upper_local = core._split_merged_boxes_for_grouping(
            remaining_boxes_local,
            split_wide=True,
            split_tall=False,
        )

        sbd_candidates_local, ma_de_candidates_local, split_x_local = core._separate_upper_id_boxes(
            remaining_for_upper_local,
            parts_local["part_i"],
        )

        sobao_danh_local = core.detect_sobao_danh_boxes(
            sbd_candidates_local,
            boxes_per_row=6,
            max_rows=10,
            row_tolerance=30,
            size_tolerance_ratio=0.35,
            debug=False,
        )

        remaining_for_ma_de_local = core._split_merged_boxes_for_grouping(
            ma_de_candidates_local,
            split_wide=False,
            split_tall=True,
        )
        ma_de_local = core.detect_ma_de_boxes(
            remaining_for_ma_de_local,
            boxes_per_row=3,
            max_rows=10,
            row_tolerance=20,
            size_tolerance_ratio=0.3,
            debug=False,
        )

        return {
            "split_x": split_x_local,
            "sbd_candidates": sbd_candidates_local,
            "ma_de_candidates": ma_de_candidates_local,
            "sobao_danh": sobao_danh_local,
            "ma_de": ma_de_local,
        }

    upper_id = _detect_upper_id_from_layout(data, parts)
    split_x = upper_id["split_x"]
    sobao_danh = upper_id["sobao_danh"]
    ma_de = upper_id["ma_de"]
    active_sbd_candidates = upper_id["sbd_candidates"]
    active_ma_de_candidates = upper_id["ma_de_candidates"]

    id_image_for_decode = image
    id_data_for_decode = data
    id_rows_on_topview = False
    id_topview_matrix = None
    sobao_rows_are_synthetic = False
    ma_de_rows_are_synthetic = False

    def _evaluate_id_decode_quality(
        src_img: np.ndarray,
        data_local: Dict[str, object],
        sbd_rows: List[List[np.ndarray]],
        ma_de_rows_local: List[List[np.ndarray]],
    ) -> Dict[str, int]:
        gray_local = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        binary_local = data_local.get("binary")

        if sbd_rows:
            sbd_eval_local = core.evaluate_digit_rows_with_binary_fallback(
                gray_local,
                sbd_rows,
                expected_cols=6,
                binary_image=binary_local,
            )
            sbd_filled_local = int(core._count_filled_digit_columns(sbd_eval_local))
        else:
            sbd_filled_local = 0

        if ma_de_rows_local:
            ma_de_eval_local = core.evaluate_digit_rows_with_binary_fallback(
                gray_local,
                ma_de_rows_local,
                expected_cols=3,
                binary_image=binary_local,
            )
            ma_de_filled_local = int(core._count_filled_digit_columns(ma_de_eval_local))
        else:
            ma_de_filled_local = 0

        return {
            "sbd_filled": sbd_filled_local,
            "ma_de_filled": ma_de_filled_local,
        }

    base_id_quality = _evaluate_id_decode_quality(
        id_image_for_decode,
        id_data_for_decode,
        sobao_danh["sobao_danh_rows"],
        ma_de["ma_de_rows"],
    )
    current_id_quality = dict(base_id_quality)

    should_retry_id_topview = (
        int(sobao_danh["row_count"]) == 0
        and int(ma_de["row_count"]) == 0
    )
    if should_retry_id_topview:
        affine_img, affine_matrix = core._apply_affine_from_corner_markers(
            image.copy(),
            corner_markers.get("corners", {}),
        )
        if affine_img is not None and affine_matrix is not None:
            data_affine, parts_affine = _run_detection_pipeline(affine_img, None)
            upper_id_affine = _detect_upper_id_from_layout(data_affine, parts_affine)
            sobao_danh_affine = upper_id_affine["sobao_danh"]
            ma_de_affine = upper_id_affine["ma_de"]

            affine_id_quality = _evaluate_id_decode_quality(
                affine_img,
                data_affine,
                sobao_danh_affine["sobao_danh_rows"],
                ma_de_affine["ma_de_rows"],
            )

            old_score = int(sobao_danh["row_count"]) + int(ma_de["row_count"])
            new_score = int(sobao_danh_affine["row_count"]) + int(ma_de_affine["row_count"])
            old_quality_score = int(base_id_quality.get("sbd_filled", 0)) + int(base_id_quality.get("ma_de_filled", 0))
            new_quality_score = int(affine_id_quality.get("sbd_filled", 0)) + int(affine_id_quality.get("ma_de_filled", 0))

            topview_candidate_improved = (
                new_score > old_score
                or new_quality_score > old_quality_score
            )

            if topview_candidate_improved:
                split_x = upper_id_affine["split_x"]
                sobao_danh = sobao_danh_affine
                ma_de = ma_de_affine
                active_sbd_candidates = upper_id_affine["sbd_candidates"]
                active_ma_de_candidates = upper_id_affine["ma_de_candidates"]
                current_id_quality = dict(affine_id_quality)
                id_image_for_decode = affine_img
                id_data_for_decode = data_affine
                id_rows_on_topview = True
                id_topview_matrix = affine_matrix
                preprocess_mode = f"{preprocess_mode}+topview-id-only"
                topview_applied = True

    id_fallback_row_threshold = 4
    id_grid_top_ratio = 0.0725
    id_grid_row_step_ratio = 0.0211
    id_sbd_x_range_ratio_fixed = (0.74, 0.865)
    id_made_x_range_ratio_fixed = (0.90, 0.96)

    def _estimate_reference_positions_from_rows(rows: List[List[np.ndarray]], target_rows: int = 10) -> List[int]:
        if not rows:
            return []

        y_positions = []
        for row in rows:
            if not row:
                continue
            ys = [cv2.boundingRect(box)[1] for box in row]
            y_positions.append(int(round(float(np.mean(ys)))))

        if not y_positions:
            return []

        y_positions = sorted(y_positions)
        expected_top = int(round(float(id_grid_top_ratio) * float(id_image_for_decode.shape[0])))
        expected_step = max(6, int(round(float(id_grid_row_step_ratio) * float(id_image_for_decode.shape[0]))))

        if len(y_positions) >= 2:
            diffs = [max(1, y_positions[i + 1] - y_positions[i]) for i in range(len(y_positions) - 1)]
            step = int(round(float(np.median(diffs))))
        else:
            step = int(expected_step)
        step = max(1, step)

        first_detected = int(y_positions[0])
        best_offset = 0
        best_score = float("inf")
        for offset in range(target_rows):
            candidate_top = first_detected - (offset * step)
            score = abs(candidate_top - expected_top)
            if score < best_score:
                best_score = score
                best_offset = offset

        top_inferred = first_detected - (best_offset * step)
        clamp_margin = max(30, int(round(step * 2.0)))
        top_inferred = int(np.clip(top_inferred, expected_top - clamp_margin, expected_top + clamp_margin))
        return [top_inferred + (i * step) for i in range(target_rows)]

    def _estimate_id_x_range_ratio_from_candidates(
        candidates: List[np.ndarray],
        image_width: int,
        default_range: Tuple[float, float],
        min_boxes: int,
        min_width_ratio: float,
        max_width_ratio: float,
    ) -> Tuple[float, float]:
        if image_width <= 0 or not candidates or len(candidates) < min_boxes:
            return default_range

        rects = [cv2.boundingRect(box) for box in candidates]
        xs = np.array([r[0] for r in rects], dtype=np.float64)
        xe = np.array([r[0] + r[2] for r in rects], dtype=np.float64)
        ws = np.array([r[2] for r in rects], dtype=np.float64)
        if xs.size == 0 or xe.size == 0 or ws.size == 0:
            return default_range

        margin = 0.5 * float(np.median(ws))
        x0 = float(np.percentile(xs, 10.0)) - margin
        x1 = float(np.percentile(xe, 90.0)) + margin

        x0 = float(np.clip(x0, 0.0, float(image_width - 2)))
        x1 = float(np.clip(x1, x0 + 1.0, float(image_width - 1)))

        r0 = x0 / float(image_width)
        r1 = x1 / float(image_width)
        width_ratio = r1 - r0
        if width_ratio < min_width_ratio or width_ratio > max_width_ratio:
            return default_range
        return (float(r0), float(r1))

    sbd_x_range_ratio = _estimate_id_x_range_ratio_from_candidates(
        active_sbd_candidates,
        id_image_for_decode.shape[1],
        id_sbd_x_range_ratio_fixed,
        min_boxes=18,
        min_width_ratio=0.08,
        max_width_ratio=0.22,
    )
    made_x_range_ratio = _estimate_id_x_range_ratio_from_candidates(
        active_ma_de_candidates,
        id_image_for_decode.shape[1],
        id_made_x_range_ratio_fixed,
        min_boxes=9,
        min_width_ratio=0.04,
        max_width_ratio=0.14,
    )

    fallback_threshold = max(0, int(id_fallback_row_threshold))
    allow_ratio_id_fallback = topview_applied
    sbd_needs_fallback = (
        sobao_danh["row_count"] <= fallback_threshold
        or int(current_id_quality.get("sbd_filled", 0)) == 0
    )
    ma_de_needs_fallback = (
        ma_de["row_count"] <= fallback_threshold
        or int(current_id_quality.get("ma_de_filled", 0)) == 0
    )

    if allow_ratio_id_fallback and sbd_needs_fallback:
        reference_positions = []
        if (
            ma_de["row_count"] >= 3
            and ma_de["ma_de_rows"]
            and int(current_id_quality.get("ma_de_filled", 0)) > 0
        ):
            reference_positions = _estimate_reference_positions_from_rows(ma_de["ma_de_rows"], target_rows=10)

        if reference_positions:
            ma_de_rects = [cv2.boundingRect(box) for row in ma_de["ma_de_rows"] for box in row]
            ma_de_row_h = None
            if ma_de_rects:
                ma_de_row_h = int(round(float(np.median([r[3] for r in ma_de_rects]))))
            sobao_rows = core._build_synthetic_id_rows_from_reference_positions(
                image_shape=id_image_for_decode.shape[:2],
                cols=6,
                x_range_ratio=sbd_x_range_ratio,
                reference_positions=reference_positions,
                row_height=ma_de_row_h,
            )
        else:
            sobao_rows = core._build_synthetic_id_rows_fixed_image_position(
                image_shape=id_image_for_decode.shape[:2],
                cols=6,
                rows=10,
                x_range_ratio=sbd_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )

        if sobao_rows:
            sobao_danh["sobao_danh_rows"] = sobao_rows
            sobao_danh["sobao_danh"] = [b for row in sobao_rows for b in row]
            sobao_danh["row_count"] = len(sobao_rows)
            sobao_rows_are_synthetic = True

    if allow_ratio_id_fallback and ma_de_needs_fallback:
        reference_positions = []
        if (
            sobao_danh["row_count"] >= 3
            and sobao_danh["sobao_danh_rows"]
            and not sobao_rows_are_synthetic
            and int(current_id_quality.get("sbd_filled", 0)) > 0
        ):
            reference_positions = _estimate_reference_positions_from_rows(sobao_danh["sobao_danh_rows"], target_rows=10)

        if reference_positions:
            sbd_rects = [cv2.boundingRect(box) for row in sobao_danh["sobao_danh_rows"] for box in row]
            sbd_row_h = None
            if sbd_rects:
                sbd_row_h = int(round(float(np.median([r[3] for r in sbd_rects]))))
            ma_de_rows = core._build_synthetic_id_rows_from_reference_positions(
                image_shape=id_image_for_decode.shape[:2],
                cols=3,
                x_range_ratio=made_x_range_ratio,
                reference_positions=reference_positions,
                row_height=sbd_row_h,
            )
        else:
            ma_de_rows = core._build_synthetic_id_rows_fixed_image_position(
                image_shape=id_image_for_decode.shape[:2],
                cols=3,
                rows=10,
                x_range_ratio=made_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )

        if ma_de_rows:
            ma_de["ma_de_rows"] = ma_de_rows
            ma_de["ma_de"] = [b for row in ma_de_rows for b in row]
            ma_de["row_count"] = len(ma_de_rows)
            ma_de_rows_are_synthetic = True

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
                if len(ma_de_rect_rows) < 10:
                    ma_de_rows_are_synthetic = True

    # Keep the same SoBaoDanh completion behavior as detect.py.
    if (
        sobao_danh["row_count"] < 10
        and sobao_danh["sobao_danh_rows"]
        and ma_de["ma_de_rows"]
        and not ma_de_rows_are_synthetic
    ):
        ref_positions = []
        for row in ma_de["ma_de_rows"][:10]:
            ys = [cv2.boundingRect(box)[1] for box in row]
            if ys:
                ref_positions.append(int(round(float(np.mean(ys)))))

        sobao_rect_rows: List[List[Tuple[int, int, int, int]]] = []
        for row in sobao_danh["sobao_danh_rows"]:
            rects = [cv2.boundingRect(box) for box in row]
            rects = sorted(rects, key=lambda r: r[0])
            if len(rects) == 6:
                sobao_rect_rows.append(rects)

        if len(ref_positions) == 10 and sobao_rect_rows:
            col_x = [int(round(float(np.median([rects[c][0] for rects in sobao_rect_rows])))) for c in range(6)]
            col_w = [int(round(float(np.median([rects[c][2] for rects in sobao_rect_rows])))) for c in range(6)]
            row_h = int(round(float(np.median([rects[0][3] for rects in sobao_rect_rows]))))
            row_h = max(1, row_h)

            detected_rows_y = [int(round(float(np.mean([r[1] for r in rects])))) for rects in sobao_rect_rows]
            align_tolerance = max(35, int(round(row_h * 1.2)))
            aligned_rows: List[Optional[List[np.ndarray]]] = [None] * 10
            matched_rows: List[Tuple[int, List[Tuple[int, int, int, int]], int]] = []

            used_ref_indices = set()
            for rects, row_y in sorted(zip(sobao_rect_rows, detected_rows_y), key=lambda t: t[1]):
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
                matched_rows.append((chosen_idx, rects, row_y))

            col_x_trend_samples: List[List[Tuple[float, float]]] = [[] for _ in range(6)]
            for rects, row_y in zip(sobao_rect_rows, detected_rows_y):
                for c in range(6):
                    col_x_trend_samples[c].append((float(row_y), float(rects[c][0])))

            def _predict_col_x(col_idx: int, y_target: float) -> int:
                samples = col_x_trend_samples[col_idx]
                base_x = col_x[col_idx]
                if len(samples) < 2:
                    return int(base_x)

                ys = np.array([p[0] for p in samples], dtype=np.float64)
                xs = np.array([p[1] for p in samples], dtype=np.float64)
                if float(np.max(ys) - np.min(ys)) < 1.0:
                    return int(base_x)

                slope, intercept = np.polyfit(ys, xs, 1)
                pred = float(slope * float(y_target) + intercept)
                margin = max(8.0, float(col_w[col_idx]) * 2.0)
                x_min = float(np.min(xs)) - margin
                x_max = float(np.max(xs)) + margin
                pred = float(np.clip(pred, x_min, x_max))
                return int(round(pred))

            y_bias = 0
            if matched_rows:
                y_deltas = [int(row_y - ref_positions[idx]) for idx, _, row_y in matched_rows]
                y_bias = int(round(float(np.median(y_deltas))))

            col_x_bias = [0] * 6
            if matched_rows:
                for c in range(6):
                    x_deltas: List[int] = []
                    for idx, rects, _ in matched_rows:
                        y_anchor = float(ref_positions[idx] + y_bias)
                        pred_x = _predict_col_x(c, y_anchor)
                        x_deltas.append(int(rects[c][0] - pred_x))
                    if x_deltas:
                        col_x_bias[c] = int(round(float(np.median(x_deltas))))

            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx] + y_bias
                synthetic_row: List[np.ndarray] = []
                for c in range(6):
                    x = _predict_col_x(c, float(y_ref)) + col_x_bias[c]
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

            sobao_completed_rows = [row for row in aligned_rows if row is not None]
            if len(sobao_completed_rows) == 10:
                sobao_danh["sobao_danh_rows"] = sobao_completed_rows
                sobao_danh["sobao_danh"] = [box for row in sobao_completed_rows for box in row]
                sobao_danh["row_count"] = 10
                if len(sobao_rect_rows) < 10:
                    sobao_rows_are_synthetic = True

    combined_results = {
        "sobao_danh_rows": sobao_danh["sobao_danh_rows"],
        "ma_de_rows": ma_de["ma_de_rows"],
        "sobao_is_synthetic": sobao_rows_are_synthetic,
        "ma_de_is_synthetic": ma_de_rows_are_synthetic,
    }
    expected_id_top_y = int(round(float(id_grid_top_ratio) * float(id_image_for_decode.shape[0])))
    expected_id_row_step = max(6, int(round(float(id_grid_row_step_ratio) * float(id_image_for_decode.shape[0]))))
    extrapolated = core.extrapolate_missing_rows(
        combined_results,
        target_rows=10,
        expected_top_y=expected_id_top_y,
        expected_row_step=expected_id_row_step,
        debug=False,
    )

    gray_for_digits = cv2.cvtColor(id_image_for_decode, cv2.COLOR_BGR2GRAY)
    binary_for_digits = id_data_for_decode.get("binary")
    sobao_rows_aligned = extrapolated.get("sobao_danh_rows_aligned", sobao_danh["sobao_danh_rows"])
    ma_de_rows_aligned = extrapolated.get("ma_de_rows_aligned", ma_de["ma_de_rows"])
    if sobao_rows_aligned is None:
        sobao_rows_aligned = []
    if ma_de_rows_aligned is None:
        ma_de_rows_aligned = []

    def _transform_rows_to_source(
        rows: List[List[np.ndarray]],
        matrix: Optional[np.ndarray],
    ) -> List[List[np.ndarray]]:
        if not rows:
            return []
        if matrix is None:
            return rows

        transformed_rows: List[List[np.ndarray]] = []
        for row in rows:
            transformed_row: List[np.ndarray] = []
            for poly in row:
                poly_float = np.asarray(poly, dtype=np.float32)
                if poly_float.ndim == 2:
                    poly_float = poly_float[:, np.newaxis, :]
                mapped = cv2.perspectiveTransform(poly_float, matrix)
                transformed_row.append(np.round(mapped).astype(np.int32))
            transformed_rows.append(transformed_row)
        return transformed_rows

    sobao_rows_for_display = sobao_rows_aligned
    ma_de_rows_for_display = ma_de_rows_aligned
    if id_rows_on_topview and id_topview_matrix is not None:
        try:
            inv_matrix = np.linalg.inv(id_topview_matrix)
            sobao_rows_for_display = _transform_rows_to_source(sobao_rows_aligned, inv_matrix)
            ma_de_rows_for_display = _transform_rows_to_source(ma_de_rows_aligned, inv_matrix)
        except np.linalg.LinAlgError:
            pass

    sobao_danh["sobao_danh_rows"] = sobao_rows_for_display
    sobao_danh["sobao_danh"] = [box for row in sobao_rows_for_display for box in row]
    sobao_danh["row_count"] = len(sobao_rows_for_display)
    ma_de["ma_de_rows"] = ma_de_rows_for_display
    ma_de["ma_de"] = [box for row in ma_de_rows_for_display for box in row]
    ma_de["row_count"] = len(ma_de_rows_for_display)

    sbd_digits = core.evaluate_digit_rows_with_binary_fallback(
        gray_for_digits,
        sobao_rows_aligned,
        expected_cols=6,
        binary_image=binary_for_digits,
    )
    made_digits = core.evaluate_digit_rows_with_binary_fallback(
        gray_for_digits,
        ma_de_rows_aligned,
        expected_cols=3,
        binary_image=binary_for_digits,
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
    for row in sobao_rows_for_display:
        for poly in row:
            cv2.polylines(overlay, [poly], True, sobao_color, 2)

    ma_de_color = (255, 255, 0)
    for row in ma_de_rows_for_display:
        for poly in row:
            cv2.polylines(overlay, [poly], True, ma_de_color, 2)

    combined_grid_image = image.copy()
    binary_threshold = data.get("binary_score", data.get("binary"))

    part_i_evals: List[Dict[str, object]] = []
    if parts["part_i"]:
        grid_result = core.extract_grid_from_boxes(
            combined_grid_image,
            boxes=parts["part_i"],
            grid_cols=4,
            grid_rows=10,
            start_offset_ratio_x=0.185,
            start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.03,
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
                circle_border_exclude_ratio=0.12,
                hough_only_ambiguous=True,
                hough_ambiguity_margin=0.10,
                hough_max_cells=120,
            )
            part_i_evals = core.suppress_false_positive_by_relative_dominance(
                part_i_evals,
                group_keys=("box_idx", "row"),
                ratio_key="fill_ratio",
                min_dominant_ratio=float(fill_ratio_phan1),
                min_gap=0.12,
                min_ratio_scale=1.35,
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
                circle_border_exclude_ratio=0.1,
                hough_only_ambiguous=True,
                hough_ambiguity_margin=0.10,
                hough_max_cells=120,
            )
            part_ii_evals = core.suppress_false_positive_by_relative_dominance(
                part_ii_evals,
                group_keys=("box_idx", "row"),
                ratio_key="fill_ratio",
                min_dominant_ratio=float(fill_ratio_phan2),
                min_gap=0.10,
                min_ratio_scale=1.25,
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
                circle_border_exclude_ratio=0.12,
                hough_only_ambiguous=True,
                hough_ambiguity_margin=0.10,
                hough_max_cells=120,
            )
            part_iii_evals = core.suppress_false_positive_by_relative_dominance(
                part_iii_evals,
                group_keys=("box_idx", "col"),
                ratio_key="fill_ratio",
                min_dominant_ratio=float(fill_ratio_phan3),
                min_gap=0.10,
                min_ratio_scale=1.20,
            )

    if sobao_rows_for_display:
        for row in sobao_rows_for_display:
            for poly in row:
                cv2.polylines(combined_grid_image, [poly], True, (255, 128, 0), 1)

    if ma_de_rows_for_display:
        for row in ma_de_rows_for_display:
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
        "resize_scale": resize_scale,
        "preprocess_mode": preprocess_mode,
        "data": data,
        "parts": parts,
        "sobao_danh": sobao_danh,
        "ma_de": ma_de,
        "split_x": split_x,
        "id_process_mode": "topview-copy" if topview_applied else "base",
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
    process_seconds: Optional[float] = None,
    cache_hit: bool = False,
) -> Dict[str, object]:
    process_text = "" if process_seconds is None else f"{float(process_seconds):.2f}"
    source_text = "cache" if cache_hit else "compute"

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
            "Time(s)": process_text,
            "Source": source_text,
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
        "Boxes": _get_boxes_count(results),
        "Preprocess": results["preprocess_mode"],
        "Time(s)": process_text,
        "Source": source_text,
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
        parts_overlay = _decode_image_from_state(results.get("parts_overlay_jpg"))
        binary_threshold = _decode_image_from_state(results.get("binary_threshold_jpg"))
        boxes_overlay = _decode_image_from_state(results.get("data", {}).get("boxes_overlay_jpg"))

        with debug_col1:
            st.markdown("**Overlay Parts**")
            if parts_overlay is not None:
                st.image(_to_rgb(parts_overlay), use_container_width=True)
            else:
                st.info("Không có overlay debug trong phiên này")
        with debug_col2:
            if binary_threshold is not None:
                st.markdown("**Binary Threshold**")
                st.image(_to_rgb(binary_threshold), use_container_width=True)
            elif boxes_overlay is not None:
                st.markdown("**Boxes Overlay**")
                st.image(_to_rgb(boxes_overlay), use_container_width=True)
            else:
                st.info("Không có ảnh debug")

        part_i_evals = results["part_i_evals"]
        part_ii_evals = results["part_ii_evals"]
        part_iii_evals = results["part_iii_evals"]
        total_cells = len(part_i_evals) + len(part_ii_evals) + len(part_iii_evals)
        total_filled = sum(1 for e in (part_i_evals + part_ii_evals + part_iii_evals) if e.get("filled", False))
        part_i_count, part_ii_count, part_iii_count = _get_part_counts(results)

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Detected Boxes", _get_boxes_count(results))
        with stats_col2:
            st.metric("Tổng Cộng Ô", total_cells)
        with stats_col3:
            pct = (total_filled / max(1, total_cells)) * 100.0
            st.metric("Tỷ Lệ Điền", f"{pct:.1f}%")

        st.caption(
            f"Preprocess: {results['preprocess_mode']} | "
            f"Resize scale: {float(results.get('resize_scale', 1.0)):.3f} | "
            f"Part I/II/III: {part_i_count}/{part_ii_count}/{part_iii_count} | "
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
    debug_mode = st.checkbox("Chế Độ Debug", value=False)

    st.markdown("---")
    st.markdown("### Ngưỡng Fill Ratio")
    fill_ratio_phan1 = st.slider("PHẦN I", 0.30, 0.95, 0.52, 0.01)
    fill_ratio_phan2 = st.slider("PHẦN II", 0.30, 0.95, 0.52, 0.01)
    fill_ratio_phan3 = st.slider("PHẦN III", 0.30, 0.95, 0.52, 0.01)

    st.markdown("---")
    st.markdown("### Ghi chú")
    st.caption("Luồng xử lý: detect box -> group part -> decode SBD/Mã đề -> grid fill ratio -> trích xuất đáp án")
    clear_cache_clicked = st.button(
        "🧹 Xóa cache RAM",
        use_container_width=True,
        help="Xóa cache xử lý và kết quả batch đang giữ trong session",
    )

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
BATCH_TIMING_KEY = "batch_timing"

if BATCH_RESULTS_KEY not in st.session_state:
    st.session_state[BATCH_RESULTS_KEY] = []
if BATCH_SIGNATURE_KEY not in st.session_state:
    st.session_state[BATCH_SIGNATURE_KEY] = ()
if BATCH_TIMING_KEY not in st.session_state:
    st.session_state[BATCH_TIMING_KEY] = {}

if clear_cache_clicked:
    st.session_state[PROCESS_CACHE_KEY] = {}
    st.session_state[BATCH_RESULTS_KEY] = []
    st.session_state[BATCH_SIGNATURE_KEY] = ()
    st.session_state[BATCH_TIMING_KEY] = {}
    st.sidebar.success("Đã xóa cache trong RAM")

current_signature: Tuple[Tuple[str, int], ...] = ()
if uploaded_files:
    current_signature = tuple((f.name, int(getattr(f, "size", 0))) for f in uploaded_files)

if process_clicked and uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Bắt đầu xử lý {len(uploaded_files)} ảnh...")

    batch_results: List[Dict[str, object]] = []
    total_files = len(uploaded_files)
    process_cache = _ensure_process_cache()
    batch_started_at = time.perf_counter()

    for idx, uploaded in enumerate(uploaded_files, start=1):
        file_name = uploaded.name
        file_started_at = time.perf_counter()
        elapsed_before = max(0.0, time.perf_counter() - batch_started_at)
        status_text.text(
            f"Đang xử lý {idx}/{total_files}: {file_name} | "
            f"Đã chạy: {_format_duration(elapsed_before)}"
        )
        latest_source = "compute"
        process_seconds = 0.0

        try:
            raw_bytes = uploaded.getvalue()
            cache_key = _build_processing_cache_key(
                file_bytes=raw_bytes,
                fill_ratio_phan1=float(fill_ratio_phan1),
                fill_ratio_phan2=float(fill_ratio_phan2),
                fill_ratio_phan3=float(fill_ratio_phan3),
                debug_mode=bool(debug_mode),
            )

            cached_item = process_cache.get(cache_key)
            if cached_item is not None:
                process_seconds = max(0.0, time.perf_counter() - file_started_at)
                latest_source = "cache"
                result_item = {
                    "file_name": file_name,
                    "status": "OK",
                    "error": "",
                    "image_jpg": cached_item.get("image_jpg"),
                    "results": cached_item.get("results"),
                    "extracted": cached_item.get("extracted"),
                    "process_seconds": process_seconds,
                    "cache_hit": True,
                    "summary": _build_batch_summary_row(
                        file_name,
                        "OK",
                        cached_item.get("results"),
                        cached_item.get("extracted"),
                        process_seconds=process_seconds,
                        cache_hit=True,
                    ),
                }
                batch_results.append(result_item)
            else:
                file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Không đọc được ảnh")

                # App không cần ghi ảnh debug trung gian ra đĩa để render UI.
                debug_prefix = None

                raw_results = _process_image_with_main_logic(
                    image,
                    fill_ratio_phan1=fill_ratio_phan1,
                    fill_ratio_phan2=fill_ratio_phan2,
                    fill_ratio_phan3=fill_ratio_phan3,
                    debug_prefix=debug_prefix,
                )
                results = _compact_results_for_storage(raw_results, keep_debug=bool(debug_mode))
                extracted = _build_structured_answers(results)
                process_seconds = max(0.0, time.perf_counter() - file_started_at)

                result_item = {
                    "file_name": file_name,
                    "status": "OK",
                    "error": "",
                    "image_jpg": _encode_image_for_state(image),
                    "results": results,
                    "extracted": extracted,
                    "process_seconds": process_seconds,
                    "cache_hit": False,
                    "summary": _build_batch_summary_row(
                        file_name,
                        "OK",
                        results,
                        extracted,
                        process_seconds=process_seconds,
                        cache_hit=False,
                    ),
                }
                batch_results.append(result_item)
                _save_process_cache_entry(
                    cache_key,
                    {
                        "image_jpg": result_item["image_jpg"],
                        "results": result_item["results"],
                        "extracted": result_item["extracted"],
                    },
                )
        except Exception as err:
            process_seconds = max(0.0, time.perf_counter() - file_started_at)
            latest_source = "error"
            batch_results.append(
                {
                    "file_name": file_name,
                    "status": "ERROR",
                    "error": str(err),
                    "image_jpg": None,
                    "results": None,
                    "extracted": None,
                    "process_seconds": process_seconds,
                    "cache_hit": False,
                    "summary": _build_batch_summary_row(
                        file_name,
                        "ERROR",
                        None,
                        None,
                        error_message=str(err),
                        process_seconds=process_seconds,
                        cache_hit=False,
                    ),
                }
            )

        processed_count = idx
        elapsed_seconds = max(0.0, time.perf_counter() - batch_started_at)
        avg_seconds = elapsed_seconds / max(1, processed_count)
        remaining = max(0, total_files - processed_count)
        eta_seconds = avg_seconds * remaining
        speed = processed_count / max(1e-6, elapsed_seconds)

        progress_bar.progress(int((processed_count / max(1, total_files)) * 100))
        status_text.text(
            f"Đã xử lý {processed_count}/{total_files} ảnh | "
            f"Ảnh gần nhất: {file_name} ({process_seconds:.2f}s - {latest_source}) | "
            f"Đã chạy: {_format_duration(elapsed_seconds)} | "
            f"ETA: {_format_duration(eta_seconds)}"
            f" | Tốc độ: {speed:.2f} ảnh/giây"
        )

    total_elapsed = max(0.0, time.perf_counter() - batch_started_at)
    avg_seconds = total_elapsed / max(1, total_files)
    speed = total_files / max(1e-6, total_elapsed)
    progress_bar.progress(100)
    status_text.text(
        f"Hoàn thành {len(uploaded_files)} ảnh trong {_format_duration(total_elapsed)} | "
        f"Tốc độ TB: {speed:.2f} ảnh/giây"
    )
    st.session_state[BATCH_RESULTS_KEY] = batch_results
    st.session_state[BATCH_SIGNATURE_KEY] = current_signature
    st.session_state[BATCH_TIMING_KEY] = {
        "file_count": total_files,
        "total_seconds": total_elapsed,
        "avg_seconds": avg_seconds,
        "speed": speed,
    }

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

    timing_info = st.session_state.get(BATCH_TIMING_KEY, {})
    if isinstance(timing_info, dict) and timing_info.get("file_count") == len(batch_results):
        time_col1, time_col2, time_col3 = st.columns(3)
        with time_col1:
            st.metric("Tổng thời gian", _format_duration(float(timing_info.get("total_seconds", 0.0))))
        with time_col2:
            st.metric("TB mỗi ảnh", f"{float(timing_info.get('avg_seconds', 0.0)):.2f}s")
        with time_col3:
            st.metric("Tốc độ TB", f"{float(timing_info.get('speed', 0.0)):.2f} ảnh/giây")

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

        selected_image = _decode_image_from_state(selected_item.get("image_jpg"))
        selected_result_image = _decode_image_from_state(selected_item["results"].get("result_image_jpg"))

        with col1:
            st.subheader("📸 Hình Ảnh Gốc")
            if selected_image is not None:
                st.image(_to_rgb(selected_image), use_container_width=True)
            else:
                st.info("Không có ảnh gốc để hiển thị")

        with col2:
            st.subheader("📊 Kết Quả Phân Tích")
            if selected_result_image is not None:
                st.image(_to_rgb(selected_result_image), use_container_width=True)
            else:
                st.info("Không có ảnh kết quả để hiển thị")

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
