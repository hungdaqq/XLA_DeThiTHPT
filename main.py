from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

import cv2
import numpy as np


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return 4 points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != 4:
        raise ValueError("Expected 4 points to order a quadrilateral")

    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def _box_to_quad(box: np.ndarray) -> np.ndarray:
    """Convert an arbitrary contour to a stable 4-point quadrilateral."""
    contour = box.reshape(-1, 2).astype(np.float32)
    if contour.shape[0] == 4:
        return _order_quad_points(contour)

    peri = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2), 0.02 * peri, True)
    if approx.shape[0] == 4:
        return _order_quad_points(approx.reshape(-1, 2))

    rect = cv2.minAreaRect(contour.reshape(-1, 1, 2))
    quad = cv2.boxPoints(rect)
    return _order_quad_points(quad)


def _lerp_point(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    return p0 + (p1 - p0) * float(t)


def _point_on_quad(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """Bilinear interpolation on ordered quad with u,v in [0,1]."""
    top = _lerp_point(quad[0], quad[1], u)
    bottom = _lerp_point(quad[3], quad[2], u)
    return _lerp_point(top, bottom, v)


def _inner_quad(
    quad: np.ndarray,
    start_offset_x: float,
    start_offset_y: float,
    end_offset_x: float,
    end_offset_y: float,
) -> np.ndarray:
    u0 = float(np.clip(start_offset_x, 0.0, 0.95))
    v0 = float(np.clip(start_offset_y, 0.0, 0.95))
    u1 = float(np.clip(1.0 - end_offset_x, u0 + 1e-4, 1.0))
    v1 = float(np.clip(1.0 - end_offset_y, v0 + 1e-4, 1.0))

    return np.array(
        [
            _point_on_quad(quad, u0, v0),
            _point_on_quad(quad, u1, v0),
            _point_on_quad(quad, u1, v1),
            _point_on_quad(quad, u0, v1),
        ],
        dtype=np.float32,
    )


def _draw_grid_lines_on_quad(
    image: np.ndarray,
    quad: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    for col in range(1, grid_cols):
        t = col / float(grid_cols)
        p_top = _lerp_point(quad[0], quad[1], t)
        p_bottom = _lerp_point(quad[3], quad[2], t)
        cv2.line(image, tuple(np.round(p_top).astype(int)), tuple(np.round(p_bottom).astype(int)), color, thickness)

    for row in range(1, grid_rows):
        t = row / float(grid_rows)
        p_left = _lerp_point(quad[0], quad[3], t)
        p_right = _lerp_point(quad[1], quad[2], t)
        cv2.line(image, tuple(np.round(p_left).astype(int)), tuple(np.round(p_right).astype(int)), color, thickness)


def _draw_grid_cells_with_pattern(
    image: np.ndarray,
    quad: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    row_col_patterns: Optional[List[List[int]]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    if grid_cols <= 0 or grid_rows <= 0:
        return

    for row in range(grid_rows):
        if row_col_patterns and row < len(row_col_patterns):
            cols_to_draw = row_col_patterns[row]
        else:
            cols_to_draw = list(range(grid_cols))

        v0 = row / float(grid_rows)
        v1 = (row + 1) / float(grid_rows)
        for col in cols_to_draw:
            if col < 0 or col >= grid_cols:
                continue
            u0 = col / float(grid_cols)
            u1 = (col + 1) / float(grid_cols)

            cell = np.array(
                [
                    _point_on_quad(quad, u0, v0),
                    _point_on_quad(quad, u1, v0),
                    _point_on_quad(quad, u1, v1),
                    _point_on_quad(quad, u0, v1),
                ],
                dtype=np.float32,
            )
            cv2.polylines(image, [cell.astype(np.int32)], True, color, thickness)


def _validate_box_dims(box: np.ndarray, box_idx: int) -> Optional[Tuple[int, int, int, int]]:
    """Return bounding rect when valid, otherwise print warning and return None."""
    x, y, w, h = cv2.boundingRect(box)
    if w <= 0 or h <= 0:
        print(f"Warning: Box {box_idx} has invalid dimensions: {w}x{h}")
        return None
    return x, y, w, h


def _build_grid_info(
    box_idx: int,
    box_bounds: Tuple[int, int, int, int],
    region_quad: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    extra: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """Build grid metadata dict shared by all grid extraction variants."""
    _, _, region_width, region_height = cv2.boundingRect(region_quad.astype(np.int32))
    if region_width <= 0 or region_height <= 0:
        print(
            f"Warning: Box {box_idx} has invalid region dimensions: "
            f"{region_width}x{region_height}"
        )
        return None

    info: Dict[str, object] = {
        "box_idx": box_idx,
        "box_bounds": box_bounds,
        "region_quad": region_quad.tolist(),
        "region_size": (region_width, region_height),
        "cell_size": (
            region_width / max(1, grid_cols),
            region_height / max(1, grid_rows),
        ),
        "grid_shape": (grid_rows, grid_cols),
    }
    if extra:
        info.update(extra)
    return info


def extract_grid_from_boxes_custom_pattern(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 12,
    start_offset_ratio_x: float = 0.2,
    start_offset_ratio_y: float = 0.1,
    end_offset_ratio_x: float = 0.0,
    end_offset_ratio_y: float = 0.0,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
    row_col_patterns: Optional[List[List[int]]] = None,
) -> Dict[str, object]:
    """
    Draw a grid on each box with custom cell patterns per row.
    
    Args:
        image: Input image (will be modified in place)
        boxes: List of boxes (as numpy arrays with polygon coordinates)
        grid_cols: Number of columns in grid (default 4)
        grid_rows: Number of rows in grid (default 12)
        start_offset_ratio_x: Offset ratio for X axis (default 0.2 = 20%)
        start_offset_ratio_y: Offset ratio for Y axis (default 0.1 = 10%)
        end_offset_ratio_x: End offset ratio for X axis (from right, default 0.0 = 0%)
        end_offset_ratio_y: End offset ratio for Y axis (from bottom, default 0.0 = 0%)
        grid_color: Color for grid lines (BGR tuple, default green)
        grid_thickness: Thickness of grid lines (default 1)
        row_col_patterns: List of column indices for each row.
                         E.g., [[0], [1, 2], [0, 1, 2, 3], ...] for custom patterns
                         If None, draw all columns normally
    
    Returns:
        Dictionary with:
        - 'image_with_grid': Image with grid drawn on it
        - 'grid_info': List of grid metadata (position, size, etc)
    """
    # Create a copy to draw on
    output_image = image.copy()
    grid_info = []
    
    for box_idx, box in enumerate(boxes):
        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(
            quad,
            start_offset_ratio_x,
            start_offset_ratio_y,
            end_offset_ratio_x,
            end_offset_ratio_y,
        )

        if row_col_patterns:
            _draw_grid_cells_with_pattern(
                output_image,
                region_quad,
                grid_cols,
                grid_rows,
                row_col_patterns,
                grid_color,
                grid_thickness,
            )
        else:
            _draw_grid_lines_on_quad(
                output_image,
                region_quad,
                grid_cols,
                grid_rows,
                grid_color,
                grid_thickness,
            )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "pattern": row_col_patterns,
                "start_offset": (start_offset_ratio_x, start_offset_ratio_y),
                "end_offset": (end_offset_ratio_x, end_offset_ratio_y),
            },
        )
        if info is not None:
            grid_info.append(info)
    
    return {
        "image_with_grid": output_image,
        "grid_info": grid_info,
    }


def extract_grid_from_boxes_variable_offsets(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 10,
    start_offset_ratios: Optional[List[Tuple[float, float]]] = None,
    end_offset_ratios_x: Optional[List[float]] = None,
    end_offset_ratios_y: Optional[List[float]] = None,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
) -> Dict[str, object]:
    """
    Draw a grid on each box with variable offset ratios per box.
    
    For each box:
    1. Use offset from start_offset_ratios[box_idx] or default to (0.2, 0.1)
    2. Calculate start point: x_start = x + offset_x * width, 
                             y_start = y + offset_y * height
     3. Calculate end point from end offsets:
         - x_end by end_offset_x (from right)
         - y_end by end_offset_y (from bottom)
    4. Draw grid_cols x grid_rows cells with lines
    
    Args:
        image: Input image (will be modified in place)
        boxes: List of boxes (as numpy arrays with polygon coordinates)
        grid_cols: Number of columns in grid (default 4)
        grid_rows: Number of rows in grid (default 10)
        start_offset_ratios: List of (offset_x, offset_y) tuples per box.
                            If None or fewer than len(boxes), use (0.2, 0.1) as default
        end_offset_ratios_x: List of end X offset ratios (from right).
                    If None or fewer than len(boxes), use 0.0
        end_offset_ratios_y: List of end Y offset ratios (from bottom).
                            If None or fewer than len(boxes), use full height
        grid_color: Color for grid lines (BGR tuple, default green)
        grid_thickness: Thickness of grid lines (default 1)
    
    Returns:
        Dictionary with:
        - 'image_with_grid': Image with grid drawn on it
        - 'grid_info': List of grid metadata (position, size, etc)
    """
    # Create a copy to draw on
    output_image = image.copy()
    grid_info = []
    
    default_offset = (0.2, 0.1)
    
    for box_idx, box in enumerate(boxes):
        # Get offset ratio for this box
        if start_offset_ratios and box_idx < len(start_offset_ratios):
            offset_x, offset_y = start_offset_ratios[box_idx]
        else:
            offset_x, offset_y = default_offset
        
        # Get end offsets if specified
        end_offset_x = 0.0
        if end_offset_ratios_x and box_idx < len(end_offset_ratios_x):
            end_offset_x = end_offset_ratios_x[box_idx]

        end_offset_y = 0.0
        if end_offset_ratios_y and box_idx < len(end_offset_ratios_y):
            end_offset_y = end_offset_ratios_y[box_idx]
        
        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(quad, offset_x, offset_y, end_offset_x, end_offset_y)
        _draw_grid_lines_on_quad(
            output_image,
            region_quad,
            grid_cols,
            grid_rows,
            grid_color,
            grid_thickness,
        )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "offset_ratios": (offset_x, offset_y),
                "end_offset_x": end_offset_x,
                "end_offset_y": end_offset_y,
            },
        )
        if info is not None:
            grid_info.append(info)
    
    return {
        "image_with_grid": output_image,
        "grid_info": grid_info,
    }


def extract_grid_from_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    grid_cols: int = 4,
    grid_rows: int = 10,
    start_offset_ratio_x: float = 0.2,
    start_offset_ratio_y: float = 0.1,
    end_offset_ratio_x: float = 0.0,
    end_offset_ratio_y: float = 0.0,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    grid_thickness: int = 1,
) -> Dict[str, object]:
    """
    Draw a grid on each box in the image.
    
    Draw a perspective-aware grid on each box with start/end offsets for both axes.
    
    Args:
        image: Input image (will be modified in place)
        boxes: List of boxes (as numpy arrays with polygon coordinates)
        grid_cols: Number of columns in grid (default 4)
        grid_rows: Number of rows in grid (default 10)
        start_offset_ratio_x: Offset ratio for X axis (default 0.2 = 20%)
        start_offset_ratio_y: Offset ratio for Y axis (default 0.1 = 10%)
        end_offset_ratio_x: End offset ratio for X axis (from right, default 0.0)
        end_offset_ratio_y: End offset ratio for Y axis (from bottom, default 0.0)
        grid_color: Color for grid lines (BGR tuple, default green)
        grid_thickness: Thickness of grid lines (default 1)
    
    Returns:
        Dictionary with:
        - 'image_with_grid': Image with grid drawn on it
        - 'grid_info': List of grid metadata (position, size, etc)
    """
    # Create a copy to draw on
    output_image = image.copy()
    grid_info = []
    
    for box_idx, box in enumerate(boxes):
        box_bounds = _validate_box_dims(box, box_idx)
        if box_bounds is None:
            continue
        x, y, w, h = box_bounds

        quad = _box_to_quad(box)
        region_quad = _inner_quad(
            quad,
            start_offset_ratio_x,
            start_offset_ratio_y,
            end_offset_ratio_x,
            end_offset_ratio_y,
        )
        _draw_grid_lines_on_quad(
            output_image,
            region_quad,
            grid_cols,
            grid_rows,
            grid_color,
            grid_thickness,
        )

        info = _build_grid_info(
            box_idx=box_idx,
            box_bounds=(x, y, w, h),
            region_quad=region_quad,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            extra={
                "start_offset": (start_offset_ratio_x, start_offset_ratio_y),
                "end_offset": (end_offset_ratio_x, end_offset_ratio_y),
            },
        )
        if info is not None:
            grid_info.append(info)
    
    return {
        "image_with_grid": output_image,
        "grid_info": grid_info,
    }



def _filter_line_components_by_length(
    line_mask: np.ndarray,
    min_length: int,
    orientation: str,
) -> np.ndarray:
    """Keep only connected line components whose main axis length >= min_length."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    filtered = np.zeros_like(line_mask)
    axis = cv2.CC_STAT_HEIGHT if orientation == "vertical" else cv2.CC_STAT_WIDTH

    for i in range(1, num_labels):
        if int(stats[i, axis]) >= min_length:
            filtered[labels == i] = 255

    return filtered


def _align_vertical_lengths_by_row(
    vertical_mask: np.ndarray,
    row_tolerance: int = 25,
    min_group_size: int = 2,
) -> np.ndarray:
    """Extend parallel vertical lines in same row to same length, preserving angle."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical_mask, connectivity=8)

    comps: List[Dict[str, int]] = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h <= 0 or w <= 0:
            continue
        comps.append(
            {
                "idx": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cy": y + h // 2,
                "y_end": y + h,
            }
        )

    if not comps:
        return vertical_mask

    comps.sort(key=lambda c: c["cy"])
    groups: List[List[Dict[str, int]]] = []
    for comp in comps:
        placed = False
        for group in groups:
            mean_cy = int(round(sum(g["cy"] for g in group) / len(group)))
            if abs(comp["cy"] - mean_cy) <= row_tolerance:
                group.append(comp)
                placed = True
                break
        if not placed:
            groups.append([comp])

    aligned = np.zeros_like(vertical_mask)

    for group in groups:
        if len(group) < min_group_size:
            # Keep original if not enough for alignment
            for g in group:
                aligned[labels == g["idx"]] = vertical_mask[labels == g["idx"]]
            continue

        # Find target length (median of group)
        target_h = int(round(float(np.median([g["h"] for g in group]))))
        target_h = max(1, target_h)

        # For each line, extend it to target length by dilating along vertical axis
        for g in group:
            line_component = (labels == g["idx"]).astype(np.uint8) * 255
            
            # Extend vertically while preserving angle
            extend_px = max(0, target_h - g["h"])
            if extend_px > 0:
                k = 2 * extend_px + 1
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
                extended = cv2.dilate(line_component, v_kernel, iterations=1)
                aligned = cv2.bitwise_or(aligned, extended)
            else:
                aligned = cv2.bitwise_or(aligned, line_component)

    # Add any components not in groups
    for i in range(1, num_labels):
        if not any(comp["idx"] == i for group in groups for comp in group):
            aligned = cv2.bitwise_or(aligned, (labels == i).astype(np.uint8) * 255)

    return aligned



def detect_grid_points(
    image: np.ndarray,
    vertical_scale: float = 0.015,
    horizontal_scale: float = 0.015,
    min_point_area: int = 8,
    block_size: int = 35,
    block_offset: int = 7,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Detect grid intersections from morphological vertical/horizontal lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        block_offset,
    )

    h, w = gray.shape
    v_len = max(3, int(h * vertical_scale))
    h_len = max(3, int(w * horizontal_scale))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    intersections = cv2.bitwise_and(vertical_lines, horizontal_lines)
    intersections = cv2.dilate(intersections, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(intersections)
    points: List[Tuple[int, int]] = []
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] < min_point_area:
            continue
        cx, cy = centroids[idx]
        points.append((int(round(cx)), int(round(cy))))

    points.sort(key=lambda p: (p[1], p[0]))

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)

    debug_data = {
        "binary": binary,
        "vertical": vertical_lines,
        "horizontal": horizontal_lines,
        "intersections": intersections,
        "points_overlay": overlay,
        "points": points,
    }

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_binary.jpg", binary)
        cv2.imwrite(f"{debug_prefix}_vertical.jpg", vertical_lines)
        cv2.imwrite(f"{debug_prefix}_horizontal.jpg", horizontal_lines)
        cv2.imwrite(f"{debug_prefix}_intersections.jpg", intersections)
        cv2.imwrite(f"{debug_prefix}_points.jpg", overlay)

    return debug_data


def detect_boxes_in_region(
    region: np.ndarray,
    min_box_area: int = 50,
    min_box_width: int = 5,
    min_box_height: int = 5,
) -> List[np.ndarray]:
    """
    Detect small boxes within a specific image region.
    Used for detecting answer boxes inside SoBaoDanh containers.
    """
    # Apply adaptive threshold to the region
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_box_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_box_width or h < min_box_height:
            continue
        
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        boxes.append(approx)
    
    # Sort boxes by position (top-left to bottom-right)
    boxes.sort(key=lambda b: (int(b[0, 0, 1]), int(b[0, 0, 0])))
    
    return boxes


def _split_merged_boxes_for_grouping(
    boxes: List[np.ndarray],
    split_wide: bool = False,
    split_tall: bool = False,
    min_area: int = 400,
    max_area: int = 10000,
) -> List[np.ndarray]:
    """Split likely merged bubble boxes (wide or tall) to improve row grouping."""
    if not boxes:
        return boxes

    rects = [cv2.boundingRect(b) for b in boxes]
    sample_ws = [w for x, y, w, h in rects if min_area <= (w * h) <= max_area]
    sample_hs = [h for x, y, w, h in rects if min_area <= (w * h) <= max_area]

    if not sample_ws or not sample_hs:
        return boxes

    median_w = float(np.median(sample_ws))
    median_h = float(np.median(sample_hs))

    out: List[np.ndarray] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h

        if split_wide and min_area <= area <= max_area and w >= 1.75 * median_w and h <= 1.5 * median_h:
            w_left = w // 2
            w_right = w - w_left
            left_poly = np.array([[[x, y]], [[x + w_left, y]], [[x + w_left, y + h]], [[x, y + h]]], dtype=np.int32)
            right_poly = np.array([[[x + w_left, y]], [[x + w, y]], [[x + w, y + h]], [[x + w_left, y + h]]], dtype=np.int32)
            out.extend([left_poly, right_poly])
            continue

        if split_tall and min_area <= area <= max_area and h >= 1.75 * median_h and w <= 1.5 * median_w:
            h_top = h // 2
            h_bottom = h - h_top
            top_poly = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h_top]], [[x, y + h_top]]], dtype=np.int32)
            bottom_poly = np.array([[[x, y + h_top]], [[x + w, y + h_top]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
            out.extend([top_poly, bottom_poly])
            continue

        out.append(box)

    return out


def _separate_upper_id_boxes(
    boxes: List[np.ndarray],
    part_i_boxes: List[np.ndarray],
    top_margin: int = 10,
    min_area: int = 350,
    max_area: int = 6000,
    row_tolerance: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Split upper ID region boxes into SoBaoDanh (left) and MaDe (right) by X split."""
    if not boxes:
        return [], [], 0.0
    if not part_i_boxes:
        return boxes, [], 0.0

    part_i_top = min(cv2.boundingRect(b)[1] for b in part_i_boxes)
    y_limit = part_i_top - top_margin

    upper_items: List[Tuple[np.ndarray, float, float]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        if y < y_limit and min_area <= area <= max_area:
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            upper_items.append((box, cx, cy))

    if not upper_items:
        return [], [], 0.0

    rows: List[List[Tuple[np.ndarray, float, float]]] = []
    for item in sorted(upper_items, key=lambda t: t[2]):
        placed = False
        for row in rows:
            mean_cy = float(np.mean([z[2] for z in row]))
            if abs(item[2] - mean_cy) <= row_tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    split_candidates: List[float] = []
    for row in rows:
        if len(row) < 8:
            continue
        xs = sorted([z[1] for z in row])
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        if not gaps:
            continue
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] > 20:
            split_candidates.append((xs[max_gap_idx] + xs[max_gap_idx + 1]) / 2.0)

    if split_candidates:
        split_x = float(np.median(split_candidates))
    else:
        split_x = float(np.percentile([z[1] for z in upper_items], 70))

    sbd_boxes = [z[0] for z in upper_items if z[1] < split_x]
    ma_de_boxes = [z[0] for z in upper_items if z[1] >= split_x]
    return sbd_boxes, ma_de_boxes, split_x


def group_boxes_into_parts(
    boxes: List[np.ndarray],
    row_tolerance: int = 30,
    size_tolerance_ratio: float = 0.15,
    min_boxes_per_group: int = 3,
) -> Dict[str, object]:
    """
    Group detected boxes into Part I (4 boxes), Part II (8 boxes), Part III (6 boxes).
    
    Groups boxes by vertical position (rows), then identifies parts based on:
    - Part I: First group with exactly 4 boxes of same size
    - Part II: Middle group with 8 boxes
    - Part III: Last group with 6 boxes
    """
    if not boxes:
        return {"part_i": [], "part_ii": [], "part_iii": [], "all_parts": []}

    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        union = (aw * ah) + (bw * bh) - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
        return np.array(
            [
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]],
            ],
            dtype=np.int32,
        )

    def _is_uniform_size(group: List[Dict[str, object]], tol: float) -> bool:
        areas = [int(b["area"]) for b in group]
        if not areas:
            return False
        mean_area = float(np.mean(areas))
        if mean_area <= 0:
            return True
        max_rel = max(abs(a - mean_area) / mean_area for a in areas)
        return max_rel <= tol

    def _select_best_subset(group: List[Dict[str, object]], expected_count: int) -> List[Dict[str, object]]:
        if len(group) < expected_count:
            return []
        sorted_group = sorted(group, key=lambda b: int(b["x"]))
        if len(sorted_group) == expected_count:
            return sorted_group

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")

        for idx_tuple in itertools.combinations(range(len(sorted_group)), expected_count):
            subset = [sorted_group[idx] for idx in idx_tuple]
            areas = [float(b["area"]) for b in subset]
            mean_area = float(np.mean(areas))
            if mean_area <= 0:
                continue
            size_var = max(abs(a - mean_area) / mean_area for a in areas)

            xs = [int(b["x"]) for b in subset]
            widths = [int(b["w"]) for b in subset]
            centers = [x + w // 2 for x, w in zip(xs, widths)]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            if gaps:
                gap_mean = float(np.mean(gaps))
                gap_var = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0
            else:
                gap_var = 0.0

            score = size_var + 0.25 * gap_var
            if score < best_score:
                best_score = score
                best_subset = subset

        return best_subset

    # Extract bounding box info
    box_info: List[Dict[str, object]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        box_info.append({
            "box": box,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_y": y + h // 2,
            "area": area,
        })

    # Keep only large containers to avoid mixing with tiny answer bubbles.
    all_areas = np.array([int(b["area"]) for b in box_info], dtype=np.float64)
    area_threshold = max(float(np.percentile(all_areas, 80)), 30000.0)
    candidates = [b for b in box_info if float(b["area"]) >= area_threshold]
    if not candidates:
        return {"part_i": [], "part_ii": [], "part_iii": [], "all_parts": []}

    # De-duplicate heavily overlapping boxes (common when both inner/outer borders are detected).
    candidates.sort(key=lambda b: int(b["area"]), reverse=True)
    deduped: List[Dict[str, object]] = []
    for cand in candidates:
        r1 = (int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"]))
        replaced = False
        for idx, keep in enumerate(deduped):
            r2 = (int(keep["x"]), int(keep["y"]), int(keep["w"]), int(keep["h"]))
            if _bbox_iou(r1, r2) >= 0.85:
                if int(cand["area"]) > int(keep["area"]):
                    deduped[idx] = cand
                replaced = True
                break
        if not replaced:
            deduped.append(cand)

    # Group boxes by row (Y proximity). Use adaptive tolerance for perspective skew.
    median_h = float(np.median([int(b["h"]) for b in deduped])) if deduped else 0.0
    row_tol = max(row_tolerance, 45, int(median_h * 0.25))
    deduped.sort(key=lambda b: int(b["center_y"]))
    groups: List[List[Dict[str, object]]] = []
    for box in deduped:
        placed = False
        for group in groups:
            mean_y = float(np.mean([int(b["center_y"]) for b in group]))
            if abs(int(box["center_y"]) - mean_y) <= row_tol:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    groups.sort(key=lambda g: float(np.mean([int(b["center_y"]) for b in g])))

    estimated_page_height = max(int(b["y"]) + int(b["h"]) for b in deduped)

    def _group_center_ratio(group: List[Dict[str, object]]) -> float:
        cy = float(np.mean([int(b["center_y"]) for b in group]))
        return (cy / float(estimated_page_height)) if estimated_page_height > 0 else 0.0

    parts = {"part_i": [], "part_ii": [], "part_iii": []}
    group_idx = 0

    # Part I: 4 large boxes in one row.
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.25 or center_ratio > 0.65:
            continue
        subset = _select_best_subset(group, expected_count=4)
        if subset and _is_uniform_size(subset, size_tolerance_ratio * 2.0):
            parts["part_i"] = [b["box"] for b in subset]
            group_idx = i + 1
            break

    # Fallback: some scans miss one Part I contour; recover from 3 evenly spaced boxes.
    if not parts["part_i"]:
        estimated_page_width = max(int(b["x"]) + int(b["w"]) for b in box_info) if box_info else 0
        for i in range(0, len(groups)):
            group = groups[i]
            center_ratio = _group_center_ratio(group)
            if center_ratio < 0.25 or center_ratio > 0.65:
                continue

            subset3 = _select_best_subset(group, expected_count=3)
            if not subset3:
                continue
            if not _is_uniform_size(subset3, size_tolerance_ratio * 2.5):
                continue

            subset3 = sorted(subset3, key=lambda b: int(b["x"]))
            xs = [int(b["x"]) for b in subset3]
            ws = [int(b["w"]) for b in subset3]
            ys = [int(b["y"]) for b in subset3]
            hs = [int(b["h"]) for b in subset3]
            centers = [x + w // 2 for x, w in zip(xs, ws)]
            gaps = [centers[j + 1] - centers[j] for j in range(len(centers) - 1)]
            if not gaps:
                continue

            mean_gap = float(np.mean(gaps))
            if mean_gap <= 0:
                continue
            gap_cv = float(np.std(gaps) / mean_gap) if mean_gap > 0 else 1.0
            if gap_cv > 0.2:
                continue

            mean_w = int(round(float(np.mean(ws))))
            mean_h = int(round(float(np.mean(hs))))
            mean_y = int(round(float(np.mean(ys))))

            left_margin = xs[0]
            right_edge = xs[-1] + ws[-1]
            right_margin = max(0, estimated_page_width - right_edge)

            # Choose missing side based on available margin.
            if left_margin > right_margin:
                missing_center = centers[0] - int(round(mean_gap))
            else:
                missing_center = centers[-1] + int(round(mean_gap))

            missing_x = int(round(missing_center - mean_w / 2.0))
            missing_x = max(0, missing_x)
            missing_box = _rect_to_poly(missing_x, mean_y, max(1, mean_w), max(1, mean_h))

            # Prefer a real detected contour close to the inferred missing slot.
            inferred_rect = cv2.boundingRect(missing_box)
            used_ids = set(id(item["box"]) for item in subset3)
            best_detected_box: Optional[np.ndarray] = None
            best_detected_score = -1.0
            for cand in box_info:
                if id(cand["box"]) in used_ids:
                    continue

                cw = int(cand["w"])
                ch = int(cand["h"])
                if mean_w > 0 and not (0.55 * mean_w <= cw <= 1.6 * mean_w):
                    continue
                if mean_h > 0 and not (0.55 * mean_h <= ch <= 1.6 * mean_h):
                    continue

                cand_rect = (int(cand["x"]), int(cand["y"]), cw, ch)
                overlap = _bbox_iou(inferred_rect, cand_rect)

                # If IoU is weak, still allow by center proximity.
                ix, iy, iw, ih = inferred_rect
                icx = ix + (iw / 2.0)
                icy = iy + (ih / 2.0)
                ccx = int(cand["x"]) + (cw / 2.0)
                ccy = int(cand["y"]) + (ch / 2.0)
                center_dist = ((icx - ccx) ** 2 + (icy - ccy) ** 2) ** 0.5
                dist_score = 1.0 / (1.0 + center_dist)

                score = overlap + 0.15 * dist_score
                if score > best_detected_score:
                    best_detected_score = score
                    best_detected_box = cand["box"]

            if best_detected_box is not None and best_detected_score >= 0.25:
                missing_box = best_detected_box

            recovered = [b["box"] for b in subset3]
            recovered.append(missing_box)
            recovered.sort(key=lambda poly: cv2.boundingRect(poly)[0])

            parts["part_i"] = recovered
            group_idx = i + 1
            break

    # Part II: ideally 8 boxes; some scans merge each column pair into 4 tall boxes.
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.45 or center_ratio > 0.75:
            continue
        subset8 = _select_best_subset(group, expected_count=8)
        if subset8 and _is_uniform_size(subset8, size_tolerance_ratio * 2.2):
            parts["part_ii"] = [b["box"] for b in subset8]
            group_idx = i + 1
            break

        subset4 = _select_best_subset(group, expected_count=4)

        # Build parent candidates from largest boxes first (usually true outer containers).
        parent_sets: List[List[Dict[str, object]]] = []
        largest4 = sorted(group, key=lambda b: int(b["area"]), reverse=True)[:4]
        largest4 = sorted(largest4, key=lambda b: int(b["x"]))
        if len(largest4) == 4:
            parent_sets.append(largest4)
        if subset4 and _is_uniform_size(subset4, size_tolerance_ratio * 2.2):
            parent_sets.append(subset4)

        detected_part_ii: List[np.ndarray] = []
        selected_parent_set: List[Dict[str, object]] = []

        for parent_set in parent_sets:
            # Prefer 8 boxes that were truly detected inside these 4 merged containers.
            local_detected: List[np.ndarray] = []
            for parent in parent_set:
                px = int(parent["x"])
                py = int(parent["y"])
                pw = int(parent["w"])
                ph = int(parent["h"])
                p_area = float(parent["area"])

                child_candidates: List[Dict[str, object]] = []
                for cand in box_info:
                    cx = int(cand["x"])
                    cy = int(cand["y"])
                    cw = int(cand["w"])
                    ch = int(cand["h"])
                    ca = float(cand["area"])

                    # Candidate must be fully inside parent (with small margin) and smaller than parent.
                    if cx < px + 2 or cy < py + 2:
                        continue
                    if cx + cw > px + pw - 2 or cy + ch > py + ph - 2:
                        continue
                    if ca >= p_area * 0.9:
                        continue

                    # Part II inner boxes are typically around half parent width and near full height.
                    width_ratio = float(cw) / float(pw) if pw > 0 else 0.0
                    height_ratio = float(ch) / float(ph) if ph > 0 else 0.0
                    if not (0.30 <= width_ratio <= 0.70):
                        continue
                    if not (0.70 <= height_ratio <= 1.05):
                        continue

                    child_candidates.append(cand)

                child_candidates.sort(key=lambda b: (int(b["x"]), -int(b["area"])))

                # Keep best non-overlapping children by X; expect 2 (left/right).
                selected_children: List[Dict[str, object]] = []
                for cand in child_candidates:
                    overlap_x = False
                    for chosen in selected_children:
                        c1x, c1w = int(cand["x"]), int(cand["w"])
                        c2x, c2w = int(chosen["x"]), int(chosen["w"])
                        left = max(c1x, c2x)
                        right = min(c1x + c1w, c2x + c2w)
                        if right > left:
                            inter_w = right - left
                            min_w = max(1, min(c1w, c2w))
                            if (inter_w / float(min_w)) > 0.5:
                                overlap_x = True
                                break
                    if not overlap_x:
                        selected_children.append(cand)
                    if len(selected_children) == 2:
                        break

                if len(selected_children) == 2:
                    local_detected.extend([selected_children[0]["box"], selected_children[1]["box"]])

            if len(local_detected) == 8:
                detected_part_ii = local_detected
                selected_parent_set = parent_set
                break

        if len(detected_part_ii) == 8:
            parts["part_ii"] = detected_part_ii
            group_idx = i + 1
            break

        if selected_parent_set:
            subset4 = selected_parent_set

        if subset4 and _is_uniform_size(subset4, size_tolerance_ratio * 2.2):

            # Split each merged detected box into left/right halves (Cau 1-2, 3-4, ...).
            split_boxes: List[np.ndarray] = []
            for item in subset4:
                x = int(item["x"])
                y = int(item["y"])
                w = int(item["w"])
                h = int(item["h"])
                w_left = w // 2
                w_right = w - w_left
                split_boxes.append(_rect_to_poly(x, y, w_left, h))
                split_boxes.append(_rect_to_poly(x + w_left, y, w_right, h))
            parts["part_ii"] = split_boxes
            group_idx = i + 1
            break

    # Part III: 6 boxes in one row (lower section).
    for i in range(group_idx, len(groups)):
        group = groups[i]
        center_ratio = _group_center_ratio(group)
        if center_ratio < 0.60:
            continue
        subset = _select_best_subset(group, expected_count=6)
        if subset and _is_uniform_size(subset, size_tolerance_ratio * 2.5):
            parts["part_iii"] = [b["box"] for b in subset]
            group_idx = i + 1
            break

    # Fallback: recover Part III from all detected boxes when large-box filtering
    # drops valid columns (common in noisy scans with mixed contour sizes).
    if not parts["part_iii"]:
        all_row_tol = max(row_tolerance, 25)
        all_groups: List[List[Dict[str, object]]] = []
        for item in sorted(box_info, key=lambda b: int(b["center_y"])):
            placed = False
            for group in all_groups:
                mean_y = float(np.mean([int(b["center_y"]) for b in group]))
                if abs(int(item["center_y"]) - mean_y) <= all_row_tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                all_groups.append([item])

        full_page_height = max(int(b["y"]) + int(b["h"]) for b in box_info) if box_info else 0
        part_i_ref_area = None
        if parts["part_i"]:
            part_i_ref_area = float(
                np.mean([
                    cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3]
                    for b in parts["part_i"]
                ])
            )

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")

        for group in all_groups:
            subset6 = _select_best_subset(group, expected_count=6)
            if not subset6:
                continue
            if not _is_uniform_size(subset6, size_tolerance_ratio * 3.0):
                continue

            subset_center = float(np.mean([int(b["center_y"]) for b in subset6]))
            center_ratio = (subset_center / float(full_page_height)) if full_page_height > 0 else 0.0
            if center_ratio < 0.58:
                continue

            areas = [float(b["area"]) for b in subset6]
            area_mean = float(np.mean(areas)) if areas else 0.0
            if area_mean <= 0:
                continue
            if area_mean < 2500:
                continue
            if part_i_ref_area is not None and area_mean < part_i_ref_area * 0.10:
                continue

            widths = [int(b["w"]) for b in subset6]
            median_w = float(np.median(widths)) if widths else 0.0
            if median_w <= 0:
                continue
            if max(widths) > median_w * 2.2:
                continue

            subset_sorted = sorted(subset6, key=lambda b: int(b["x"]))
            centers = [int(b["x"]) + int(b["w"]) // 2 for b in subset_sorted]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            gap_mean = float(np.mean(gaps)) if gaps else 0.0
            gap_cv = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0
            area_cv = float(np.std(areas) / area_mean) if area_mean > 0 else 1.0

            # Prefer stable geometry and rows closer to the lower section.
            score = area_cv + 0.2 * gap_cv - 0.0008 * subset_center - 0.00001 * area_mean
            if score < best_score:
                best_score = score
                best_subset = subset_sorted

        if best_subset:
            parts["part_iii"] = [b["box"] for b in best_subset]

    # Fallback: some scans merge the whole Part III row into one large container.
    if not parts["part_iii"]:
        best_container: Optional[Dict[str, object]] = None
        best_score = -1.0
        for group in groups:
            if not group:
                continue
            center_ratio = _group_center_ratio(group)
            if center_ratio < 0.65:
                continue

            # Prefer large lower single-box groups as Part III container candidates.
            if len(group) == 1:
                item = group[0]
                w = float(item["w"])
                h = float(item["h"])
                if h <= 0:
                    continue
                aspect = w / h
                if aspect < 1.8:
                    continue

                score = float(item["area"]) * (1.0 + 0.1 * aspect)
                if score > best_score:
                    best_score = score
                    best_container = item

        if best_container is not None:
            x = int(best_container["x"])
            y = int(best_container["y"])
            w = int(best_container["w"])
            h = int(best_container["h"])
            if w >= 6 and h >= 20:
                base_w = w // 6
                rem_w = w - (base_w * 6)
                split_boxes: List[np.ndarray] = []
                cur_x = x
                for idx in range(6):
                    col_w = base_w + (1 if idx < rem_w else 0)
                    col_w = max(1, col_w)
                    split_boxes.append(_rect_to_poly(cur_x, y, col_w, h))
                    cur_x += col_w
                parts["part_iii"] = split_boxes

    # Fallback for scans where Part II boxes are much smaller than Part I/III and
    # were excluded by the large-container area threshold.
    if not parts["part_ii"]:
        all_row_tol = max(row_tolerance, 25)
        all_groups: List[List[Dict[str, object]]] = []
        for item in sorted(box_info, key=lambda b: int(b["center_y"])):
            placed = False
            for group in all_groups:
                mean_y = float(np.mean([int(b["center_y"]) for b in group]))
                if abs(int(item["center_y"]) - mean_y) <= all_row_tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                all_groups.append([item])

        part_i_center = None
        if parts["part_i"]:
            part_i_center = float(np.mean([cv2.boundingRect(b)[1] + cv2.boundingRect(b)[3] / 2.0 for b in parts["part_i"]]))
        part_iii_center = None
        if parts["part_iii"]:
            part_iii_center = float(np.mean([cv2.boundingRect(b)[1] + cv2.boundingRect(b)[3] / 2.0 for b in parts["part_iii"]]))

        ref_area = None
        ref_areas: List[float] = []
        if parts["part_i"]:
            ref_areas.append(float(np.mean([cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3] for b in parts["part_i"]])))
        if parts["part_iii"]:
            ref_areas.append(float(np.mean([cv2.boundingRect(b)[2] * cv2.boundingRect(b)[3] for b in parts["part_iii"]])))
        if ref_areas:
            ref_area = min(ref_areas)

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")
        target_center = None
        if part_i_center is not None and part_iii_center is not None:
            target_center = (part_i_center + part_iii_center) / 2.0

        for group in all_groups:
            subset8 = _select_best_subset(group, expected_count=8)
            if not subset8:
                continue
            if not _is_uniform_size(subset8, size_tolerance_ratio * 3.0):
                continue

            subset_center = float(np.mean([int(b["center_y"]) for b in subset8]))
            if part_i_center is not None and subset_center <= part_i_center + all_row_tol:
                continue
            if part_iii_center is not None and subset_center >= part_iii_center - all_row_tol:
                continue

            subset_area = float(np.mean([int(b["area"]) for b in subset8]))
            if ref_area is not None:
                # Part II boxes are often smaller than Part I/III, but not tiny noise.
                if subset_area < ref_area * 0.08:
                    continue
                if subset_area > ref_area * 0.95:
                    continue

            xs = [int(b["x"]) for b in subset8]
            ws = [int(b["w"]) for b in subset8]
            centers = [x + w // 2 for x, w in zip(xs, ws)]
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            gap_mean = float(np.mean(gaps)) if gaps else 0.0
            gap_var = float(np.std(gaps) / gap_mean) if gap_mean > 0 else 1.0

            areas = [float(b["area"]) for b in subset8]
            area_mean = float(np.mean(areas))
            area_var = max(abs(a - area_mean) / area_mean for a in areas) if area_mean > 0 else 1.0

            center_penalty = 0.0
            if target_center is not None and estimated_page_height > 0:
                center_penalty = abs(subset_center - target_center) / float(estimated_page_height)

            score = area_var + 0.3 * gap_var + 0.8 * center_penalty
            if score < best_score:
                best_score = score
                best_subset = subset8

        if best_subset:
            parts["part_ii"] = [b["box"] for b in best_subset]

    parts["all_parts"] = parts["part_i"] + parts["part_ii"] + parts["part_iii"]
    return parts


def _rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]],
        ],
        dtype=np.int32,
    )


def _build_box_info(boxes: List[np.ndarray]) -> List[Dict[str, object]]:
    box_info: List[Dict[str, object]] = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        box_info.append(
            {
                "box": box,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "center_y": y + h // 2,
                "area": w * h,
            }
        )
    return box_info


def _group_box_info_by_row(
    box_info: List[Dict[str, object]],
    row_tolerance: int,
) -> List[List[Dict[str, object]]]:
    sorted_info = sorted(box_info, key=lambda b: b["center_y"])
    groups: List[List[Dict[str, object]]] = []
    for box in sorted_info:
        placed = False
        for group in groups:
            mean_y = np.mean([b["center_y"] for b in group])
            if abs(box["center_y"] - mean_y) <= row_tolerance:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    groups.sort(key=lambda g: np.mean([b["center_y"] for b in g]))
    return groups


def _is_uniform_size_group(group: List[Dict[str, object]], size_tolerance_ratio: float) -> bool:
    areas = [b["area"] for b in group]
    mean_area = np.mean(areas)
    return max([abs(a - mean_area) / mean_area for a in areas]) <= size_tolerance_ratio if mean_area > 0 else True


def _filter_rows_by_global_size_consistency(
    rows: List[List[np.ndarray]],
    size_tolerance_ratio: float,
    debug: bool = False,
) -> List[List[np.ndarray]]:
    if not rows:
        return rows

    all_areas: List[int] = []
    for row in rows:
        for box in row:
            _, _, w, h = cv2.boundingRect(box)
            all_areas.append(w * h)

    if not all_areas:
        return rows

    global_mean_area = np.mean(all_areas)
    if global_mean_area <= 0:
        return rows

    filtered_rows: List[List[np.ndarray]] = []
    for row_idx, row in enumerate(rows):
        row_areas: List[int] = []
        for box in row:
            _, _, w, h = cv2.boundingRect(box)
            row_areas.append(w * h)

        row_mean_area = np.mean(row_areas) if row_areas else 0
        row_variance = abs(row_mean_area - global_mean_area) / global_mean_area if global_mean_area > 0 else 0
        if row_variance <= size_tolerance_ratio:
            filtered_rows.append(row)
        elif debug:
            print(
                f"  ✗ Row {row_idx + 1} filtered out: row_avg={int(row_mean_area)}, "
                f"global_avg={int(global_mean_area)}, variance={row_variance:.2f}"
            )

    return filtered_rows


def _trim_rows_to_consistent_window(
    rows: List[List[np.ndarray]],
    max_rows: int,
) -> List[List[np.ndarray]]:
    if len(rows) <= max_rows:
        return rows

    row_infos = []
    for row in rows:
        y_vals = [cv2.boundingRect(box)[1] for box in row]
        a_vals = [cv2.boundingRect(box)[2] * cv2.boundingRect(box)[3] for box in row]
        row_infos.append(
            {
                "row": row,
                "mean_y": float(np.mean(y_vals)) if y_vals else 0.0,
                "mean_area": float(np.mean(a_vals)) if a_vals else 0.0,
            }
        )

    row_infos.sort(key=lambda r: r["mean_y"])
    best_start = 0
    best_score = float("inf")

    for start in range(0, len(row_infos) - max_rows + 1):
        window = row_infos[start:start + max_rows]
        ys = [r["mean_y"] for r in window]
        areas = [r["mean_area"] for r in window]

        if len(ys) >= 2:
            spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            mean_spacing = float(np.mean(spacings)) if spacings else 0.0
            spacing_cv = float(np.std(spacings) / mean_spacing) if mean_spacing > 0 else 1.0
        else:
            spacing_cv = 1.0

        mean_area = float(np.mean(areas)) if areas else 0.0
        area_cv = float(np.std(areas) / mean_area) if mean_area > 0 else 1.0
        score = spacing_cv + 0.2 * area_cv - 0.0005 * float(np.mean(ys))
        if score < best_score:
            best_score = score
            best_start = start

    return [r["row"] for r in row_infos[best_start:best_start + max_rows]]


def detect_sobao_danh_boxes(
    boxes: List[np.ndarray],
    boxes_per_row: int = 6,
    max_rows: int = 10,
    row_tolerance: int = 30,
    size_tolerance_ratio: float = 0.3,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Detect and group SoBaoDanh (roll number) boxes.
    
    SoBaoDanh typically has:
    - 6 boxes per row
    - Up to 10 consecutive rows
    - Uniform box sizes within each row (relaxed tolerance for various document qualities)
    
    Returns:
        Dictionary with:
        - 'sobao_danh': List of all detected SoBaoDanh boxes
        - 'sobao_danh_rows': List of rows, each row containing 6 boxes
        - 'row_count': Number of detected rows
    """
    if not boxes:
        return {
            "sobao_danh": [],
            "sobao_danh_rows": [],
            "row_count": 0,
        }
    
    box_info = _build_box_info(boxes)
    groups = _group_box_info_by_row(box_info, row_tolerance)
    
    if debug:
        print(f"\n[DEBUG] Total groups: {len(groups)}")
        for idx, group in enumerate(groups):
            areas = [cv2.boundingRect(b["box"])[2] * cv2.boundingRect(b["box"])[3] for b in group]
            mean_area = np.mean(areas) if areas else 0
            print(f"  Group {idx}: {len(group)} boxes, areas={[int(a) for a in areas]}, mean={int(mean_area)}")
    
    # Helper function to check size uniformity
    def is_uniform_size(group: List[Dict[str, object]]) -> bool:
        return _is_uniform_size_group(group, size_tolerance_ratio)

    def _try_recover_merged_row(group: List[Dict[str, object]]) -> Optional[List[np.ndarray]]:
        # Recovery for rows where one merged contour combines two adjacent bubbles
        # and the row appears as 5 boxes instead of 6.
        if len(group) != boxes_per_row - 1:
            return None

        sorted_group = sorted(group, key=lambda b: b["x"])
        widths = [int(b["w"]) if "w" in b else cv2.boundingRect(b["box"])[2] for b in sorted_group]
        heights = [int(b["h"]) if "h" in b else cv2.boundingRect(b["box"])[3] for b in sorted_group]
        if not widths or not heights:
            return None

        median_w = float(np.median(widths))
        median_h = float(np.median(heights))
        if median_w <= 0 or median_h <= 0:
            return None

        merged_idx = int(np.argmax(widths))
        merged_item = sorted_group[merged_idx]
        merged_w = widths[merged_idx]
        merged_h = heights[merged_idx]

        # Require a clearly wider contour but with comparable height.
        if merged_w < median_w * 1.45:
            return None
        if not (0.75 * median_h <= merged_h <= 1.35 * median_h):
            return None

        mx = int(merged_item["x"])
        my = int(merged_item["y"])
        mh = int(merged_item["h"]) if "h" in merged_item else merged_h
        left_w = merged_w // 2
        right_w = merged_w - left_w
        if left_w <= 0 or right_w <= 0:
            return None

        left_poly = _rect_to_poly(mx, my, left_w, mh)
        right_poly = _rect_to_poly(mx + left_w, my, right_w, mh)

        candidate_boxes: List[np.ndarray] = []
        for i, item in enumerate(sorted_group):
            if i == merged_idx:
                candidate_boxes.extend([left_poly, right_poly])
            else:
                candidate_boxes.append(item["box"])

        if len(candidate_boxes) != boxes_per_row:
            return None

        candidate_boxes = sorted(candidate_boxes, key=lambda b: cv2.boundingRect(b)[0])
        temp_group = []
        for box in candidate_boxes:
            x, y, w, h = cv2.boundingRect(box)
            temp_group.append({"box": box, "x": x, "y": y, "area": w * h})

        if not is_uniform_size(temp_group):
            return None

        return candidate_boxes
    
    # Find all rows with exactly boxes_per_row (6) boxes
    # Also try to extract 6-box sub-rows from larger groups
    sobao_danh_rows: List[List[np.ndarray]] = []
    len_minus_one_groups: List[List[Dict[str, object]]] = []
    
    for idx, group in enumerate(groups):
        # Only accept rows with exactly boxes_per_row boxes and uniform size
        if len(group) == boxes_per_row and is_uniform_size(group):
            sorted_boxes = [b["box"] for b in sorted(group, key=lambda b: b["x"])]
            sobao_danh_rows.append(sorted_boxes)
            if debug:
                print(f"  ✓ Group {idx} matched as SoBaoDanh row {len(sobao_danh_rows)}")
        elif len(group) > boxes_per_row:
            # Try to extract non-overlapping 6-box rows from larger groups
            sorted_group = sorted(group, key=lambda b: b["x"])
            used_indices = set()
            
            for start_idx in range(len(sorted_group) - boxes_per_row + 1):
                # Skip if we've already used any of these boxes
                if any(i in used_indices for i in range(start_idx, start_idx + boxes_per_row)):
                    continue
                
                sub_group = sorted_group[start_idx:start_idx + boxes_per_row]
                
                if is_uniform_size(sub_group):
                    sorted_boxes = [b["box"] for b in sub_group]
                    sobao_danh_rows.append(sorted_boxes)
                    # Mark these boxes as used
                    for i in range(start_idx, start_idx + boxes_per_row):
                        used_indices.add(i)
                    if debug:
                        print(f"  ✓ Group {idx} sub-row extracted as SoBaoDanh row {len(sobao_danh_rows)}")

            if debug and len(group) <= 15:
                print(f"  ✗ Group {idx} rejected: len={len(group)} (need 6), no valid non-overlapping 6-box sub-rows")
        elif len(group) == boxes_per_row - 1:
            len_minus_one_groups.append(group)
            recovered = _try_recover_merged_row(group)
            if recovered is not None:
                sobao_danh_rows.append(recovered)
                if debug:
                    print(f"  ✓ Group {idx} recovered as SoBaoDanh row {len(sobao_danh_rows)} (split merged box)")
        elif debug and len(group) <= 10:
            areas = [b["area"] for b in group]
            mean_area = np.mean(areas) if areas else 0
            variance = max([abs(a - mean_area) / mean_area for a in areas]) if mean_area > 0 else 0
            print(f"  ✗ Group {idx} rejected: len={len(group)} (need 6), variance={variance:.2f} (max {size_tolerance_ratio})")
    
    sobao_danh_rows = _filter_rows_by_global_size_consistency(
        sobao_danh_rows,
        size_tolerance_ratio,
        debug=debug,
    )
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    sobao_danh_rows = _trim_rows_to_consistent_window(sobao_danh_rows, max_rows)
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    # Handle a common pattern: one unrelated header-like top row plus one valid
    # row detected as only 5 boxes due a missing/merged contour.
    if len(sobao_danh_rows) == max_rows and len_minus_one_groups:
        row_with_y = []
        for row in sobao_danh_rows:
            ys = [cv2.boundingRect(box)[1] for box in row]
            row_with_y.append((float(np.mean(ys)) if ys else 0.0, row))
        row_with_y.sort(key=lambda t: t[0])

        ys_only = [t[0] for t in row_with_y]
        if len(ys_only) >= 3:
            gaps = [ys_only[i + 1] - ys_only[i] for i in range(len(ys_only) - 1)]
            tail_gaps = gaps[1:] if len(gaps) > 1 else gaps
            median_tail_gap = float(np.median(tail_gaps)) if tail_gaps else 0.0

            top_gap_is_outlier = median_tail_gap > 1 and gaps[0] > (median_tail_gap * 1.6)
            if top_gap_is_outlier:
                if debug:
                    print(
                        f"  • Top SBD row looks like outlier (first gap={gaps[0]:.1f}, "
                        f"median tail gap={median_tail_gap:.1f}); attempting replacement from len-5 row"
                    )

                kept_rows = [row for _, row in row_with_y[1:]]
                template_rows = kept_rows[:]

                if template_rows:
                    col_x_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
                    col_w_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
                    h_list: List[int] = []

                    for row in template_rows:
                        rects = sorted([cv2.boundingRect(b) for b in row], key=lambda r: r[0])
                        if len(rects) != boxes_per_row:
                            continue
                        for c, (rx, ry, rw, rh) in enumerate(rects):
                            col_x_lists[c].append(int(rx))
                            col_w_lists[c].append(int(rw))
                            h_list.append(int(rh))

                    if all(col_x_lists[c] for c in range(boxes_per_row)) and h_list:
                        col_x = [int(round(float(np.median(col_x_lists[c])))) for c in range(boxes_per_row)]
                        col_w = [int(round(float(np.median(col_w_lists[c])))) for c in range(boxes_per_row)]
                        row_h = max(1, int(round(float(np.median(h_list)))))

                        def rect_to_poly(x: int, y: int, w: int, h: int) -> np.ndarray:
                            return np.array(
                                [
                                    [[x, y]],
                                    [[x + w, y]],
                                    [[x + w, y + h]],
                                    [[x, y + h]],
                                ],
                                dtype=np.int32,
                            )

                        expected_y = ys_only[1] - median_tail_gap
                        best_group = None
                        best_dist = float("inf")
                        for g in len_minus_one_groups:
                            gy = float(np.mean([int(it["y"]) for it in g])) if g else 0.0
                            dist = abs(gy - expected_y)
                            if dist < best_dist:
                                best_dist = dist
                                best_group = g

                        replacement_row: Optional[List[np.ndarray]] = None
                        if best_group is not None and best_dist <= max(35.0, median_tail_gap * 0.9):
                            g_sorted = sorted(best_group, key=lambda it: int(it["x"]))
                            assigned: List[Optional[np.ndarray]] = [None] * boxes_per_row
                            used_cols = set()

                            for it in g_sorted:
                                box = it["box"]
                                x, y, w, h = cv2.boundingRect(box)
                                best_col = None
                                best_col_dist = float("inf")
                                for c in range(boxes_per_row):
                                    if c in used_cols:
                                        continue
                                    dist = abs(x - col_x[c])
                                    if dist < best_col_dist:
                                        best_col_dist = dist
                                        best_col = c
                                if best_col is not None:
                                    assigned[best_col] = box
                                    used_cols.add(best_col)

                            row_y = int(round(float(np.mean([int(it["y"]) for it in g_sorted]))))
                            for c in range(boxes_per_row):
                                if assigned[c] is None:
                                    assigned[c] = _rect_to_poly(col_x[c], row_y, max(1, col_w[c]), row_h)

                            replacement_row = [box for box in assigned if box is not None]

                        if replacement_row is not None and len(replacement_row) == boxes_per_row:
                            kept_rows.append(replacement_row)
                            kept_rows.sort(key=lambda row: float(np.mean([cv2.boundingRect(b)[1] for b in row])))
                            sobao_danh_rows = kept_rows[:max_rows]
                            sobao_danh = [box for row in sobao_danh_rows for box in row]
                            if debug:
                                print("  ✓ Replaced top outlier SBD row using len-5 recovery")
                        else:
                            sobao_danh_rows = kept_rows
                            sobao_danh = [box for row in sobao_danh_rows for box in row]
                            if debug:
                                print("  • Dropped top outlier SBD row (no suitable len-5 replacement)")
    
    return {
        "sobao_danh": sobao_danh,
        "sobao_danh_rows": sobao_danh_rows,
        "row_count": len(sobao_danh_rows),
    }


def detect_ma_de_boxes(
    boxes: List[np.ndarray],
    boxes_per_row: int = 3,
    max_rows: int = 10,
    row_tolerance: int = 15,
    size_tolerance_ratio: float = 0.3,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Detect and group MaDe (code/exam ID) boxes.
    
    MaDe typically has:
    - 3 boxes per row
    - Up to 10 rows max
    - Uniform box sizes within each row
    
    Returns:
        Dictionary with:
        - 'ma_de': List of all detected MaDe boxes
        - 'ma_de_rows': List of rows, each row containing 3 boxes
        - 'row_count': Number of detected rows
    """
    if not boxes:
        return {
            "ma_de": [],
            "ma_de_rows": [],
            "row_count": 0,
        }
    
    box_info = _build_box_info(boxes)
    groups = _group_box_info_by_row(box_info, row_tolerance)
    
    if debug:
        print(f"\n[DEBUG] MaDe Detection - Total groups: {len(groups)}")
        for idx, group in enumerate(groups):
            areas = [cv2.boundingRect(b["box"])[2] * cv2.boundingRect(b["box"])[3] for b in group]
            mean_area = np.mean(areas) if areas else 0
            print(f"  Group {idx}: {len(group)} boxes, areas={[int(a) for a in areas]}, mean={int(mean_area)}")
    
    # Helper function to check size uniformity
    def is_uniform_size(group: List[Dict[str, object]]) -> bool:
        return _is_uniform_size_group(group, size_tolerance_ratio)
    
    # Find all rows with exactly boxes_per_row (3) boxes
    # Also try to extract valid 3-box rows from larger groups (non-overlapping)
    ma_de_rows: List[List[np.ndarray]] = []
    
    for idx, group in enumerate(groups):
        # Only accept rows with exactly boxes_per_row boxes and uniform size
        if len(group) == boxes_per_row and is_uniform_size(group):
            sorted_boxes = [b["box"] for b in sorted(group, key=lambda b: b["x"])]
            ma_de_rows.append(sorted_boxes)
            if debug:
                print(f"  ✓ Group {idx} matched as MaDe row {len(ma_de_rows)}")
        elif len(group) > boxes_per_row:
            # Try to extract non-overlapping 3-box rows from larger groups
            sorted_group = sorted(group, key=lambda b: b["x"])
            used_indices = set()
            
            for start_idx in range(len(sorted_group) - boxes_per_row + 1):
                # Skip if we've already used any of these boxes
                if any(i in used_indices for i in range(start_idx, start_idx + boxes_per_row)):
                    continue
                
                sub_group = sorted_group[start_idx:start_idx + boxes_per_row]
                
                if is_uniform_size(sub_group):
                    sorted_boxes = [b["box"] for b in sub_group]
                    ma_de_rows.append(sorted_boxes)
                    # Mark these boxes as used
                    for i in range(start_idx, start_idx + boxes_per_row):
                        used_indices.add(i)
                    if debug:
                        print(f"  ✓ Group {idx} sub-row extracted as MaDe row {len(ma_de_rows)}")
            
            if debug and len(group) <= 15:
                print(f"  ✗ Group {idx} rejected: len={len(group)} (need 3), no valid non-overlapping 3-box sub-rows")
        elif debug and len(group) <= 10:
            areas = [b["area"] for b in group]
            mean_area = np.mean(areas) if areas else 0
            variance = max([abs(a - mean_area) / mean_area for a in areas]) if mean_area > 0 else 0
            print(f"  ✗ Group {idx} rejected: len={len(group)} (need 3), variance={variance:.2f} (max {size_tolerance_ratio})")
    
    ma_de_rows = _filter_rows_by_global_size_consistency(
        ma_de_rows,
        size_tolerance_ratio,
        debug=debug,
    )
    ma_de = [box for row in ma_de_rows for box in row]

    ma_de_rows = _trim_rows_to_consistent_window(ma_de_rows, max_rows)
    ma_de = [box for row in ma_de_rows for box in row]
    
    return {
        "ma_de": ma_de,
        "ma_de_rows": ma_de_rows,
        "row_count": len(ma_de_rows),
    }


def detect_boxes_from_morph_lines(
    image: np.ndarray,
    vertical_scale: float = 0.015,
    horizontal_scale: float = 0.015,
    min_line_length: int = 30,
    align_vertical_rows: bool = True,
    vertical_row_tolerance: int = 25,
    block_size: int = 35,
    block_offset: int = 7,
    min_box_area: int = 100,
    min_box_width: int = 8,
    min_box_height: int = 8,
    close_kernel_size: int = 6,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """
    Detect enclosed boxes formed by morphology-detected horizontal/vertical lines.

    Steps:
    1. Detect vertical/horizontal line masks.
    2. Merge line masks and close small gaps on line network.
    3. Flood-fill the outer background to isolate enclosed regions.
    4. Connected components on enclosed regions -> cell interior boxes.
    """
    grid = detect_grid_points(
        image=image,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
        min_point_area=8,
        block_size=block_size,
        block_offset=block_offset,
        debug_prefix=None,
    )

    vertical = grid["vertical"]
    horizontal = grid["horizontal"]

    # Only line segments >= min_line_length are allowed to form boxes.
    vertical = _filter_line_components_by_length(vertical, min_line_length, "vertical")
    horizontal = _filter_line_components_by_length(horizontal, min_line_length, "horizontal")

    # Align parallel vertical lines in same row to identical start/end positions.
    if align_vertical_rows:
        vertical = _align_vertical_lengths_by_row(
            vertical,
            row_tolerance=vertical_row_tolerance,
            min_group_size=2,
        )

    lines = cv2.bitwise_or(vertical, horizontal)
    k = max(1, close_kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    lines_closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    inv = cv2.bitwise_not(lines_closed)

    h, w = inv.shape
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=128)

    enclosed = np.where(flood == 255, 255, 0).astype(np.uint8)

    # Extract contours (actual polygon shapes) instead of axis-aligned rectangles
    contours, _ = cv2.findContours(enclosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[np.ndarray] = []  # Store contour polygons instead of (x,y,w,h)
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_box_area:
            continue
        
        # Approximate contour to a polygon with tighter epsilon for more vertices
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding box for size filtering
        x, y, bw, bh = cv2.boundingRect(approx)
        
        if bw < min_box_width or bh < min_box_height:
            continue

        boxes.append(approx)

    # Sort boxes by first vertex position
    boxes.sort(key=lambda b: (int(b[0, 0, 1]), int(b[0, 0, 0])))

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for poly in boxes:
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)

    result = {
        "vertical": vertical,
        "horizontal": horizontal,
        "lines": lines,
        "lines_closed": lines_closed,
        "enclosed": enclosed,
        "boxes_overlay": overlay,
        "boxes": boxes,
    }

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_vertical.jpg", vertical)
        cv2.imwrite(f"{debug_prefix}_horizontal.jpg", horizontal)
        cv2.imwrite(f"{debug_prefix}_lines.jpg", lines)
        # cv2.imwrite(f"{debug_prefix}_lines_closed.jpg", lines_closed)
        # cv2.imwrite(f"{debug_prefix}_enclosed.jpg", enclosed)
        cv2.imwrite(f"{debug_prefix}_boxes.jpg", overlay)

    return result


def extrapolate_missing_rows(
    detection_results: Dict[str, object],
    target_rows: int = 10,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Extrapolate missing rows for SoBaoDanh and MaDe based on row spacing.
    
    Assumes all rows (SoBaoDanh and MaDe) are vertically aligned and
    should be evenly spaced. Fills in missing rows to reach target_rows count.
    
    Args:
        detection_results: Dictionary with 'sobao_danh_rows' and 'ma_de_rows'
        target_rows: Target number of rows (default 10)
        debug: Print debug information
    
    Returns:
        Updated detection_results with extrapolated rows
    """
    sobao_danh_rows = detection_results.get("sobao_danh_rows", [])
    ma_de_rows = detection_results.get("ma_de_rows", [])
    
    # Calculate row positions (average Y of first box in each row)
    sobao_y_positions = []
    for row in sobao_danh_rows:
        if row:
            y_sum = sum(cv2.boundingRect(box)[1] for box in row)
            avg_y = y_sum // len(row)
            sobao_y_positions.append(avg_y)
    
    ma_de_y_positions = []
    for row in ma_de_rows:
        if row:
            y_sum = sum(cv2.boundingRect(box)[1] for box in row)
            avg_y = y_sum // len(row)
            ma_de_y_positions.append(avg_y)
    
    if debug:
        print(f"\n[DEBUG] SoBaoDanh Y positions: {sobao_y_positions}")
        print(f"[DEBUG] MaDe Y positions: {ma_de_y_positions}")
    
    # Use SoBaoDanh Y positions as reference (since it has all 10 rows detected)
    # If SoBaoDanh has 10 rows, use them directly
    if len(sobao_y_positions) == target_rows:
        reference_positions = sobao_y_positions
        if debug:
            print(f"[DEBUG] Using SoBaoDanh Y positions directly: {reference_positions}")
    else:
        # If SoBaoDanh doesn't have all rows, calculate expected positions
        # Calculate average row spacing from the section with more rows
        spacing_positions = sobao_y_positions if len(sobao_y_positions) >= len(ma_de_y_positions) else ma_de_y_positions
        
        if len(spacing_positions) >= 2:
            spacings = [spacing_positions[i + 1] - spacing_positions[i] for i in range(len(spacing_positions) - 1)]
            avg_spacing = int(np.mean(spacings)) if spacings else 0
        else:
            avg_spacing = 0
        
        if spacing_positions:
            first_row_y = spacing_positions[0]
            reference_positions = [first_row_y + i * avg_spacing for i in range(target_rows)]
            if debug:
                print(f"[DEBUG] Calculated reference positions: {reference_positions}")
        else:
            # No reference rows available: return a complete empty-aligned structure
            # so downstream callers can still read summary keys safely.
            result = detection_results.copy()
            result["sobao_danh_rows_aligned"] = [None] * target_rows
            result["ma_de_rows_aligned"] = [None] * target_rows
            result["reference_positions"] = []
            result["sobao_y_positions"] = sobao_y_positions
            result["ma_de_y_positions"] = ma_de_y_positions
            result["sobao_missing_count"] = target_rows
            result["ma_de_missing_count"] = target_rows
            result["sobao_detected_count"] = 0
            result["ma_de_detected_count"] = 0
            return result
    
    # Map detected MaDe rows to SoBaoDanh Y positions
    def align_rows_to_reference_positions(detected_rows, detected_y_positions, reference_positions, name="", tolerance=20):
        """Align detected rows to reference positions (SoBaoDanh Y positions)."""
        aligned = [None] * len(reference_positions)
        
        for row_idx, (row, y_pos) in enumerate(zip(detected_rows, detected_y_positions)):
            # Prefer nearest available reference slot; avoid overwriting existing row.
            sorted_indices = sorted(
                range(len(reference_positions)),
                key=lambda i: abs(reference_positions[i] - y_pos),
            )

            assigned_idx = None
            for idx in sorted_indices:
                distance = abs(reference_positions[idx] - y_pos)
                if distance > tolerance:
                    break
                if aligned[idx] is None:
                    assigned_idx = idx
                    break

            # If all nearby slots are occupied, keep the closest one only when it is empty.
            if assigned_idx is not None:
                aligned[assigned_idx] = row
                if debug:
                    print(f"  {name} row {row_idx} (Y={y_pos}) -> position {assigned_idx} (reference Y={reference_positions[assigned_idx]})")
            else:
                closest_idx = sorted_indices[0] if sorted_indices else -1
                distance = abs(reference_positions[closest_idx] - y_pos) if closest_idx >= 0 else 9999
                if debug:
                    print(f"  {name} row {row_idx} (Y={y_pos}) -> UNALIGNED (distance={distance} > {tolerance} or slot occupied)")
        
        return aligned
    
    # Align both SoBaoDanh and MaDe rows to reference positions
    aligned_sobao = align_rows_to_reference_positions(
        sobao_danh_rows, sobao_y_positions, reference_positions, "SoBaoDanh", tolerance=30)
    aligned_ma_de = align_rows_to_reference_positions(
        ma_de_rows, ma_de_y_positions, reference_positions, "MaDe", tolerance=90)
    
    # Create result with aligned/extrapolated rows
    result = detection_results.copy()
    result["sobao_danh_rows_aligned"] = aligned_sobao
    result["ma_de_rows_aligned"] = aligned_ma_de
    result["reference_positions"] = reference_positions
    result["sobao_y_positions"] = sobao_y_positions
    result["ma_de_y_positions"] = ma_de_y_positions
    
    # Count actual and missing rows
    sobao_detected = sum(1 for r in aligned_sobao if r is not None)
    ma_de_detected = sum(1 for r in aligned_ma_de if r is not None)
    sobao_missing = target_rows - sobao_detected
    ma_de_missing = target_rows - ma_de_detected
    
    result["sobao_missing_count"] = sobao_missing
    result["ma_de_missing_count"] = ma_de_missing
    result["sobao_detected_count"] = sobao_detected
    result["ma_de_detected_count"] = ma_de_detected
    
    return result


def _normalize_image_stem(image_arg: Optional[str]) -> str:
    """Normalize user-provided image argument to stem format: PhieuQG.XXXX."""
    if not image_arg:
        return "PhieuQG.0015"

    raw = image_arg.strip()
    if not raw:
        return "PhieuQG.0015"

    token = Path(raw).name
    token_lower = token.lower()
    if token_lower.endswith((".jpg", ".jpeg", ".png")):
        token = Path(token).stem

    if token.startswith("PhieuQG."):
        suffix = token.split(".", 1)[1]
    elif token.lower().startswith("phieuqg."):
        suffix = token.split(".", 1)[1]
    else:
        suffix = token

    if suffix.isdigit():
        suffix = suffix.zfill(4)

    return f"PhieuQG.{suffix}"


def _demo(image_arg: Optional[str] = None) -> None:
    base_image_name = _normalize_image_stem(image_arg)
    candidate_paths = [
        Path("PhieuQG") / f"{base_image_name}.jpg",
        Path("PhieuQG") / f"{base_image_name}.jpeg",
        Path("PhieuQG") / f"{base_image_name}.png",
    ]
    image_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    out_dir = Path("output/detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        tried = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Cannot read image. Tried: {tried}")

    def _run_detection_pipeline(src_img: np.ndarray, prefix: Optional[str]) -> Tuple[Dict[str, object], Dict[str, object]]:
        data_local = detect_boxes_from_morph_lines(
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
        parts_local = group_boxes_into_parts(data_local["boxes"], row_tolerance=30)
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

        # Favor Part III located in lower page region.
        if p3 and page_h > 0:
            y_vals = [cv2.boundingRect(b)[1] for b in parts_local["part_iii"]]
            p3_ratio = float(np.mean(y_vals)) / float(page_h)
            score += max(0.0, min(1.0, (p3_ratio - 0.55) / 0.20))
        return score

    debug_prefix = str(out_dir / image_path.stem)
    data, parts = _run_detection_pipeline(img, debug_prefix)

    preprocess_mode = "base"
    page_h = img.shape[0] if img is not None else 0

    # Fallback for difficult scans: enhance contrast and re-detect.
    if len(parts["part_ii"]) < 8 or len(parts["part_iii"]) < 6:
        img_clahe = _preprocess_clahe(img)
        data_clahe, parts_clahe = _run_detection_pipeline(img_clahe, None)

        base_score = _parts_score(parts, page_h)
        clahe_score = _parts_score(parts_clahe, page_h)
        if clahe_score > base_score + 0.05:
            data = data_clahe
            parts = parts_clahe
            preprocess_mode = "clahe"

    print(f"Detected boxes: {len(data['boxes'])}")
    print(f"Preprocess mode: {preprocess_mode}")
    
    # Group boxes into parts
    print(f"Part I boxes: {len(parts['part_i'])}")
    print(f"Part II boxes: {len(parts['part_ii'])}")
    print(f"Part III boxes: {len(parts['part_iii'])}")
    
    # Detect SoBaoDanh boxes (remaining boxes after parts)
    # Track indices of boxes that are part of the parts
    part_box_set = set(id(box) for box in parts["all_parts"])
    remaining_boxes = [box for box in data["boxes"] if id(box) not in part_box_set]

    # Some scans merge adjacent bubbles into one wide box (e.g. 2 bubbles fused).
    # Split these before separating/grouping upper ID region.
    remaining_for_upper = _split_merged_boxes_for_grouping(
        remaining_boxes,
        split_wide=True,
        split_tall=False,
    )

    # Separate upper ID region into SoBaoDanh (left) and MaDe (right) by X split.
    sbd_candidates, ma_de_candidates, split_x = _separate_upper_id_boxes(
        remaining_for_upper,
        parts["part_i"],
    )

    sobao_danh = detect_sobao_danh_boxes(
        sbd_candidates,
        boxes_per_row=6,
        max_rows=10,
        row_tolerance=30,
        size_tolerance_ratio=0.35,
        debug=False,
    )
    print(f"SoBaoDanh rows: {sobao_danh['row_count']}")
    print(f"SoBaoDanh boxes: {len(sobao_danh['sobao_danh'])}")
    
    # Detect MaDe boxes (remaining boxes after parts and SoBaoDanh)
    # Track indices of boxes that are part of parts and SoBaoDanh
    # Some scans merge two consecutive MaDe rows vertically into a tall box.
    # Split these before grouping MaDe rows.
    remaining_for_ma_de = _split_merged_boxes_for_grouping(
        ma_de_candidates,
        split_wide=False,
        split_tall=True,
    )
    ma_de = detect_ma_de_boxes(
        remaining_for_ma_de,
        boxes_per_row=3,
        max_rows=10,
        row_tolerance=20,
        size_tolerance_ratio=0.3,
        debug=False,
    )

    # Complete MaDe to 10 rows using SoBaoDanh Y references when some rows are missing.
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
            # Build a stable geometry template for 3 MaDe columns.
            col_x = [int(round(float(np.median([rects[c][0] for rects in ma_de_rect_rows])))) for c in range(3)]
            col_w = [int(round(float(np.median([rects[c][2] for rects in ma_de_rect_rows])))) for c in range(3)]
            row_h = int(round(float(np.median([rects[0][3] for rects in ma_de_rect_rows]))))
            row_h = max(1, row_h)

            detected_rows_y = [int(round(float(np.mean([r[1] for r in rects])))) for rects in ma_de_rect_rows]
            align_tolerance = max(35, int(round(row_h * 1.2)))
            aligned_rows: List[Optional[List[np.ndarray]]] = [None] * 10

            used_ref_indices = set()
            for rects, row_y in sorted(zip(ma_de_rect_rows, detected_rows_y), key=lambda t: t[1]):
                candidates = sorted(
                    range(10),
                    key=lambda idx: abs(ref_positions[idx] - row_y),
                )
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

            # Fill missing rows with synthetic boxes at the reference Y.
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

    print(f"MaDe rows: {ma_de['row_count']}")
    print(f"MaDe boxes: {len(ma_de['ma_de'])}")
    print(f"Upper split X: {split_x:.1f}")
    
    # Extrapolate missing rows to reach 10 rows for both sections
    combined_results = {
        "sobao_danh_rows": sobao_danh["sobao_danh_rows"],
        "ma_de_rows": ma_de["ma_de_rows"]
    }
    extrapolated = extrapolate_missing_rows(combined_results, target_rows=10, debug=False)
    
    print(f"\nExtrapolation Summary:")
    print(f"  SoBaoDanh: {extrapolated['sobao_detected_count']}/10 detected, {extrapolated['sobao_missing_count']} missing")
    print(f"  MaDe: {extrapolated['ma_de_detected_count']}/10 detected, {extrapolated['ma_de_missing_count']} missing")
    
    # Show missing row positions
    aligned_sobao = extrapolated.get("sobao_danh_rows_aligned", [])
    aligned_ma_de = extrapolated.get("ma_de_rows_aligned", [])
    reference_positions = extrapolated.get("reference_positions", [])
    
    if aligned_ma_de and None in aligned_ma_de:
        print(f"\n  Missing MaDe rows at positions:")
        for idx, row in enumerate(aligned_ma_de):
            if row is None:
                y_pos = reference_positions[idx] if idx < len(reference_positions) else "?"
                print(f"    Row {idx + 1}: Y ≈ {y_pos}")
    
    # Draw parts on overlay with different colors and labels
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1.5, 2
    
    # Part configurations: (part_key, label, color)
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
            cv2.putText(overlay, label, (50, min_y - 20), font, font_scale, color, font_thickness)
    
    # Draw SoBaoDanh boxes with row labels
    sobao_color = (255, 128, 0)  # Orange color for SoBaoDanh
    for row_idx, row in enumerate(sobao_danh["sobao_danh_rows"]):
        for poly in row:
            cv2.polylines(overlay, [poly], True, sobao_color, 2)
    
    # Draw MaDe boxes with row labels
    ma_de_color = (255, 255, 0)  # Cyan color for MaDe
    for row_idx, row in enumerate(ma_de["ma_de_rows"]):
        for poly in row:
            cv2.polylines(overlay, [poly], True, ma_de_color, 2)
    
    # Do not draw extrapolated placeholders in _parts image;
    # keep this visualization strictly for detected boxes.
    
    cv2.imwrite(f"{debug_prefix}_parts.jpg", overlay)
    print(f"Parts visualization saved to {debug_prefix}_parts.jpg")
    
    # Draw all parts grids on a single image
    print(f"\n=== Drawing grids on all parts ===")
    combined_grid_image = img.copy()
    
    # Draw 4x10 grid on Part I boxes
    if parts["part_i"]:
        print(f"\n=== Part I: 4x10 grid (20% x, 10% y) ===")
        grid_result = extract_grid_from_boxes(
            combined_grid_image,
            boxes=parts["part_i"],
            grid_cols=4,
            grid_rows=10,
            start_offset_ratio_x=0.2,
            start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.015,
            end_offset_ratio_y=0.015,
            grid_color=(0, 255, 0),  # Green
            grid_thickness=1,
        )
        combined_grid_image = grid_result["image_with_grid"]
        
        print(f"Grid drawn on {len(grid_result['grid_info'])} boxes")
        for info in grid_result["grid_info"]:
            print(f"  Box {info['box_idx']}: region {info['region_size']}, "
                  f"cell_size ~{info['cell_size'][0]:.1f}x{info['cell_size'][1]:.1f}")
    
    # Draw 2x4 grid on Part II boxes
    if parts["part_ii"]:
        print(f"\n=== Part II: 2x4 grid (alternating offsets, 30% y, -5% bottom) ===")
        # Part II has 8 boxes, alternating offset pattern:
        # Box 0, 2, 4, 6 (even index): 20% x, 30% y
        # Box 1, 3, 5, 7 (odd index): 0% x, 30% y
        # All boxes: -5% from bottom
        offset_ratios = []
        end_offset_x = []
        end_offset_y = []
        for box_idx in range(len(parts["part_ii"])):
            if box_idx % 2 == 0:  # Even index
                offset_ratios.append((0.3, 0.33))
            else:  # Odd index
                offset_ratios.append((0.0, 0.33))
            end_offset_x.append(0.0)  # 0% from right for all boxes
            end_offset_y.append(0.03)  # 3% from bottom for all boxes
        
        grid_result_ii = extract_grid_from_boxes_variable_offsets(
            combined_grid_image,
            boxes=parts["part_ii"],
            grid_cols=2,
            grid_rows=4,
            start_offset_ratios=offset_ratios,
            end_offset_ratios_x=end_offset_x,
            end_offset_ratios_y=end_offset_y,
            grid_color=(0, 165, 255),  # Orange
            grid_thickness=1,
        )
        combined_grid_image = grid_result_ii["image_with_grid"]
        
        print(f"Grid drawn on {len(grid_result_ii['grid_info'])} boxes")
        for info in grid_result_ii["grid_info"]:
            print(f"  Box {info['box_idx']}: offset {info['offset_ratios']}, end_y -{info['end_offset_y']*100:.0f}%, "
                  f"region {info['region_size']}, cell_size ~{info['cell_size'][0]:.1f}x{info['cell_size'][1]:.1f}")
    
    # Draw 3x2 grid on Part III boxes
    if parts["part_iii"]:
        print(f"\n=== Part III: 4x12 grid with custom pattern (20% x, 10% y) ===")
        # Custom pattern for Part III:
        # Row 0: only column 0 (first cell)
        # Row 1: columns 1, 2 (middle two cells)
        # Rows 2-11: all columns 0-3 (normal)
        custom_pattern = [
            [0],                # Row 0: only 1st cell
            [1, 2],             # Row 1: 2nd and 3rd cells (middle)
        ]
        # Add rows 2-11 with all 4 columns
        for _ in range(10):
            custom_pattern.append([0, 1, 2, 3])
        
        grid_result_iii = extract_grid_from_boxes_custom_pattern(
            combined_grid_image,
            boxes=parts["part_iii"],
            grid_cols=4,
            grid_rows=12,
            start_offset_ratio_x=0.22,
            start_offset_ratio_y=0.16,
            end_offset_ratio_x=0.1,
            end_offset_ratio_y=0.015,
            grid_color=(255, 0, 0),  # Blue (BGR format, so red in display)
            grid_thickness=1,
            row_col_patterns=custom_pattern,
        )
        combined_grid_image = grid_result_iii["image_with_grid"]
        
        print(f"Grid drawn on {len(grid_result_iii['grid_info'])} boxes")
        for info in grid_result_iii["grid_info"]:
            print(f"  Box {info['box_idx']}: region {info['region_size']}, "
                  f"cell_size ~{info['cell_size'][0]:.1f}x{info['cell_size'][1]:.1f}, "
                  f"pattern={info['pattern'][:2]}... (12 rows total)")

    # Draw SoBaoDanh and MaDe box grids on the same combined grid image.
    # These sections are already grid-like bubble matrices, so draw their box contours.
    sobao_grid_color = (255, 128, 0)  # Orange
    ma_de_grid_color = (255, 255, 0)  # Cyan

    if sobao_danh["sobao_danh_rows"]:
        print(f"\n=== SoBaoDanh: drawing detected box grid ===")
        for row in sobao_danh["sobao_danh_rows"]:
            for poly in row:
                cv2.polylines(combined_grid_image, [poly], True, sobao_grid_color, 1)
        print(f"Grid drawn on {len(sobao_danh['sobao_danh'])} SoBaoDanh boxes")

    if ma_de["ma_de_rows"]:
        print(f"\n=== MaDe: drawing detected box grid ===")
        for row in ma_de["ma_de_rows"]:
            for poly in row:
                cv2.polylines(combined_grid_image, [poly], True, ma_de_grid_color, 1)
        print(f"Grid drawn on {len(ma_de['ma_de'])} MaDe boxes")
    
    # Save combined image with all grids
    combined_grid_path = f"{debug_prefix}_all_parts_with_grid.jpg"
    cv2.imwrite(combined_grid_path, combined_grid_image)
    print(f"\n✓ Combined grid image saved to: {combined_grid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and draw answer grids from form images.")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image identifier, e.g. 0015, 31, PhieuQG.0031, or PhieuQG/PhieuQG.0031.jpg",
    )
    parser.add_argument(
        "--image",
        dest="image_opt",
        default=None,
        help="Image identifier (same accepted formats as positional image).",
    )
    args = parser.parse_args()
    _demo(image_arg=args.image_opt or args.image)
