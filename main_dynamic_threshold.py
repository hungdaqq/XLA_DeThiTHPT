"""
main_dynamic_threshold.py – Grid analysis with dynamic per-box thresholds
Instead of detecting circles, use adaptive thresholding for each individual box
"""

import cv2
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

TARGET_W, TARGET_H = 900, 1270

# ============================================================
# 1. PERSPECTIVE TRANSFORM
# ============================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    d    = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def find_corner_markers(image: np.ndarray,
                        debug_out: str = None) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    corners = None

    # Strategy A: adaptive threshold + solid contour
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (40 < area < W * H * 0.006):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if not (0.15 < bw / max(bh, 1) < 6.5):
            continue
        roi  = binary[y:y+bh, x:x+bw]
        fill = np.sum(roi > 0) / max(bw * bh, 1)
        if fill > 0.55:
            cands.append((x + bw / 2, y + bh / 2, area))
    
    # Pick 4 corners
    if len(cands) >= 4:
        pts = np.array([(c[0], c[1]) for c in cands], dtype="float32")
        s    = pts.sum(axis=1)
        diff = pts[:, 0] - pts[:, 1]
        tl, br = pts[np.argmin(s)],    pts[np.argmax(s)]
        tr, bl = pts[np.argmax(diff)], pts[np.argmin(diff)]
        corners = np.array([tl, tr, br, bl])
        ok = (tl[0] < W*.55 and tl[1] < H*.55 and
              tr[0] > W*.45 and tr[1] < H*.55 and
              br[0] > W*.45 and br[1] > H*.45 and
              bl[0] < W*.55 and bl[1] > H*.45)
        if not ok:
            corners = None

    # Fallback
    if corners is None:
        pad = 6
        corners = np.array([[pad,pad],[W-pad,pad],[W-pad,H-pad],[pad,H-pad]], dtype="float32")

    corners = order_points(corners)
    if debug_out:
        dbg = image.copy()
        for pt, lbl, col in zip(corners,["TL","TR","BR","BL"],
                                 [(0,255,0),(0,140,255),(0,0,255),(255,140,0)]):
            p = tuple(pt.astype(int))
            cv2.circle(dbg, p, 18, col, -1)
            cv2.putText(dbg, lbl, (p[0]+6, p[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.polylines(dbg, [corners.astype(int).reshape(-1,1,2)], True, (0,255,0), 3)
        cv2.imwrite(debug_out, dbg)
    return corners


def warp(image: np.ndarray, corners: np.ndarray,
         W: int = TARGET_W, H: int = TARGET_H) -> np.ndarray:
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (W, H))


# ============================================================
# 1B. GRID LINE DETECTION VIA MORPHOLOGY
# ============================================================

def detect_grid_points(image: np.ndarray,
                       vertical_scale: float = 0.015,
                       horizontal_scale: float = 0.015,
                       min_point_area: int = 8,
                       block_size: int = 35,
                       block_offset: int = 7,
                       debug_prefix: Optional[str] = None) -> Dict[str, object]:
    """Detect grid intersections via threshold -> morphology pipeline."""
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

    H, W = gray.shape
    v_len = max(3, int(H * vertical_scale))
    h_len = max(3, int(W * horizontal_scale))
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


# ============================================================
# 2. LAYOUT CONFIG
# ============================================================

SECTIONS = {
    "sobaodanh": {
        "y_range":     (0.075, 0.283),
        "groups": [
            {"x_range": (0.740, 0.860), "label": "SoBaoDanh"},
        ],
        "rows":         ["0","1","2","3","4","5","6","7","8","9"],
        "n_digit_cols": 6,
        "darkness_thresh":  85.0,
        "otsu_threshold_upper": 175.0,
    },
    "made": {
        "y_range":     (0.075, 0.283),
        "groups": [
            {"x_range": (0.900, 0.960), "label": "MaDe"},
        ],
        "rows":         ["0","1","2","3","4","5","6","7","8","9"],
        "n_digit_cols": 3,
        "darkness_thresh":  85.0,
        "otsu_threshold_upper": 175.0,
    },
    "phan1": {
        "y_range":     (0.345, 0.525),
        "groups": [
            {"x_range": (0.065, 0.230), "q_start":  1},
            {"x_range": (0.301, 0.470), "q_start": 11},
            {"x_range": (0.538, 0.710), "q_start": 21},
            {"x_range": (0.780, 0.950), "q_start": 31},
        ],
        "choices":     ["A","B","C","D"],
        "n_rows":      10,
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.10,
    },
    "phan2": {
        "y_range":     (0.598, 0.668),
        "groups": [
            {"x_range": (0.065, 0.230), "cau": (1,2)},
            {"x_range": (0.301, 0.470), "cau": (3,4)},
            {"x_range": (0.538, 0.710), "cau": (5,6)},
            {"x_range": (0.780, 0.950), "cau": (7,8)},
        ],
        "rows":        ["a","b","c","d"],
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.10,
    },
    "phan3": {
        "y_range":      (0.753, 0.970),
        "groups": [
            {"x_range": (0.066, 0.168), "cau": 1},
            {"x_range": (0.220, 0.320), "cau": 2},
            {"x_range": (0.372, 0.472), "cau": 3},
            {"x_range": (0.520, 0.620), "cau": 4},
            {"x_range": (0.675, 0.780), "cau": 5},
            {"x_range": (0.825, 0.930), "cau": 6},
        ],
        "rows":         ["-",",","0","1","2","3","4","5","6","7","8","9"],
        "n_digit_cols": 4,
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.15,
    },
}


# ============================================================
# 3. DYNAMIC THRESHOLD ANALYSIS PER BOX
# ============================================================

def analyze_digit_section(gray: np.ndarray, cfg: Dict, H: int, W: int, section_name: str) -> List[Dict]:
    """Analyze a digit section (SO BAO DANH or MA DE) using darkness metrics.
    
    Args:
        gray: grayscale image
        cfg: section configuration dict
        H, W: image height and width
        section_name: name of section for display
    
    Returns:
        list of grid items with filled status
    """
    grid_items = []
    y1 = int(cfg["y_range"][0] * H)
    y2 = int(cfg["y_range"][1] * H)
    n_rows = len(cfg["rows"])
    n_cols = cfg["n_digit_cols"]
    darkness_thresh = cfg["darkness_thresh"]
    
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_pct = grp["x_range"][0]
        x2_pct = grp["x_range"][1]
        
        # Generate x and y percentages for grid cells
        x_pcts = [x1_pct + (x2_pct - x1_pct) * (col + 0.5) / n_cols for col in range(n_cols)]
        y_pcts = [y1/H + (y2 - y1) / H * (row + 0.5) / n_rows for row in range(n_rows)]
        
        # Parse digit grid using cell_darkness
        filled_str = parse_digit_grid(gray, x_pcts, y_pcts, r=7)
        
        # Show individual cells for debugging
        for col in range(n_cols):
            x_col_pct = x_pcts[col]
            cx = int(x_col_pct * W)
            for row in range(n_rows):
                y_row_pct = y_pcts[row]
                cy = int(y_row_pct * H)
                darkness = cell_darkness(gray, cx, cy, r=7)
                row_label = cfg["rows"][row]
                filled_status = "FILLED" if row == int(filled_str[col]) else "EMPTY"
                print(f"    Digit{row_label}-Col{col}: {filled_status} (darkness: {darkness:.1f})")
                
                grid_items.append({
                    "digit": row_label, "col": col,
                    "box_bounds": (0, 0, 0, 0),
                    "filled": row == int(filled_str[col]),
                    "mean_darkness": darkness,
                    "otsu_threshold": 128.0,
                })
        
        print(f"  Group {grp_idx} - Filled digits: {filled_str if filled_str else '(empty)'}")
    
    return grid_items

def analyze_box_with_binary_image(binary_image: np.ndarray,
                                   x1: int, y1: int, x2: int, y2: int,
                                   fill_ratio_thresh: float = 0.15) -> dict:
    """Analyze a single grid box using binary morphological image.
    
    Strategy:
    - Count white pixels (binary > 0) in the box region
    - Calculate fill ratio as white_pixels / total_pixels
    - Higher fill  ratio = more ink = FILLED
    - If fill_ratio > fill_ratio_thresh: FILLED, else EMPTY
    
    Args:
        binary_image: binary morphological image from grid detection
        x1, y1, x2, y2: box coordinates
        fill_ratio_thresh: threshold for fill ratio (> this = filled)
    
    Returns:
        dict with fill_ratio, is_filled, and box information
    """
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(binary_image.shape[1], x2)
    y2 = min(binary_image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return {
            "fill_ratio": 0.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Extract ROI from binary image
    roi = binary_image[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "fill_ratio": 0.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Calculate fill ratio: white pixels (>0) in box
    white_pixels = np.sum(roi > 0)
    total_pixels = roi.size
    fill_ratio = white_pixels / max(total_pixels, 1)
    
    # Classify: higher fill  ratio = more ink = filled
    # Threshold: if fill_ratio > fill_ratio_thresh, box is filled
    is_filled = fill_ratio > fill_ratio_thresh
    
    return {
        "fill_ratio": fill_ratio,
        "is_filled": is_filled,
        "box_bounds": (x1, y1, x2, y2),
    }

def calculate_box_otsu_threshold(roi: np.ndarray) -> float:
    """
    Calculate Otsu threshold for a given ROI (box).
    Returns the threshold value which can be used to determine fill ratio.
    
    Args:
        roi: grayscale image region of interest
    
    Returns:
        otsu_threshold: the optimal threshold value (0-255)
    """
    if roi.size == 0:
        return 128.0
    
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    return float(otsu_thresh)


def analyze_box_with_dynamic_threshold(warped: np.ndarray, 
                                       x1: int, y1: int, x2: int, y2: int, 
                                       otsu_threshold_upper: float = 175.0) -> dict:
    """
    Analyze a single grid box using Otsu threshold.
    
    Strategy:
    - Otsu threshold calculates optimal threshold value
    - Lower Otsu threshold = darker box content = more ink = FILLED
    - Higher Otsu threshold = lighter box content = empty
    - If otsu_threshold <= otsu_threshold_upper: FILLED
    
    Args:
        warped: warped image
        x1, y1, x2, y2: box coordinates
        otsu_threshold_upper: threshold for Otsu value (≤ this = filled)
    
    Returns:
        dict with otsu_threshold, is_filled, and box information
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray.shape[1], x2)
    y2 = min(gray.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return {
            "otsu_threshold": 128.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Extract ROI
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "otsu_threshold": 128.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Calculate dynamic Otsu threshold for this specific box
    # Lower Otsu threshold = darker pixels = filled
    otsu_thresh = calculate_box_otsu_threshold(roi)
    
    # Determine if filled: lower otsu threshold means darker = filled
    is_filled = otsu_thresh <= otsu_threshold_upper
    
    return {
        "otsu_threshold": otsu_thresh,
        "is_filled": is_filled,
        "box_bounds": (x1, y1, x2, y2),
    }


def cell_darkness(gray: np.ndarray, cx: int, cy: int, r: int = 7) -> float:
    """Calculate mean darkness in a circular cell region
    
    Args:
        gray: grayscale image
        cx, cy: center coordinates
        r: radius of circle
    
    Returns:
        mean pixel value (lower = darker = filled)
    """
    x1, y1 = max(0, cx-r), max(0, cy-r)
    x2, y2 = min(gray.shape[1], cx+r), min(gray.shape[0], cy+r)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 255.0
    
    # Create circular mask
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center = (roi.shape[1]//2, roi.shape[0]//2)
    radius = min(roi.shape[0], roi.shape[1]) // 2 - 1
    cv2.circle(mask, center, max(1, radius), 255, -1)
    
    # Calculate mean of pixels within circle
    pixels = roi[mask > 0]
    return float(np.mean(pixels)) if len(pixels) > 0 else 255.0


def parse_digit_grid(gray: np.ndarray,
                     x_pcts: List[float],
                     y_pcts: List[float],
                     r: int = 7) -> str:
    """Parse a row of digit cells (SBD or MaDe) using mean darkness
    
    Strategy:
    - Calculate darkness (mean pixel value) for each cell
    - For each column, find the digit with LOWEST darkness (darkest = filled)
    
    Args:
        gray: grayscale image
        x_pcts: x positions as percentages of width for each column
        y_pcts: y positions as percentages of height for each row
        r: radius of circular cell
    
    Returns:
        string of digits, e.g. '001013' or '004'
    """
    H, W = gray.shape
    digits_str = []
    
    for xp in x_pcts:
        cx = int(xp * W)
        # Calculate darkness for each digit in this column
        col_vals = [(cell_darkness(gray, cx, int(yp*H), r), di)
                    for di, yp in enumerate(y_pcts)]
        
        # Find digit with lowest darkness (darkest/filled)
        best_dark, best_digit = min(col_vals, key=lambda v: v[0])
        
        # Add best digit to result
        digits_str.append(str(best_digit))
    
    return "".join(digits_str)


def render_digit_grid(out: np.ndarray, gray: np.ndarray,
                      x_pcts: List[float], y_pcts: List[float],
                      label: str, r: int = 7) -> None:
    """Render digit grid visualization with filled cells in green with opacity
    
    Args:
        out: output image to draw on
        gray: grayscale image for darkness calculation
        x_pcts: x positions as percentages
        y_pcts: y positions as percentages
        label: label for the section
        r: radius of circles
    """
    H, W = gray.shape
    color_filled = (0, 255, 0)  # Green for filled
    overlay = out.copy()
    
    for col_i, xp in enumerate(x_pcts):
        cx = int(xp * W)
        # Find which digit is filled in this column
        col_vals = [(cell_darkness(gray, cx, int(yp*H), r), di)
                    for di, yp in enumerate(y_pcts)]
        best_dark, best_digit = min(col_vals, key=lambda v: v[0])
        
        # Draw filled circle in green with transparency on overlay
        for di, yp in enumerate(y_pcts):
            cy = int(yp * H)
            is_filled = (di == best_digit)
            
            if is_filled:
                # Draw filled circle in green on overlay (will blend with transparency)
                cv2.circle(overlay, (cx, cy), r+2, color_filled, -1)
                cv2.circle(overlay, (cx, cy), r+2, (0, 120, 20), 2)
    
    # Blend overlay with original image for transparency effect
    alpha = 0.5  # 50% opacity
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)


# ============================================================
# 4. GRID ANALYSIS WITH DYNAMIC THRESHOLDS
# ============================================================

def analyze_grid(warped: np.ndarray, grid_debug: Dict = None) -> Dict:
    """Phân tích grid sử dụng binary image cho phan1/2/3 và darkness cho digit sections"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    H, W = warped.shape[:2]
    results = {}
    
    # Extract binary horizontal lines image for phan1/2/3 analysis
    binary_image = None
    if grid_debug and "horizontal" in grid_debug:
        binary_image = grid_debug["horizontal"]
        print("\n[METHOD] Using BINARY MORPHOLOGICAL IMAGE (Horizontal Lines) for PHAN 1, 2, 3")

    # SO BAO DANH
    print("\n=== SO BAO DANH ===")
    results["sobaodanh"] = analyze_digit_section(gray, SECTIONS["sobaodanh"], H, W, "SoBaoDanh")

    # MA DE
    print("\n=== MA DE ===")
    results["made"] = analyze_digit_section(gray, SECTIONS["made"], H, W, "MaDe")

    # PHAN 1
    print("\n=== PHAN I ===")
    phan1_grid = []
    cfg = SECTIONS["phan1"]
    y1 = int(cfg["y_range"][0] * H)
    y2 = int(cfg["y_range"][1] * H)
    n_rows = cfg["n_rows"]
    n_cols = len(cfg["choices"])
    fill_ratio_thresh = cfg.get("fill_ratio_thresh", 0.15)
    
    for group_idx, grp in enumerate(cfg["groups"]):
        x1 = int(grp["x_range"][0] * W)
        x2 = int(grp["x_range"][1] * W)
        q_start = grp["q_start"]
        
        for row in range(n_rows):
            row_y1 = int(y1 + row * (y2 - y1) / n_rows)
            row_y2 = int(y1 + (row + 1) * (y2 - y1) / n_rows)
            
            for col in range(n_cols):
                col_x1 = int(x1 + col * (x2 - x1) / n_cols)
                col_x2 = int(x1 + (col + 1) * (x2 - x1) / n_cols)
                
                # Use binary image for classification if available
                if binary_image is not None:
                    box_data = analyze_box_with_binary_image(binary_image, col_x1, row_y1, col_x2, row_y2, fill_ratio_thresh)
                    filled = box_data["is_filled"]
                    metric = f"fillratio={box_data['fill_ratio']:.2f}"
                else:
                    # Fallback to Otsu threshold if binary image not available
                    otsu_thresh_upper = cfg.get("otsu_threshold_upper", 175.0)
                    box_data = analyze_box_with_dynamic_threshold(warped, col_x1, row_y1, col_x2, row_y2, otsu_thresh_upper)
                    filled = box_data["is_filled"]
                    metric = f"[OTSU] otsu={box_data['otsu_threshold']:.1f}"
                
                q_num = q_start + row
                choice = cfg["choices"][col]
                phan1_grid.append({
                    "q": q_num, "choice": choice,
                    "box_bounds": box_data["box_bounds"],
                    "filled": filled, 
                    "fill_ratio": box_data.get("fill_ratio", 0.0),
                })
                
                status = "✓" if filled else "·"
                print(f"  Q{q_num:02d}-{choice}: {status} ({metric})")
    
    results["phan1"] = phan1_grid

    # PHAN 2
    print("\n=== PHAN II ===")
    phan2_grid = []
    cfg = SECTIONS["phan2"]
    y1 = int(cfg["y_range"][0] * H)
    y2 = int(cfg["y_range"][1] * H)
    n_rows = len(cfg["rows"])
    n_cols = 4
    fill_ratio_thresh = cfg.get("fill_ratio_thresh", 0.15)
    
    for group_idx, grp in enumerate(cfg["groups"]):
        x1 = int(grp["x_range"][0] * W)
        x2 = int(grp["x_range"][1] * W)
        cau_odd, cau_even = grp["cau"]
        
        for row in range(n_rows):
            row_y1 = int(y1 + row * (y2 - y1) / n_rows)
            row_y2 = int(y1 + (row + 1) * (y2 - y1) / n_rows)
            
            # Câu lẻ (2 columns)
            for col in range(2):
                col_x1 = int(x1 + col * (x2 - x1) / 4)
                col_x2 = int(x1 + (col + 1) * (x2 - x1) / 4)
                
                # Use binary image for classification if available
                if binary_image is not None:
                    box_data = analyze_box_with_binary_image(binary_image, col_x1, row_y1, col_x2, row_y2, fill_ratio_thresh)
                    metric = f"fillratio={box_data['fill_ratio']:.2f}"
                else:
                    otsu_thresh_upper = cfg.get("otsu_threshold_upper", 175.0)
                    box_data = analyze_box_with_dynamic_threshold(warped, col_x1, row_y1, col_x2, row_y2, otsu_thresh_upper)
                    metric = f"[OTSU] otsu={box_data.get('otsu_threshold', 0):.1f}"
                
                filled = box_data["is_filled"]
                col_label = "Dung" if col == 0 else "Sai"
                filled_status = "✓" if filled else "·"
                phan2_grid.append({
                    "cau": cau_odd, "row": cfg["rows"][row], "col": col_label,
                    "box_bounds": box_data["box_bounds"],
                    "filled": filled, 
                    "fill_ratio": box_data.get("fill_ratio", 0.0),
                })
                print(f"  Cau{cau_odd}-{cfg['rows'][row]}-{col_label}: {filled_status} ({metric})")
            
            # Câu chẵn (2 columns)
            for col in range(2):
                col_x1 = int(x1 + (col + 2) * (x2 - x1) / 4)
                col_x2 = int(x1 + (col + 3) * (x2 - x1) / 4)
                
                # Use binary image for classification if available
                if binary_image is not None:
                    box_data = analyze_box_with_binary_image(binary_image, col_x1, row_y1, col_x2, row_y2, fill_ratio_thresh)
                    metric = f"fillratio={box_data['fill_ratio']:.2f}"
                else:
                    otsu_thresh_upper = cfg.get("otsu_threshold_upper", 175.0)
                    box_data = analyze_box_with_dynamic_threshold(warped, col_x1, row_y1, col_x2, row_y2, otsu_thresh_upper)
                    metric = f"[OTSU] otsu={box_data.get('otsu_threshold', 0):.1f}"
                
                filled = box_data["is_filled"]
                col_label = "Dung" if col == 0 else "Sai"
                filled_status = "✓" if filled else "·"
                phan2_grid.append({
                    "cau": cau_even, "row": cfg["rows"][row], "col": col_label,
                    "box_bounds": box_data["box_bounds"],
                    "filled": filled, 
                    "fill_ratio": box_data.get("fill_ratio", 0.0),
                })
                print(f"  Cau{cau_even}-{cfg['rows'][row]}-{col_label}: {filled_status} ({metric})")
    
    results["phan2"] = phan2_grid

    # PHAN 3
    print("\n=== PHAN III ===")
    phan3_grid = []
    cfg = SECTIONS["phan3"]
    y1 = int(cfg["y_range"][0] * H)
    y2 = int(cfg["y_range"][1] * H)
    n_rows = len(cfg["rows"])
    n_cols = cfg["n_digit_cols"]
    fill_ratio_thresh = cfg.get("fill_ratio_thresh", 0.15)
    
    for group_idx, grp in enumerate(cfg["groups"]):
        x1 = int(grp["x_range"][0] * W)
        x2 = int(grp["x_range"][1] * W)
        cau_num = grp["cau"]
        
        for row in range(n_rows):
            row_y1 = int(y1 + row * (y2 - y1) / n_rows)
            row_y2 = int(y1 + (row + 1) * (y2 - y1) / n_rows)
            row_label = cfg["rows"][row]
            
            # Determine which columns to process based on row
            if row == 0:  # Hàng "-": chỉ col 0
                col_range = [0]
            elif row == 1:  # Hàng ",": chỉ col 1, 2
                col_range = [1, 2]
            else:  # Hàng khác: tất cả col 0-3
                col_range = range(n_cols)
            
            for col in col_range:
                col_x1 = int(x1 + col * (x2 - x1) / n_cols)
                col_x2 = int(x1 + (col + 1) * (x2 - x1) / n_cols)
                
                # Use binary image for classification if available
                if binary_image is not None:
                    box_data = analyze_box_with_binary_image(binary_image, col_x1, row_y1, col_x2, row_y2, fill_ratio_thresh)
                    metric = f"fillratio={box_data['fill_ratio']:.2f}"
                else:
                    otsu_thresh_upper = cfg.get("otsu_threshold_upper", 175.0)
                    box_data = analyze_box_with_dynamic_threshold(warped, col_x1, row_y1, col_x2, row_y2, otsu_thresh_upper)
                    metric = f"[OTSU] otsu={box_data.get('otsu_threshold', 0):.1f}"
                
                filled = box_data["is_filled"]
                phan3_grid.append({
                    "cau": cau_num, "row": row_label, "col": col,
                    "box_bounds": box_data["box_bounds"],
                    "filled": filled, 
                    "fill_ratio": box_data.get("fill_ratio", 0.0),
                })
                
                if col == col_range[0]:
                    filled_status = "✓" if filled else "·"
                    print(f"  Cau{cau_num}-{row_label}: {filled_status} ({metric})")
    
    results["phan3"] = phan3_grid
    return results


# ============================================================
# 5. VISUALIZATION
# ============================================================

def visualize(warped: np.ndarray, grid_data: Dict, out_path: str):
    """Vẽ group boxes + grid lines + filled regions với transparency (50%)
    Sử dụng dynamic threshold thay vì circle detection"""
    out = warped.copy()
    H, W = warped.shape[:2]
    overlay = out.copy()
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    GREEN = (0, 255, 0)
    GRID_COLOR = (0, 255, 0)
    BOX_COLOR = (100, 200, 0)
    
    # ── SO BAO DANH ──
    cfg = SECTIONS["sobaodanh"]
    y1_sbd = int(cfg["y_range"][0] * H)
    y2_sbd = int(cfg["y_range"][1] * H)
    n_rows_sbd = len(cfg["rows"])
    n_cols_sbd = cfg["n_digit_cols"]
    
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_grp = int(grp["x_range"][0] * W)
        x2_grp = int(grp["x_range"][1] * W)
        grp_color = (0, 200, 200)  # Cyan
        
        cv2.rectangle(out, (x1_grp - 3, y1_sbd - 3), (x2_grp + 3, y2_sbd + 3), grp_color, 2)
        cv2.putText(out, grp["label"], (x1_grp + 2, y1_sbd - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, grp_color, 1)
        
        # Render digit circles with filled cells in green
        x1_pct = grp["x_range"][0]
        x2_pct = grp["x_range"][1]
        x_pcts = [x1_pct + (x2_pct - x1_pct) * (col + 0.5) / n_cols_sbd for col in range(n_cols_sbd)]
        y_pcts = [y1_sbd/H + (y2_sbd - y1_sbd) / H * (row + 0.5) / n_rows_sbd for row in range(n_rows_sbd)]
        render_digit_grid(out, gray, x_pcts, y_pcts, "SBD", r=7)
    
    # ── MA DE ──
    cfg = SECTIONS["made"]
    y1_md = int(cfg["y_range"][0] * H)
    y2_md = int(cfg["y_range"][1] * H)
    n_rows_md = len(cfg["rows"])
    n_cols_md = cfg["n_digit_cols"]
    
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_grp = int(grp["x_range"][0] * W)
        x2_grp = int(grp["x_range"][1] * W)
        grp_color = (255, 200, 0)  # Blue-yellow
        
        cv2.rectangle(out, (x1_grp - 3, y1_md - 3), (x2_grp + 3, y2_md + 3), grp_color, 2)
        cv2.putText(out, grp["label"], (x1_grp + 2, y1_md - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, grp_color, 1)
        
        # Render digit circles with filled cells in green
        x1_pct = grp["x_range"][0]
        x2_pct = grp["x_range"][1]
        x_pcts = [x1_pct + (x2_pct - x1_pct) * (col + 0.5) / n_cols_md for col in range(n_cols_md)]
        y_pcts = [y1_md/H + (y2_md - y1_md) / H * (row + 0.5) / n_rows_md for row in range(n_rows_md)]
        render_digit_grid(out, gray, x_pcts, y_pcts, "MaDe", r=7)
    
    # ── PHAN 1 ──
    cfg = SECTIONS["phan1"]
    y1_p1 = int(cfg["y_range"][0] * H)
    y2_p1 = int(cfg["y_range"][1] * H)
    n_rows_p1 = cfg["n_rows"]
    n_cols_p1 = len(cfg["choices"])
    
    group_colors = [(100, 200, 0), (50, 150, 255), (255, 100, 100), (150, 100, 255)]
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_grp = int(grp["x_range"][0] * W)
        x2_grp = int(grp["x_range"][1] * W)
        grp_color = group_colors[grp_idx]
        
        # Vẽ group box
        cv2.rectangle(out, (x1_grp - 3, y1_p1 - 3), (x2_grp + 3, y2_p1 + 3), grp_color, 2)
        q_start = grp["q_start"]
        cv2.putText(out, f"Q{q_start}-{q_start+9}", (x1_grp + 2, y1_p1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, grp_color, 1)
        
        # Vẽ grid lines
        for col in range(1, n_cols_p1):
            x_line = int(x1_grp + col * (x2_grp - x1_grp) / n_cols_p1)
            cv2.line(out, (x_line, y1_p1), (x_line, y2_p1), GRID_COLOR, 1)
        for row in range(1, n_rows_p1):
            y_line = int(y1_p1 + row * (y2_p1 - y1_p1) / n_rows_p1)
            cv2.line(out, (x1_grp, y_line), (x2_grp, y_line), GRID_COLOR, 1)
    
    # ── PHAN 2 ──
    cfg = SECTIONS["phan2"]
    y1_p2 = int(cfg["y_range"][0] * H)
    y2_p2 = int(cfg["y_range"][1] * H)
    n_rows_p2 = len(cfg["rows"])
    n_cols_p2 = 4
    
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_grp = int(grp["x_range"][0] * W)
        x2_grp = int(grp["x_range"][1] * W)
        grp_color = group_colors[grp_idx]
        
        cv2.rectangle(out, (x1_grp - 3, y1_p2 - 3), (x2_grp + 3, y2_p2 + 3), grp_color, 2)
        cau_odd, cau_even = grp["cau"]
        cv2.putText(out, f"C{cau_odd},{cau_even}", (x1_grp + 2, y1_p2 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, grp_color, 1)
        
        # Vẽ grid lines
        for col in range(1, n_cols_p2):
            x_line = int(x1_grp + col * (x2_grp - x1_grp) / n_cols_p2)
            cv2.line(out, (x_line, y1_p2), (x_line, y2_p2), GRID_COLOR, 1)
        for row in range(1, n_rows_p2):
            y_line = int(y1_p2 + row * (y2_p2 - y1_p2) / n_rows_p2)
            cv2.line(out, (x1_grp, y_line), (x2_grp, y_line), GRID_COLOR, 1)
    
    # ── PHAN 3 ──
    cfg = SECTIONS["phan3"]
    y1_p3 = int(cfg["y_range"][0] * H)
    y2_p3 = int(cfg["y_range"][1] * H)
    n_rows_p3 = len(cfg["rows"])
    n_cols_p3 = cfg["n_digit_cols"]
    
    group_colors_p3 = [(100, 200, 0), (50, 150, 255), (255, 100, 100), (150, 100, 255), (50, 255, 255), (255, 150, 50)]
    for grp_idx, grp in enumerate(cfg["groups"]):
        x1_grp = int(grp["x_range"][0] * W)
        x2_grp = int(grp["x_range"][1] * W)
        grp_color = group_colors_p3[grp_idx] if grp_idx < len(group_colors_p3) else (100, 100, 100)
        
        cv2.rectangle(out, (x1_grp - 3, y1_p3 - 3), (x2_grp + 3, y2_p3 + 3), grp_color, 2)
        cau_num = grp["cau"]
        cv2.putText(out, f"C{cau_num}", (x1_grp + 2, y1_p3 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, grp_color, 1)
        
        # Vẽ grid lines
        for col in range(1, n_cols_p3):
            x_line = int(x1_grp + col * (x2_grp - x1_grp) / n_cols_p3)
            cv2.line(out, (x_line, y1_p3), (x_line, y2_p3), GRID_COLOR, 1)
        for row in range(1, n_rows_p3):
            y_line = int(y1_p3 + row * (y2_p3 - y1_p3) / n_rows_p3)
            cv2.line(out, (x1_grp, y_line), (x2_grp, y_line), GRID_COLOR, 1)
    
    # Vẽ filled boxes với transparency overlay
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN, -1)
    
    # Blend overlay với transparency (50%)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    
    # Vẽ border rectangles cho các ô filled (xanh đậm)
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
    
    # Vẽ border rectangles cho các ô không được fill (cam nhạt)
    UNFILLED_COLOR = (0, 165, 255)  # Cam
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if not item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), UNFILLED_COLOR, 1)
    
    cv2.imwrite(out_path, out)
    print(f"\n[SAVED] {out_path}")


# ============================================================
# 6. MAIN
# ============================================================

def process(image_path: str, out_dir: str = "./outputs_grid", debug: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  GRID ANALYSIS WITH DYNAMIC THRESHOLDS")
    print(f"  {image_path}")
    print(f"{'='*60}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    
    corners = find_corner_markers(image, debug_out=f"{out_dir}/01_corners.jpg" if debug else None)
    warped = warp(image, corners)
    if debug:
        cv2.imwrite(f"{out_dir}/02_warped.jpg", warped)
        grid_debug = detect_grid_points(warped, debug_prefix=f"{out_dir}/grid")
    else:
        grid_debug = detect_grid_points(warped)

    print(f"[GRID] {len(grid_debug['points'])} intersections detected via morphology")
    
    grid_data = analyze_grid(warped, grid_debug)
    visualize(warped, grid_data, f"{out_dir}/03_grid_filled.jpg")
    
    print(f"\n[OK] Output: {out_dir}/")


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "./PhieuQG/PhieuQG.0045.jpg"
    process(img_path, debug=True)
