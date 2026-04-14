"""
main_dynamic_threshold.py – Phân tích phiếu trắc nghiệm với ngưỡng động

Mục đích: Phát hiện và phân tích tự động ô được tô/chưa tô trên phiếu thi trắc nghiệm
Phương pháp: 
- Dùng binary morphology để phát hiện lưới
- Dùng fill ratio hoặc Otsu threshold để xác định ô được tô

Phần được hỗ trợ:
1. Số báo danh (6 chữ số)
2. Mã đề (3 chữ số)
3. Phần 1: 40 câu trắc nghiệm (4 lựa chọn)
4. Phần 2: 8 câu (trả lời Đúng/Sai)
5. Phần 3: 6 câu (nhập số)

Output: Ảnh với các ô được tô được tô màu xanh (50% opacity)
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
    """Sắp xếp 4 góc của tờ phiếu theo thứ tự: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)  # Tổng tọa độ: góc trên trái có tổng nhỏ nhất
    d    = np.diff(pts, axis=1).ravel()  # Hiệu x-y: góc trên phải có hiệu lớn nhất
    rect[0] = pts[np.argmin(s)]   # Top-left
    rect[2] = pts[np.argmax(s)]   # Bottom-right
    rect[1] = pts[np.argmin(d)]   # Top-right
    rect[3] = pts[np.argmax(d)]   # Bottom-left
    return rect


def find_corner_markers(image: np.ndarray,
                        debug_out: str = None) -> Tuple[np.ndarray, bool]:
    """Tìm 4 góc của tờ phiếu (đánh dấu gốc) bằng phương pháp adaptive threshold.
    
    Chiến lược: Tìm các contour hình tròn/vuông ở 4 góc => sắp xếp lại vị trí
    
    Returns:
        Tuple[corners, is_valid]: corners và flag cho biết có tìm được đủ marker không
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    corners = None
    is_valid = False

    # Chiến lược A: adaptive threshold + kiểm tra contour rắn chắc
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
    
    # Validation: Check if enough marker candidates detected
    if len(cands) < 20:
        print(f"[⚠️  INVALID SHEET] Phát hiện quá ít marker ({len(cands)} < 20) - Đây không phải là một phiếu trả lời hợp lệ!")
        # Return default corners (no marker detection possible)
        pad = 6
        corners = np.array([[pad,pad],[W-pad,pad],[W-pad,H-pad],[pad,H-pad]], dtype="float32")
        return order_points(corners), False
    
    is_valid = True
    
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
        is_valid = False

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
    return corners, is_valid


def warp(image: np.ndarray, corners: np.ndarray,
         W: int = TARGET_W, H: int = TARGET_H) -> np.ndarray:
    """Biến đổi perspective để chỉnh thẳng tờ phiếu.
    
    Chuyển 4 góc bất kỳ thành hình chữ nhật chuẩn có kích thước W x H.
    """
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")  # Hình chữ nhật chuẩn
    M   = cv2.getPerspectiveTransform(corners, dst)  # Ma trận biến đổi
    return cv2.warpPerspective(image, M, (W, H))  # Áp dụng biến đổi


# ============================================================
# 1B. GRID LINE DETECTION VIA MORPHOLOGY
# ============================================================

def detect_grid_points(image: np.ndarray,
                       vertical_scale: float = 0.015,  # Tỷ lệ chiều cao kernel ngang
                       horizontal_scale: float = 0.015,  # Tỷ lệ chiều rộng kernel dọc
                       min_point_area: int = 8,  # Diện tích tối thiểu của giao điểm
                       block_size: int = 35,  # Kích thước block adaptive threshold
                       block_offset: int = 7,  # Offset cho adaptive threshold
                       debug_prefix: Optional[str] = None) -> Dict[str, object]:
    """Phát hiện giao điểm lưới bằng: threshold -> morphological operations -> connected components.
    
    Phương pháp:
    1. Adaptive threshold để phân biệt đường lưới (nền sáng)
    2. Erode + dilate với kernel dọc/ngang tách các đường
    3. Tìm giao điểm (intersection) của các đường
    4. Connected components để lấy tâm các giao điểm
    """
    # Chuyển sang grayscale nếu cần
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Mờ để giảm nhiễu

    # Đảm bảo block_size là số lẻ (yêu cầu của adaptive threshold)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    # Adaptive threshold: đảo ngược để đường lưới thành trắng (255)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,  # So sánh với trung bình của vùng
        cv2.THRESH_BINARY_INV,  # Đảo ngược: đường = 255, nền = 0
        block_size,
        block_offset,  # Hạ threshold để tách tốt hơn
    )

    # Tính kích thước kernel dựa trên kích thước ảnh
    H, W = gray.shape
    v_len = max(3, int(H * vertical_scale))  # Chiều dài kernel cho đường dọc
    h_len = max(3, int(W * horizontal_scale))  # Chiều dài kernel cho đường ngang
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))  # Kernel dọc (1xN)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))  # Kernel ngang (Nx1)

    # Tách các đường dọc: erode loại bỏ những phần ngang, dilate phục hồi
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    # Tách các đường ngang: erode loại bỏ những phần dọc, dilate phục hồi
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    # Giao điểm = nơi hai đường cắt nhau
    intersections = cv2.bitwise_and(vertical_lines, horizontal_lines)
    intersections = cv2.dilate(intersections, np.ones((3, 3), np.uint8), iterations=1)  # Phục hồi giao điểm

    # Tìm các component liên thông và lấy tâm của chúng = giao điểm
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(intersections)
    points: List[Tuple[int, int]] = []
    for idx in range(1, num_labels):  # idx=0 là background
        if stats[idx, cv2.CC_STAT_AREA] < min_point_area:  # Lọc những component quá nhỏ
            continue
        cx, cy = centroids[idx]  # Tâm của giao điểm
        points.append((int(round(cx)), int(round(cy))))

    # Sắp xếp giao điểm theo thứ tự: từ trên xuống, trái sang phải
    points.sort(key=lambda p: (p[1], p[0]))

    # Vẽ các giao điểm lên ảnh để debug
    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)  # Vẽ điểm đỏ

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
# 2. CẤU HÌNH LAYOUT - ĐỊNH NGHĨA CÁC PHẦN TRÊ PHIẾU
# ============================================================

SECTIONS = {
    # PHẦN SỐ BÁO DANH (6 chữ số)
    "sobaodanh": {
        "y_range":     (0.075, 0.283),  # Vị trí Y của phần (tính theo tỷ lệ 0-1)
        "groups": [
            {"x_range": (0.740, 0.860), "label": "SoBaoDanh"},  # Vị trí X (tính theo tỷ lệ)
        ],
        "rows":         ["0","1","2","3","4","5","6","7","8","9"],  # Các chữ số 0-9
        "n_digit_cols": 6,  # 6 cột số
        "darkness_thresh":  85.0,  # Ngưỡng độ tối
        "otsu_threshold_upper": 175.0,  # Ngưỡng Otsu tối đa
    },
    # PHẦN MÃ ĐỀ (3 chữ số)
    "made": {
        "y_range":     (0.075, 0.283),
        "groups": [
            {"x_range": (0.900, 0.960), "label": "MaDe"},
        ],
        "rows":         ["0","1","2","3","4","5","6","7","8","9"],
        "n_digit_cols": 3,  # 3 cột số
        "darkness_thresh":  85.0,
        "otsu_threshold_upper": 175.0,
    },
    # PHẦN 1 - 40 câu hỏi trắc nghiệm (4 lựa chọn A,B,C,D)
    "phan1": {
        "y_range":     (0.345, 0.525),  # Phần chiếm vị trí Y từ 34.5% đến 52.5%
        "groups": [
            {"x_range": (0.065, 0.230), "q_start":  1},   # Nhóm 1: Câu 1-10
            {"x_range": (0.301, 0.470), "q_start": 11},   # Nhóm 2: Câu 11-20
            {"x_range": (0.538, 0.710), "q_start": 21},   # Nhóm 3: Câu 21-30
            {"x_range": (0.780, 0.950), "q_start": 31},   # Nhóm 4: Câu 31-40
        ],
        "choices":     ["A","B","C","D"],  # 4 lựa chọn
        "n_rows":      10,  # 10 hàng (10 câu/nhóm)
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.10,  # Ngưỡng tỷ lệ fill
    },
    # PHẦN 2 - 8 câu: Trả lời Đúng(D)/Sai(S)
    "phan2": {
        "y_range":     (0.598, 0.668),
        "groups": [
            {"x_range": (0.065, 0.230), "cau": (1,2)},    # Nhóm 1: Câu 1,2
            {"x_range": (0.301, 0.470), "cau": (3,4)},    # Nhóm 2: Câu 3,4
            {"x_range": (0.538, 0.710), "cau": (5,6)},    # Nhóm 3: Câu 5,6
            {"x_range": (0.780, 0.950), "cau": (7,8)},    # Nhóm 4: Câu 7,8
        ],
        "rows":        ["a","b","c","d"],  # 4 hàng (a,b,c,d)
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.10,
    },
    # PHẦN 3 - 6 câu: Nhập thứ tự (0-9, dấu phẩy, dấu trừ)
    "phan3": {
        "y_range":      (0.753, 0.970),
        "groups": [
            {"x_range": (0.066, 0.168), "cau": 1},   # Câu 1
            {"x_range": (0.220, 0.320), "cau": 2},   # Câu 2
            {"x_range": (0.372, 0.472), "cau": 3},   # Câu 3
            {"x_range": (0.520, 0.620), "cau": 4},   # Câu 4
            {"x_range": (0.675, 0.780), "cau": 5},   # Câu 5
            {"x_range": (0.825, 0.930), "cau": 6},   # Câu 6
        ],
        "rows":         ["-",",","0","1","2","3","4","5","6","7","8","9"],  # 12 ký tự
        "n_digit_cols": 4,  # 4 cột (nhập tối đa 4 chữ số)
        "otsu_threshold_upper": 175.0,
        "fill_ratio_thresh": 0.15,
    },
}


# ============================================================
# 3. DYNAMIC THRESHOLD ANALYSIS PER BOX
# ============================================================

def analyze_digit_section(gray: np.ndarray, cfg: Dict, H: int, W: int, section_name: str) -> List[Dict]:
    """Phân tích phần số (SỐ BÁO DANH hoặc MÃ ĐỀ) bằng độ tối của các ô.
    
    Phương pháp:
    - Tính độ tối (mean pixel value) của từng ô
    - Ô tối = điểm được tô = "filled"
    - Ô sáng = không tô = "empty"
    
    Args:
        gray: ảnh grayscale
        cfg: dict cấu hình của section
        H, W: chiều cao và chiều rộng ảnh
        section_name: tên phần (để hiển thị)
    
    Returns:
        danh sách các ô lưới với trạng thái filled/empty
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
    """Phân tích một ô lưới bằng ảnh binary từ phát hiện lưới.
    
    Chiến lược:
    - Đếm các pixel trắng (binary > 0) trong ô
    - Tính fill_ratio = white_pixels / total_pixels
    - Tỷ lệ cao = nhiều mực = ĐÃ TÔ (FILLED)
    - Nếu fill_ratio > ngưỡng => FILLED, nếu không => EMPTY
    
    Args:
        binary_image: ảnh binary từ phát hiện lưới (đường lưới = 255, nền = 0)
        x1, y1, x2, y2: tọa độ các góc của ô
        fill_ratio_thresh: ngưỡng tỷ lệ fill (> này = đã tô)
    
    Returns:
        dict chứa: fill_ratio, is_filled, box_bounds
    """
    # Đảm bảo tọa độ nằm trong giới hạn ảnh
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(binary_image.shape[1], x2)
    y2 = min(binary_image.shape[0], y2)
    
    # Kiểm tra ô là hợp lệ
    if x2 <= x1 or y2 <= y1:
        return {
            "fill_ratio": 0.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Lấy vùng ảnh của ô
    roi = binary_image[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "fill_ratio": 0.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Tính tỷ lệ fill: số pixel trắng (>0) / tổng pixel trong ô
    white_pixels = np.sum(roi > 0)  # Đếm pixel có giá trị > 0
    total_pixels = roi.size
    fill_ratio = white_pixels / max(total_pixels, 1)
    
    # Phân loại: tỷ lệ cao = nhiều mực = đã tô
    is_filled = fill_ratio > fill_ratio_thresh
    
    return {
        "fill_ratio": fill_ratio,
        "is_filled": is_filled,
        "box_bounds": (x1, y1, x2, y2),
    }

def calculate_box_otsu_threshold(roi: np.ndarray) -> float:
    """Tính ngưỡng Otsu cho một ô (ROI).
    
    Phương pháp Otsu: tìm ngưỡng tối ưu để chia ảnh thành 2 lớp (foreground/background)
    - Ngưỡng thấp = ô tối (nhiều mực) = FILLED
    - Ngưỡng cao = ô sáng (ít mực) = EMPTY
    
    Args:
        roi: vùng ảnh grayscale của ô
    
    Returns:
        otsu_threshold: ngưỡng tối ưu (0-255)
    """
    if roi.size == 0:
        return 128.0
    
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    return float(otsu_thresh)


def analyze_box_with_dynamic_threshold(warped: np.ndarray, 
                                       x1: int, y1: int, x2: int, y2: int, 
                                       otsu_threshold_upper: float = 175.0) -> dict:
    """Phân tích một ô lưới bằng ngưỡng Otsu động.
    
    Chiến lược:
    - Tính ngưỡng Otsu tối ưu cho từng ô
    - Ngưỡng Otsu thấp = ô tối = nhiều mực = ĐÃ TÔ (FILLED)
    - Ngưỡng Otsu cao = ô sáng = ít mực = EMPTY
    - Nếu otsu_threshold <= otsu_threshold_upper => FILLED
    
    Args:
        warped: ảnh đã chỉnh thẳng
        x1, y1, x2, y2: tọa độ các góc của ô
        otsu_threshold_upper: ngưỡng Otsu (≤ này = đã tô)
    
    Returns:
        dict chứa: otsu_threshold, is_filled, box_bounds
    """
    # Chuyển sang grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Đảm bảo tọa độ nằm trong giới hạn ảnh
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray.shape[1], x2)
    y2 = min(gray.shape[0], y2)
    
    # Kiểm tra ô là hợp lệ
    if x2 <= x1 or y2 <= y1:
        return {
            "otsu_threshold": 128.0,  # Giá trị mặc định giữa (trung bình)
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Lấy vùng ảnh của ô
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "otsu_threshold": 128.0,
            "is_filled": False,
            "box_bounds": (x1, y1, x2, y2),
        }
    
    # Tính ngưỡng Otsu động cho ô này
    # Ngưỡng thấp = ô tối = được tô
    otsu_thresh = calculate_box_otsu_threshold(roi)
    
    # Phân loại: ngưỡng Otsu thấp => đã tô
    is_filled = otsu_thresh <= otsu_threshold_upper
    
    return {
        "otsu_threshold": otsu_thresh,
        "is_filled": is_filled,
        "box_bounds": (x1, y1, x2, y2),
    }


def cell_darkness(gray: np.ndarray, cx: int, cy: int, r: int = 7) -> float:
    """Tính độ tối trung bình của một ô hình tròn.
    
    Args:
        gray: ảnh grayscale
        cx, cy: tọa độ tâm ô
        r: bán kính hình tròn
    
    Returns:
        giá trị pixel trung bình (thấp = tối = được tô)
    """
    # Lấy vùng hình vuông quanh tâm
    x1, y1 = max(0, cx-r), max(0, cy-r)
    x2, y2 = min(gray.shape[1], cx+r), min(gray.shape[0], cy+r)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 255.0  # Mặc định: sáng (empty)
    
    # Tạo mặt nạ hình tròn để chỉ lấy pixel trong vùng tròn
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center = (roi.shape[1]//2, roi.shape[0]//2)
    radius = min(roi.shape[0], roi.shape[1]) // 2 - 1
    cv2.circle(mask, center, max(1, radius), 255, -1)  # Vẽ hình tròn trắng
    
    # Tính trung bình pixel trong vùng tròn
    pixels = roi[mask > 0]  # Lấy pixel trong mặt nạ
    return float(np.mean(pixels)) if len(pixels) > 0 else 255.0


def parse_digit_grid(gray: np.ndarray,
                     x_pcts: List[float],  # Vị trí x của từng cột (0-1)
                     y_pcts: List[float],  # Vị trí y của tất cả các hàng (0-1)
                     r: int = 7) -> str:
    """Phân tích một dãy các ô số (SỐ BÁO DANH hoặc MÃ ĐỀ) bằng độ tối.
    
    Chiến lược:
    - Tính độ tối (mean pixel value) cho từng ô
    - Với mỗi cột, tìm chữ số có độ tối THẤP NHẤT (tối nhất = được tô)
    - Ví dụ: nếu cột 0 có Y5 tối nhất => chữ số là '5'
    
    Args:
        gray: ảnh grayscale
        x_pcts: tọa độ x của từng cột (0-1)
        y_pcts: tọa độ y tất cả các hàng/chữ số (0-1)
        r: bán kính ô hình tròn
    
    Returns:
        chuỗi các chữ số, ví dụ: '001013' hoặc '004'
    """
    H, W = gray.shape
    digits_str = []
    
    # Với mỗi cột
    for xp in x_pcts:
        cx = int(xp * W)  # Tọa độ x của cột
        # Tính độ tối cho từng ô trong cột này
        col_vals = [(cell_darkness(gray, cx, int(yp*H), r), di)
                    for di, yp in enumerate(y_pcts)]
        
        # Tìm chữ số có độ tối THẤp NHẤT (tối nhất = được tô)
        best_dark, best_digit = min(col_vals, key=lambda v: v[0])
        
        # Thêm chữ số tốt nhất vào kết quả
        digits_str.append(str(best_digit))
    
    return "".join(digits_str)


def render_digit_grid(out: np.ndarray, gray: np.ndarray,
                      x_pcts: List[float], y_pcts: List[float],
                      label: str, r: int = 7) -> None:
    """Vẽ trực quan các ô số với ô được chọn (được tô) màu xanh.
    
    Phương pháp:
    - Vẽ một hình tròn xanh lá cây (opacity 50%) ở vị trí ô được tô
    - Các ô khác không được tô
    
    Args:
        out: ảnh đầu ra để vẽ lên
        gray: ảnh grayscale để tính độ tối
        x_pcts: tọa độ x tính bằng phần trăm
        y_pcts: tọa độ y tính bằng phần trăm
        label: nhãn của phần (e.g. 'SBD')
        r: bán kính hình tròn
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
    """Vẽ tờ phiếu: nhóm box + đường lưới + vùng được tô với opacity (50%).
    
    Vẽ đường viền:
    - Sắc xanh (đậm) lính (2 px) cho ô được tô
    - Sắc cam nhạt (đường) cho ô không được tô
    - Overlay xanh 50% opacity cho phần được tô
    """
    # Chuyển sang grayscale để cao chức vẽ digit grid
    out = warped.copy()
    H, W = warped.shape[:2]
    overlay = out.copy()  # Overlay lưu ý tương copy để thực hiện transparency
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Màu sắc
    GREEN = (0, 255, 0)  # Xanh: ô được tô
    GRID_COLOR = (0, 255, 0)  # Xanh: đường lưới
    BOX_COLOR = (100, 200, 0)  # Xanh vàng: nhóm box
    
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
    
    # Vẽ overlay xác ĐIỂM CHỈ ÔNG ĐƯỢC TÔ với màu xanh (50% opacity)
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN, -1)  # Fill với màu xanh
    
    # Blend overlay lên ảnh gốc với transparency (50%)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    
    # Vẽ đường viền đặc LOẠI: ÔNG ĐƯỢC TÔ (xanh lá)
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)  # Viền xanh đậm (2 px)
    
    # Vẽ đường viền nhỏ cho ÔNG CHƯA TÔ (cam nhạt)
    UNFILLED_COLOR = (0, 165, 255)  # Cam nhạt
    for section in ["sobaodanh", "made", "phan1", "phan2", "phan3"]:
        for item in grid_data[section]:
            if not item["filled"]:
                x1, y1, x2, y2 = item["box_bounds"]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), UNFILLED_COLOR, 1)  # Viền cam nhạt
    
    cv2.imwrite(out_path, out)
    print(f"\n[SAVED] {out_path}")


# ============================================================
# 6. MAIN
# ============================================================

def process(image_path: str, out_dir: str = "./outputs_grid", debug: bool = True):
    """Xử lý một ảnh tờ thi:
    1. Tìm 4 góc
    2. Chỉnh thẳng (perspective transform)
    3. Phát hiện đường lưới bằng morphology
    4. Phân tích tất cả các ô (filled/empty)
    5. Vẽ kết quả
    
    Args:
        image_path: đường dẫn đến ảnh thi
        out_dir: thư mục đầu ra
        debug: có lưu debug images không
    """
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  PHÂN TÍCH TẠO PHƠIẾU VỚI NGƯỠNG THẬP ĐỘNG")
    print(f"  {image_path}")
    print(f"{'='*60}")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    
    # Bước 1: Tìm 4 góc của tờ
    corners, corners_valid = find_corner_markers(image, debug_out=f"{out_dir}/01_corners.jpg" if debug else None)
    
    # Kiểm tra xem có tìm được đủ marker không
    if not corners_valid:
        print("\n[ERROR] Không tìm thấy đủ điểm góc - Hủy xử lý")
        return
    
    # Bước 2: Chỉnh thẳng tờ
    warped = warp(image, corners)
    if debug:
        cv2.imwrite(f"{out_dir}/02_warped.jpg", warped)
    
    # Bước 3: Phát hiện lưới (giao điểm các đường)
    grid_debug = detect_grid_points(warped, debug_prefix=f"{out_dir}/grid" if debug else None)
    print(f"[LƯỚI] {len(grid_debug['points'])} giao điểm phát hiện bằng morphology")
    
    # Bước 4: Phân tích tất cả các ô
    grid_data = analyze_grid(warped, grid_debug)
    
    # Bước 5: Vẽ kết quả
    visualize(warped, grid_data, f"{out_dir}/03_grid_filled.jpg")
    
    print(f"\n[OK] Kết quả: {out_dir}/")


if __name__ == "__main__":
    # Lấy đường dẫn ảnh từ tham số dòng lệnh hoặc dùng mẫu mặc định
    img_path = sys.argv[1] if len(sys.argv) > 1 else "./PhieuQG/PhieuQG.0045.jpg"
    
    # Xử lý ảnh với chế độ debug (lưu ảnh trung gian)
    process(img_path, debug=True)
