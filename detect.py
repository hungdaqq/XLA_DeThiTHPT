from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def resize_image_for_inference(
    image: np.ndarray,
    max_side: int = 2200,
) -> Tuple[np.ndarray, float]:
    # Chuẩn hóa ảnh lớn về kích thước phù hợp để giảm thời gian inference.
    if image is None or image.size == 0:
        return image, 1.0

    h, w = image.shape[:2]
    longest_side = max(int(h), int(w))
    max_side = int(max(256, max_side))
    if longest_side <= max_side:
        return image, 1.0

    scale = float(max_side) / float(longest_side)
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Sắp xếp 4 đỉnh tứ giác theo thứ tự chuẩn: trên-trái, trên-phải, dưới-phải, dưới-trái.

    Quy trình:
    1. Chuẩn hóa mảng điểm về dạng (4, 2) float32.
    2. Dùng tổng tọa độ `(x + y)` để tìm góc trên-trái và dưới-phải.
    3. Dùng hiệu tọa độ `(x - y)` để tách hai góc còn lại.
    4. Trả về mảng 4 điểm theo thứ tự cố định để các bước nội suy hoạt động ổn định.

    Args:
        points: Mảng chứa 4 điểm của contour/tứ giác.

    Returns:
        Mảng `np.ndarray` kích thước `(4, 2)` theo thứ tự chuẩn.
    """
    pts = points.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != 4:
        raise ValueError("Expected 4 points to order a quadrilateral")

    ordered = np.zeros((4, 2), dtype=np.float32)
    # Mẹo hình học cổ điển:
    # - Điểm có tổng (x+y) nhỏ nhất thường là góc trên-trái.
    # - Điểm có tổng lớn nhất thường là góc dưới-phải.
    s = pts.sum(axis=1)
    # Hiệu (x-y) giúp tách 2 góc còn lại:
    # - Nhỏ nhất ~ trên-phải, lớn nhất ~ dưới-trái (theo hệ tọa độ ảnh OpenCV).
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def _box_to_quad(box: np.ndarray) -> np.ndarray:
    # Hàm phụ chuẩn hóa contour/box về dạng dễ xử lý.
    """
    Chuẩn hóa contour bất kỳ về tứ giác 4 điểm ổn định.

    Quy trình:
    1. Nếu contour đã có đúng 4 điểm thì chỉ cần sắp lại thứ tự chuẩn.
    2. Nếu chưa đủ/không đúng 4 điểm, thử xấp xỉ polygon bằng `approxPolyDP`.
    3. Nếu vẫn không được 4 điểm, fallback về `minAreaRect` để luôn có tứ giác hợp lệ.

    Args:
        box: Contour đầu vào (đa giác bất kỳ).

    Returns:
        Tứ giác 4 điểm đã chuẩn hóa thứ tự.
    """
    contour = box.reshape(-1, 2).astype(np.float32)
    if contour.shape[0] == 4:
        return _order_quad_points(contour)

    # Ưu tiên xấp xỉ polygon từ contour gốc để giữ biên sát thực tế.
    peri = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2), 0.02 * peri, True)
    if approx.shape[0] == 4:
        return _order_quad_points(approx.reshape(-1, 2))

    # Fallback cuối cùng: minAreaRect luôn trả 4 điểm ổn định ngay cả khi contour méo.
    rect = cv2.minAreaRect(contour.reshape(-1, 1, 2))
    quad = cv2.boxPoints(rect)
    return _order_quad_points(quad)


def _lerp_point(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    # Hàm phụ nội suy tuyến tính giữa hai điểm.
    """
    Nội suy tuyến tính giữa hai điểm theo hệ số t.

    Args:
        p0: Điểm đầu.
        p1: Điểm cuối.
        t: Hệ số nội suy (0 -> p0, 1 -> p1).

    Returns:
        Điểm nội suy trên đoạn nối p0-p1.
    """
    return p0 + (p1 - p0) * float(t)


def _point_on_quad(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    # Hàm phụ ánh xạ tọa độ trong miền chuẩn sang miền ảnh.
    """
    Nội suy song tuyến tính trên tứ giác đã sắp thứ tự với u, v thuộc [0, 1].

    Args:
        quad: Tứ giác 4 điểm theo thứ tự chuẩn.
        u: Tọa độ chuẩn theo trục ngang, trong [0, 1].
        v: Tọa độ chuẩn theo trục dọc, trong [0, 1].

    Returns:
        Điểm ảnh tương ứng sau khi ánh xạ từ miền chuẩn.
    """
    # Nội suy 2 bước:
    # 1) Nội suy theo chiều ngang trên cạnh trên/dưới.
    # 2) Nội suy theo chiều dọc giữa 2 điểm vừa tìm được.
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
    # Hàm phụ co vùng quan tâm để giảm nhiễu biên.
    """
    Tạo tứ giác vùng làm việc bên trong từ tứ giác gốc và các offset đầu/cuối.

    Args:
        quad: Tứ giác gốc của box.
        start_offset_x: Tỉ lệ lùi đầu theo trục X.
        start_offset_y: Tỉ lệ lùi đầu theo trục Y.
        end_offset_x: Tỉ lệ lùi cuối theo trục X.
        end_offset_y: Tỉ lệ lùi cuối theo trục Y.

    Returns:
        Tứ giác vùng trong sau khi áp offset và clamp an toàn.
    """
    # Clamp offset để không bị đảo hình hoặc co về diện tích âm.
    u0 = float(np.clip(start_offset_x, 0.0, 0.95))
    v0 = float(np.clip(start_offset_y, 0.0, 0.95))
    # Bắt buộc u1 > u0 và v1 > v0 bằng epsilon nhỏ để tránh quad suy biến.
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
    # Hàm hỗ trợ vẽ hình học/phần tử phụ trợ trong pipeline.
    """
    Vẽ lưới hàng/cột đều trên một tứ giác phối cảnh.

    Args:
        image: Ảnh đích để vẽ.
        quad: Tứ giác vùng lưới.
        grid_cols: Số cột lưới.
        grid_rows: Số hàng lưới.
        color: Màu đường lưới (BGR).
        thickness: Độ dày nét vẽ.

    Returns:
        Không trả về giá trị.
    """
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    # Kẻ các line dọc bằng cách đi từ cạnh trên xuống cạnh dưới tại cùng tỉ lệ t.
    for col in range(1, grid_cols):
        t = col / float(grid_cols)
        p_top = _lerp_point(quad[0], quad[1], t)
        p_bottom = _lerp_point(quad[3], quad[2], t)
        cv2.line(image, tuple(np.round(p_top).astype(int)), tuple(np.round(p_bottom).astype(int)), color, thickness)

    # Kẻ các line ngang bằng cách đi từ cạnh trái sang cạnh phải tại cùng tỉ lệ t.
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
    # Hàm hỗ trợ vẽ hình học/phần tử phụ trợ trong pipeline.
    """
    Vẽ các ô lưới theo pattern cột riêng cho từng hàng.

    Args:
        image: Ảnh đích để vẽ.
        quad: Tứ giác vùng lưới.
        grid_cols: Số cột tổng của lưới.
        grid_rows: Số hàng tổng của lưới.
        row_col_patterns: Danh sách cột được phép vẽ cho từng hàng.
        color: Màu nét ô (BGR).
        thickness: Độ dày nét vẽ.

    Returns:
        Không trả về giá trị.
    """
    cv2.polylines(image, [quad.astype(np.int32)], True, color, thickness)

    if grid_cols <= 0 or grid_rows <= 0:
        return

    for row in range(grid_rows):
        # Nếu có pattern thì chỉ vẽ các cột được phép ở hàng hiện tại.
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

            # Mỗi cell vẫn là một quad phối cảnh, không giả định hình chữ nhật trục chuẩn.
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
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Kiểm tra kích thước box trước khi dùng để vẽ/chấm.

    Args:
        box: Contour box cần kiểm tra.
        box_idx: Chỉ số box trong danh sách để in cảnh báo dễ theo dõi.

    Returns:
        Tuple `(x, y, w, h)` nếu hợp lệ, ngược lại trả về `None`.
    """
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
    # Hàm phụ dựng cấu trúc dữ liệu trung gian dùng lại nhiều nơi.
    """
    Tạo metadata lưới dùng chung cho các biến thể hàm trích xuất grid.

    Args:
        box_idx: Chỉ số box hiện tại.
        box_bounds: Bounding rect của box `(x, y, w, h)`.
        region_quad: Tứ giác vùng lưới sau khi áp offset.
        grid_rows: Số hàng lưới.
        grid_cols: Số cột lưới.
        extra: Metadata phụ để gắn thêm theo từng bài toán.

    Returns:
        Dictionary metadata nếu vùng hợp lệ, ngược lại trả về `None`.
    """
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


def _quad_cell_at(region_quad: np.ndarray, row: int, col: int, rows: int, cols: int) -> np.ndarray:
    # Hàm phụ thao tác trên tứ giác phối cảnh.
    """
    Lấy tứ giác của một ô con theo chỉ số hàng/cột trong vùng tứ giác phối cảnh.

    Args:
        region_quad: Tứ giác vùng lưới.
        row: Chỉ số hàng của ô.
        col: Chỉ số cột của ô.
        rows: Tổng số hàng.
        cols: Tổng số cột.

    Returns:
        Tứ giác 4 điểm của ô cần lấy.
    """
    # Chia miền chuẩn [0,1]x[0,1] theo chỉ số hàng/cột rồi ánh xạ lại lên quad thực.
    u0 = col / float(max(1, cols))
    u1 = (col + 1) / float(max(1, cols))
    v0 = row / float(max(1, rows))
    v1 = (row + 1) / float(max(1, rows))

    return np.array(
        [
            _point_on_quad(region_quad, u0, v0),
            _point_on_quad(region_quad, u1, v0),
            _point_on_quad(region_quad, u1, v1),
            _point_on_quad(region_quad, u0, v1),
        ],
        dtype=np.float32,
    )


def _shrink_quad_towards_center(quad: np.ndarray, margin_ratio: float) -> np.ndarray:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Co tứ giác về tâm để giảm ảnh hưởng của viền lên phép đo fill-ratio.

    Args:
        quad: Tứ giác ô gốc.
        margin_ratio: Tỉ lệ co về tâm.

    Returns:
        Tứ giác đã co, phù hợp để chấm vùng bên trong bubble.
    """
    # Giới hạn margin để giữ đủ diện tích đo; >0.45 thường làm mất vùng bubble có ích.
    ratio = float(np.clip(margin_ratio, 0.0, 0.45))
    if ratio <= 0:
        return quad.astype(np.float32)

    center = np.mean(quad.astype(np.float32), axis=0)
    return center + (quad.astype(np.float32) - center) * (1.0 - ratio)


def _fill_ratio_in_quad(binary_image: np.ndarray, quad: np.ndarray) -> float:
    # Hàm phụ tính toán tỉ lệ/điểm số trong vùng quan tâm.
    """
    Tính tỉ lệ điểm trắng trong mặt nạ tứ giác.

    Args:
        binary_image: Ảnh nhị phân đầu vào.
        quad: Tứ giác vùng cần chấm.

    Returns:
        Giá trị fill-ratio trong khoảng [0, 1].
    """
    if binary_image is None or binary_image.size == 0:
        return 0.0

    h, w = binary_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(quad).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(mask, pts, 255)

    pixels = binary_image[mask > 0]
    if pixels.size == 0:
        return 0.0
    # Ảnh binary dùng THRESH_BINARY_INV nên pixel != 0 tương ứng vùng tô đậm.
    return float(np.count_nonzero(pixels)) / float(pixels.size)


def _estimate_circle_from_quad(quad: np.ndarray, radius_scale: float = 0.46) -> Tuple[np.ndarray, float]:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Ước lượng tâm và bán kính bubble từ tứ giác của ô.

    Quy trình:
    1. Lấy tâm bằng trung bình 4 đỉnh.
    2. Tính kích thước trung bình theo ngang/dọc của ô.
    3. Chọn cạnh ngắn hơn làm chuẩn để tránh bán kính tràn viền ở ô méo.
    4. Nhân với `radius_scale` (đã clip) để lấy bán kính cuối.

    Args:
        quad: Tứ giác của ô bubble.
        radius_scale: Hệ số co bán kính so với nửa cạnh chuẩn.

    Returns:
        Tuple `(center, radius)` gồm tâm và bán kính ước lượng.
    """
    q = quad.astype(np.float32)
    center = np.mean(q, axis=0)

    top_w = float(np.linalg.norm(q[1] - q[0]))
    bottom_w = float(np.linalg.norm(q[2] - q[3]))
    left_h = float(np.linalg.norm(q[3] - q[0]))
    right_h = float(np.linalg.norm(q[2] - q[1]))

    avg_w = 0.5 * (top_w + bottom_w)
    avg_h = 0.5 * (left_h + right_h)
    # Lấy cạnh ngắn hơn để tránh bán kính tràn ra ngoài ô khi ô bị méo phối cảnh.
    base_radius = 0.5 * min(avg_w, avg_h)
    # Chặn scale ở khoảng an toàn để giảm outlier khi người dùng set tham số quá tay.
    radius = max(1.0, base_radius * float(np.clip(radius_scale, 0.20, 0.60)))
    return center, radius


def _detect_single_circle_hough_in_quad(
    binary_image: np.ndarray,
    quad: np.ndarray,
    radius_scale: float = 0.46,
) -> Tuple[np.ndarray, float, bool]:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Phát hiện một vòng tròn bubble trong ô bằng Hough + NMS + gộp có trọng số.

    Quy trình:
    1. Ước lượng tâm/bán kính ban đầu từ hình học của ô.
    2. Chạy Hough trên ROI đã mask để lấy nhiều candidate vòng tròn.
    3. Chấm điểm candidate theo tín hiệu cạnh và độ phù hợp với prior hình học.
    4. Dùng NMS để loại trùng, sau đó gộp cụm có trọng số để ổn định kết quả cuối.

    Args:
        binary_image: Ảnh nhị phân đầu vào.
        quad: Tứ giác của ô bubble.
        radius_scale: Hệ số ước lượng bán kính ban đầu.

    Returns:
        Tuple `(center, radius, found)` với:
        - `center`: tâm vòng tròn.
        - `radius`: bán kính vòng tròn.
        - `found`: True nếu Hough tìm được candidate đáng tin cậy.
    """
    est_center, est_radius = _estimate_circle_from_quad(quad, radius_scale=radius_scale)
    if binary_image is None or binary_image.size == 0:
        return est_center, est_radius, False

    h, w = binary_image.shape[:2]
    pts = np.round(quad).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    x, y, bw, bh = cv2.boundingRect(pts)
    if bw < 8 or bh < 8:
        return est_center, est_radius, False

    roi = binary_image[y : y + bh, x : x + bw]
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    local_quad = pts.copy()
    local_quad[:, 0] -= x
    local_quad[:, 1] -= y

    roi_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, local_quad, 255)

    # Chỉ giữ gradient trong ô hiện tại để Hough không bị nhiễu bởi ô lân cận.
    masked = cv2.bitwise_and(roi, roi, mask=roi_mask)
    prep = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(prep, 30, 100)

    # Bó bán kính quanh estimate hình học để giảm nhảy kích thước giữa các frame/ảnh.
    min_r = max(2, int(round(est_radius * 0.72)))
    max_r = max(min_r + 1, int(round(est_radius * 1.22)))
    min_dist = max(4, int(round(est_radius * 2.2)))

    def _ring_edge_support(cx_local: float, cy_local: float, r_local: float) -> float:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Đo mức "đỡ" theo biên: vòng tròn nào ăn khớp cạnh Canny tốt hơn sẽ được ưu tiên.
        ring = np.zeros((bh, bw), dtype=np.uint8)
        cxy = (int(round(cx_local)), int(round(cy_local)))
        rr = max(1, int(round(r_local)))
        cv2.circle(ring, cxy, rr, 255, 2)
        ring = cv2.bitwise_and(ring, roi_mask)
        denom = int(np.count_nonzero(ring))
        if denom <= 0:
            return 0.0
        num = int(np.count_nonzero(cv2.bitwise_and(edges, ring)))
        return float(num) / float(denom)

    def _circle_iou(ca: Dict[str, float], cb: Dict[str, float]) -> float:
        # Hàm phụ thao tác với hình tròn bubble/mặt nạ tròn.
        # IoU hình tròn dùng để so mức trùng giữa hai candidate bất kể lệch tâm nhỏ.
        r1 = float(max(1.0, ca["r"]))
        r2 = float(max(1.0, cb["r"]))
        dx = float(ca["cx"] - cb["cx"])
        dy = float(ca["cy"] - cb["cy"])
        d = float(np.hypot(dx, dy))

        if d >= (r1 + r2):
            inter = 0.0
        elif d <= abs(r1 - r2):
            inter = float(np.pi * min(r1, r2) ** 2)
        else:
            a1 = float(np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0)))
            a2 = float(np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0)))
            term = float((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            inter = r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * np.sqrt(max(0.0, term))

        area1 = float(np.pi * r1 * r1)
        area2 = float(np.pi * r2 * r2)
        union = area1 + area2 - inter
        if union <= 1e-6:
            return 0.0
        return float(inter / union)

    def _is_same_circle(ca: Dict[str, float], cb: Dict[str, float]) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Kết hợp 2 tiêu chí:
        # - gần nhau về tâm + bán kính (nhanh)
        # - hoặc IoU đủ lớn (an toàn khi bị méo nhẹ)
        center_dist = float(np.hypot(ca["cx"] - cb["cx"], ca["cy"] - cb["cy"]))
        radius_ref = float(max(1.0, max(ca["r"], cb["r"], est_radius)))
        radius_diff = abs(float(ca["r"] - cb["r"])) / radius_ref
        if center_dist <= 0.48 * radius_ref and radius_diff <= 0.34:
            return True
        return _circle_iou(ca, cb) >= 0.35

    candidates: List[Dict[str, float]] = []
    search_pairs = ((1.1, 12), (1.2, 14), (1.1, 10), (1.2, 11), (1.3, 9))
    for dp, param2 in search_pairs:
        # Quét nhẹ nhiều cặp tham số Hough để tăng độ bền trên scan khó.
        circles = cv2.HoughCircles(
            prep,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=float(min_dist),
            param1=80,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None or circles.size == 0:
            continue

        for c in circles[0][:8]:
            cx_local = float(c[0])
            cy_local = float(c[1])
            rr = float(max(1.0, c[2]))

            cx = cx_local + float(x)
            cy = cy_local + float(y)
            center_norm = float(np.hypot(cx - est_center[0], cy - est_center[1])) / float(max(1.0, est_radius))
            radius_norm = abs(rr - float(est_radius)) / float(max(1.0, est_radius))
            # prior_score: tin vào hình học; edge_score: tin vào tín hiệu cạnh thực.
            prior_score = max(0.0, 1.0 - (0.70 * center_norm) - (0.60 * radius_norm))
            edge_score = _ring_edge_support(cx_local, cy_local, rr)
            score = (0.58 * edge_score) + (0.42 * prior_score)

            candidates.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "r": rr,
                    "score": float(score),
                }
            )

        if len(candidates) >= 24:
            break

    if not candidates:
        return est_center, est_radius, False

    candidates = sorted(candidates, key=lambda it: float(it["score"]), reverse=True)

    # NMS cứng để loại candidate gần trùng trước khi hợp cụm.
    kept: List[Dict[str, float]] = []
    for cand in candidates:
        if any(_is_same_circle(cand, k) for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= 8:
            break

    if not kept:
        return est_center, est_radius, False

    # Hợp cụm bằng trung bình có trọng số score để ổn định tâm/bán kính đầu ra.
    clusters: List[List[Dict[str, float]]] = [[] for _ in kept]
    for cand in candidates:
        best_idx = -1
        best_dist = float("inf")
        for idx_keep, keep in enumerate(kept):
            if not _is_same_circle(cand, keep):
                continue
            dist = float(np.hypot(cand["cx"] - keep["cx"], cand["cy"] - keep["cy"]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx_keep
        if best_idx >= 0:
            clusters[best_idx].append(cand)

    merged: List[Dict[str, float]] = []
    for idx_keep, keep in enumerate(kept):
        cluster = clusters[idx_keep] if clusters[idx_keep] else [keep]
        ws = np.array([max(1e-3, float(c["score"])) for c in cluster], dtype=np.float32)
        cx_vals = np.array([float(c["cx"]) for c in cluster], dtype=np.float32)
        cy_vals = np.array([float(c["cy"]) for c in cluster], dtype=np.float32)
        r_vals = np.array([float(c["r"]) for c in cluster], dtype=np.float32)

        cxm = float(np.sum(ws * cx_vals) / np.sum(ws))
        cym = float(np.sum(ws * cy_vals) / np.sum(ws))
        rm = float(np.sum(ws * r_vals) / np.sum(ws))
        # Candidate có nhiều phiếu bầu tương đồng sẽ được bonus nhẹ.
        support_bonus = 0.08 * float(len(cluster) - 1)
        base_score = float(np.max([float(c["score"]) for c in cluster])) + support_bonus

        # Phạt nếu lệch quá xa estimate hình học ban đầu.
        center_penalty = float(np.hypot(cxm - est_center[0], cym - est_center[1])) / float(max(1.0, est_radius))
        radius_penalty = abs(rm - float(est_radius)) / float(max(1.0, est_radius))
        final_score = base_score - (0.25 * center_penalty) - (0.20 * radius_penalty)

        merged.append({"cx": cxm, "cy": cym, "r": max(1.0, rm), "score": final_score})

    if not merged:
        return est_center, est_radius, False

    best = max(merged, key=lambda it: float(it["score"]))
    return np.array([best["cx"], best["cy"]], dtype=np.float32), float(best["r"]), True


def _circle_polygon(center: np.ndarray, radius: float, n_pts: int = 28) -> np.ndarray:
    # Hàm phụ thao tác với hình tròn bubble/mặt nạ tròn.
    """
    Xấp xỉ hình tròn bằng polygon để vẽ overlay hoặc tạo mặt nạ.

    Args:
        center: Tâm vòng tròn.
        radius: Bán kính vòng tròn.
        n_pts: Số đỉnh polygon dùng để xấp xỉ.

    Returns:
        Mảng điểm polygon biểu diễn hình tròn.
    """
    cx, cy = float(center[0]), float(center[1])
    rr = float(max(1.0, radius))
    angles = np.linspace(0.0, 2.0 * np.pi, num=max(8, int(n_pts)), endpoint=False)
    pts = np.stack([cx + rr * np.cos(angles), cy + rr * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


def _fill_ratio_in_circle(
    binary_image: np.ndarray,
    quad: np.ndarray,
    radius_scale: float = 0.46,
    border_exclude_ratio: float = 0.10,
    use_hough_detection: bool = False,
) -> Tuple[float, np.ndarray, bool]:
    # Hàm phụ tính toán tỉ lệ/điểm số trong vùng quan tâm.
    """
    Tính tỉ lệ điểm trắng bên trong vùng tròn bubble của một ô.

    Hỗ trợ 2 chế độ:
    - Ước lượng hình học đơn giản từ ô.
    - Hough-circle để tăng độ bền khi bubble méo/nhòe.

    Args:
        binary_image: Ảnh nhị phân đầu vào.
        quad: Tứ giác ô cần chấm.
        radius_scale: Hệ số bán kính bubble.
        border_exclude_ratio: Tỉ lệ bỏ viền ngoài bubble khi đo fill-ratio.
        use_hough_detection: Bật/tắt chế độ Hough-circle.

    Returns:
        Tuple `(fill_ratio, score_poly, circle_found)` gồm:
        - `fill_ratio`: tỉ lệ tô.
        - `score_poly`: polygon vùng thực tế dùng để chấm.
        - `circle_found`: True nếu Hough tìm được vòng tròn tin cậy.
    """
    if binary_image is None or binary_image.size == 0:
        return 0.0, quad.astype(np.float32), False

    h, w = binary_image.shape[:2]
    qmask = np.zeros((h, w), dtype=np.uint8)
    qpts = np.round(quad).astype(np.int32)
    qpts[:, 0] = np.clip(qpts[:, 0], 0, w - 1)
    qpts[:, 1] = np.clip(qpts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(qmask, qpts, 255)

    def _score_for_circle(center_in: np.ndarray, outer_r_in: float) -> Tuple[float, float]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Loại viền ngoài của bubble (vốn hay đậm hơn nền) để giảm false-positive.
        inner_r_local = outer_r_in * (1.0 - float(np.clip(border_exclude_ratio, 0.0, 0.4)))
        inner_r_local = max(1.0, inner_r_local)

        cmask = np.zeros((h, w), dtype=np.uint8)
        cxy = tuple(np.round(center_in).astype(np.int32))
        cv2.circle(cmask, cxy, int(round(inner_r_local)), 255, -1)

        mask = cv2.bitwise_and(qmask, cmask)
        pixels = binary_image[mask > 0]
        if pixels.size == 0:
            return 0.0, inner_r_local
        return float(np.count_nonzero(pixels)) / float(pixels.size), inner_r_local

    if use_hough_detection:
        base_scale = float(np.clip(radius_scale, 0.20, 0.60))
        # Thử thêm scale lân cận vì một scale cố định có thể hụt trên ô méo cục bộ.
        candidate_scales: List[float] = []
        for s in (base_scale, base_scale - 0.04, base_scale + 0.04):
            s_clipped = float(np.clip(s, 0.20, 0.60))
            if all(abs(s_clipped - existing) > 1e-6 for existing in candidate_scales):
                candidate_scales.append(s_clipped)

        best_ratio = -1.0
        best_center: Optional[np.ndarray] = None
        best_inner_r = 1.0
        found_any = False
        base_fallback: Optional[Tuple[float, np.ndarray, float]] = None

        for s in candidate_scales:
            center_s, outer_r_s, found_s = _detect_single_circle_hough_in_quad(
                binary_image,
                quad,
                radius_scale=s,
            )
            ratio_s, inner_r_s = _score_for_circle(center_s, outer_r_s)

            if abs(s - base_scale) <= 1e-6:
                # Lưu phương án baseline để fallback khi Hough không tìm được vòng tròn nào.
                base_fallback = (ratio_s, center_s, inner_r_s)

            if not found_s:
                continue

            found_any = True
            if ratio_s > best_ratio:
                best_ratio = ratio_s
                best_center = center_s
                best_inner_r = inner_r_s

        if found_any and best_center is not None:
            return best_ratio, _circle_polygon(best_center, best_inner_r), True

        if base_fallback is not None:
            ratio_fb, center_fb, inner_r_fb = base_fallback
            return ratio_fb, _circle_polygon(center_fb, inner_r_fb), False

        center, outer_r = _estimate_circle_from_quad(quad, radius_scale=base_scale)
        ratio, inner_r = _score_for_circle(center, outer_r)
        return ratio, _circle_polygon(center, inner_r), False

    center, outer_r = _estimate_circle_from_quad(quad, radius_scale=radius_scale)
    ratio, inner_r = _score_for_circle(center, outer_r)
    return ratio, _circle_polygon(center, inner_r), False


def evaluate_grid_fill_from_binary(
    binary_image: np.ndarray,
    grid_info: List[Dict[str, object]],
    fill_ratio_thresh: float,
    inner_margin_ratio: float = 0.18,
    mask_mode: str = "quad",
    circle_radius_scale: float = 0.46,
    circle_border_exclude_ratio: float = 0.10,
    hough_only_ambiguous: bool = True,
    hough_ambiguity_margin: float = 0.08,
    hough_max_cells: int = 96,
) -> List[Dict[str, object]]:
    # Đánh giá và trả về chỉ số/kết quả quyết định cho bước này.
    """
    Đánh giá tô/không tô cho từng ô lưới dựa trên fill-ratio của ảnh nhị phân.

    Hỗ trợ ba chế độ mask:
    - `quad`: chấm theo tứ giác ô.
    - `circle`: chấm theo vùng tròn ước lượng.
    - `hough-circle`: chấm theo vùng tròn có tinh chỉnh Hough.

    Args:
        binary_image: Ảnh nhị phân đầu vào.
        grid_info: Metadata lưới của các box cần chấm.
        fill_ratio_thresh: Ngưỡng phân loại ô tô.
        inner_margin_ratio: Tỉ lệ co vùng chấm trong mỗi ô.
        mask_mode: Chế độ mặt nạ (`quad`/`circle`/`hough-circle`).
        circle_radius_scale: Hệ số bán kính bubble ở chế độ tròn.
        circle_border_exclude_ratio: Tỉ lệ bỏ viền ngoài bubble.
        hough_only_ambiguous: Nếu bật, chỉ chạy Hough cho ô gần ngưỡng phân loại.
        hough_ambiguity_margin: Biên quanh ngưỡng để coi là ô mơ hồ cần Hough.
        hough_max_cells: Số ô tối đa được phép chạy Hough trong một lần gọi.

    Returns:
        Danh sách dictionary kết quả cho từng ô (box, row, col, ratio, filled, ...).
    """
    evaluations: List[Dict[str, object]] = []
    mode = str(mask_mode).lower()
    # Chế độ circle/hough-circle dùng mask tròn để giảm nhiễu ở các góc của ô.
    use_circle = mode in ("circle", "hough-circle")
    use_hough_circle = mode == "hough-circle"
    hough_calls_left = int(hough_max_cells) if int(hough_max_cells) > 0 else 0
    ambiguity_margin = float(max(0.0, hough_ambiguity_margin))

    for info in grid_info:
        if "region_quad" not in info or "grid_shape" not in info:
            continue

        region_quad = np.array(info["region_quad"], dtype=np.float32)
        rows, cols = info["grid_shape"]
        rows = int(rows)
        cols = int(cols)
        pattern = info.get("pattern")

        for row in range(rows):
            # Một số phần (đặc biệt Part III) có pattern cột đặc thù theo từng hàng.
            if isinstance(pattern, list) and row < len(pattern):
                cols_to_check = [int(c) for c in pattern[row] if 0 <= int(c) < cols]
            else:
                cols_to_check = list(range(cols))

            for col in cols_to_check:
                cell_quad = _quad_cell_at(region_quad, row=row, col=col, rows=rows, cols=cols)
                inner_quad = _shrink_quad_towards_center(cell_quad, inner_margin_ratio)
                quad_ratio: Optional[float] = None
                if use_circle:
                    # Ưu tiên đo trong vùng tròn để khớp hình bubble thực tế.
                    if use_hough_circle:
                        # Tính nhanh trước bằng circle ước lượng; chỉ Hough khi ô gần ngưỡng.
                        base_ratio, base_poly, _ = _fill_ratio_in_circle(
                            binary_image,
                            inner_quad,
                            radius_scale=circle_radius_scale,
                            border_exclude_ratio=circle_border_exclude_ratio,
                            use_hough_detection=False,
                        )

                        # Đo thêm trên quad để phát hiện trường hợp circle estimate bị lệch tâm
                        # (dễ gây miss ô đã tô khi chỉ dựa vào base_ratio).
                        quad_ratio = _fill_ratio_in_quad(binary_image, inner_quad)

                        should_run_hough = True
                        if hough_only_ambiguous:
                            near_threshold = abs(float(base_ratio) - float(fill_ratio_thresh)) <= ambiguity_margin
                            likely_filled_from_quad = quad_ratio >= max(
                                float(fill_ratio_thresh) * 0.78,
                                float(fill_ratio_thresh) - 0.14,
                            )
                            circle_quad_disagree = (quad_ratio - float(base_ratio)) >= 0.14
                            weak_circle_signal = float(base_ratio) <= float(fill_ratio_thresh) * 0.65
                            should_run_hough = bool(
                                near_threshold
                                or (likely_filled_from_quad and (circle_quad_disagree or weak_circle_signal))
                            )
                        if hough_calls_left <= 0:
                            should_run_hough = False

                        if should_run_hough:
                            fill_ratio, score_poly, circle_found = _fill_ratio_in_circle(
                                binary_image,
                                inner_quad,
                                radius_scale=circle_radius_scale,
                                border_exclude_ratio=circle_border_exclude_ratio,
                                use_hough_detection=True,
                            )
                            hough_calls_left -= 1
                        else:
                            # Nếu hết budget Hough nhưng quad cho tín hiệu ô tô rõ hơn,
                            # hợp nhất nhẹ để giảm miss mà vẫn hạn chế false-positive.
                            fused_ratio = max(
                                float(base_ratio),
                                min(float(quad_ratio) * 0.85, float(base_ratio) + 0.12),
                            )
                            fill_ratio = fused_ratio
                            score_poly = base_poly
                            circle_found = False
                    else:
                        fill_ratio, score_poly, circle_found = _fill_ratio_in_circle(
                            binary_image,
                            inner_quad,
                            radius_scale=circle_radius_scale,
                            border_exclude_ratio=circle_border_exclude_ratio,
                            use_hough_detection=False,
                        )
                else:
                    # Fallback mask tứ giác khi không dùng chế độ circle/hough-circle.
                    fill_ratio = _fill_ratio_in_quad(binary_image, inner_quad)
                    score_poly = inner_quad
                    circle_found = False

                effective_ratio = float(fill_ratio)
                effective_thresh = float(fill_ratio_thresh)
                if use_hough_circle and quad_ratio is not None:
                    # Rescue nhẹ cho ô có tín hiệu tô rõ theo quad nhưng circle vẫn thấp.
                    if (
                        effective_ratio < effective_thresh
                        and not bool(circle_found)
                        and float(quad_ratio) >= effective_thresh * 1.02
                        and float(base_ratio) >= effective_thresh * 0.70
                    ):
                        effective_ratio = max(effective_ratio, float(quad_ratio) * 0.88)

                    # Khi circle chưa bắt được ổn định, nới ngưỡng một chút để giảm miss.
                    if not bool(circle_found):
                        effective_thresh = max(0.35, effective_thresh - 0.015)

                evaluations.append(
                    {
                        "box_idx": int(info.get("box_idx", -1)),
                        "row": row,
                        "col": col,
                        "fill_ratio": effective_ratio,
                        "filled": effective_ratio >= effective_thresh,
                        "cell_quad": score_poly,
                        "mask_mode": "hough-circle" if use_hough_circle else ("circle" if use_circle else "quad"),
                        "circle_detected": bool(circle_found),
                    }
                )

    return evaluations


def suppress_false_positive_by_relative_dominance(
    evaluations: List[Dict[str, object]],
    group_keys: Tuple[str, str] = ("box_idx", "row"),
    ratio_key: str = "fill_ratio",
    min_dominant_ratio: float = 0.52,
    min_gap: float = 0.12,
    min_ratio_scale: float = 1.35,
) -> List[Dict[str, object]]:
    # Hậu xử lý theo tương quan trong cùng câu: giữ ô vượt trội, loại các ô nhiễu còn lại.
    if not evaluations:
        return evaluations

    group_keys = tuple(group_keys)
    if len(group_keys) < 2:
        return evaluations

    refined = [dict(item) for item in evaluations]
    grouped_indices: Dict[Tuple[int, int], List[int]] = {}

    for idx, item in enumerate(refined):
        key_vals: List[int] = []
        valid_key = True
        for key in group_keys:
            value = item.get(key, None)
            if value is None:
                valid_key = False
                break
            key_vals.append(int(value))
        if not valid_key:
            continue
        grouped_indices.setdefault((key_vals[0], key_vals[1]), []).append(idx)

    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue

        scored = sorted(
            indices,
            key=lambda i: float(refined[i].get(ratio_key, 0.0)),
            reverse=True,
        )
        best_idx = scored[0]
        second_idx = scored[1]
        best_ratio = float(refined[best_idx].get(ratio_key, 0.0))
        second_ratio = float(refined[second_idx].get(ratio_key, 0.0))

        has_strong_abs = best_ratio >= float(min_dominant_ratio)
        has_strong_gap = (best_ratio - second_ratio) >= float(min_gap)
        has_scale_gap = best_ratio >= float(max(0.0, second_ratio)) * float(min_ratio_scale)
        is_dominant = bool(has_strong_abs and has_strong_gap and has_scale_gap)

        if not is_dominant or not bool(refined[best_idx].get("filled", False)):
            continue

        refined[best_idx]["dominant_choice"] = True
        for idx in scored[1:]:
            if bool(refined[idx].get("filled", False)):
                refined[idx]["filled"] = False
                refined[idx]["suppressed_by_dominance"] = True

    return refined


def draw_filled_cells_overlay(
    image: np.ndarray,
    evaluations: List[Dict[str, object]],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    # Vẽ lớp hiển thị phục vụ debug hoặc trực quan kết quả.
    """
    Vẽ lớp phủ trong suốt cho các ô được xác định là đã tô.

    Args:
        image: Ảnh gốc để vẽ.
        evaluations: Danh sách kết quả chấm từng ô.
        color: Màu phủ cho ô tô.
        alpha: Độ trong suốt lớp phủ.

    Returns:
        Ảnh đã thêm overlay vùng ô tô.
    """
    if image is None or image.size == 0 or not evaluations:
        return image

    overlay = image.copy()
    for item in evaluations:
        if not bool(item.get("filled", False)):
            continue
        # Dùng đúng cell_quad đã chấm để overlay khớp chính xác với vùng score.
        quad = np.round(np.array(item["cell_quad"], dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(overlay, quad, color)

    out = image.copy()
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0, out)
    return out


def draw_binary_fillratio_debug(
    binary_image: np.ndarray,
    evaluations: List[Dict[str, object]],
    out_path: str,
) -> None:
    # Vẽ lớp hiển thị phục vụ debug hoặc trực quan kết quả.
    """
    Lưu ảnh debug gồm nền nhị phân và nhãn fill-ratio cho từng ô.

    Args:
        binary_image: Ảnh nhị phân đã dùng để chấm.
        evaluations: Danh sách kết quả chấm fill-ratio.
        out_path: Đường dẫn ảnh debug đầu ra.

    Returns:
        Không trả về giá trị.
    """
    if binary_image is None or binary_image.size == 0:
        return

    if binary_image.ndim == 2:
        canvas = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = binary_image.copy()

    for item in evaluations:
        quad = np.round(np.array(item["cell_quad"], dtype=np.float32)).astype(np.int32)
        ratio = float(item.get("fill_ratio", 0.0))
        filled = bool(item.get("filled", False))
        color = (0, 255, 0) if filled else (0, 165, 255)

        cv2.polylines(canvas, [quad], True, color, 1)

        cxy = np.mean(quad, axis=0)
        tx = int(cxy[0]) - 10
        ty = int(cxy[1]) + 4
        cv2.putText(
            canvas,
            f"{ratio:.2f}",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, canvas)


def _print_fill_summary(title: str, evaluations: List[Dict[str, object]], limit: int = 40) -> None:
    # In tóm tắt nhanh để theo dõi chất lượng xử lý khi chạy.
    """
    In tóm tắt ngắn các ô đã tô cho một section lưới.

    Args:
        title: Tên section để hiển thị log.
        evaluations: Danh sách kết quả chấm của section.
        limit: Số dòng chi tiết tối đa cần in.

    Returns:
        Không trả về giá trị.
    """
    total = len(evaluations)
    filled_items = [e for e in evaluations if bool(e.get("filled", False))]
    hough_items = [e for e in evaluations if str(e.get("mask_mode", "")) == "hough-circle"]
    if hough_items:
        detected = sum(1 for e in hough_items if bool(e.get("circle_detected", False)))
        print(f"{title}: filled {len(filled_items)}/{total}, hough circles {detected}/{len(hough_items)}")
    else:
        print(f"{title}: filled {len(filled_items)}/{total}")
    for idx, item in enumerate(filled_items):
        if idx >= limit:
            print(f"  ... and {len(filled_items) - limit} more")
            break
        print(
            f"  box={item['box_idx']} r={item['row']} c={item['col']} "
            f"ratio={item['fill_ratio']:.3f}"
        )


def _mean_darkness_in_box_circle(
    gray_image: np.ndarray,
    box: np.ndarray,
    radius_scale: float = 0.36,
) -> float:
    # Hàm phụ tính đặc trưng trung bình để phục vụ suy luận.
    """
    Tính độ tối trung bình (mức xám) trong vòng tròn trung tâm của box.

    Args:
        gray_image: Ảnh xám đầu vào.
        box: Contour box bubble.
        radius_scale: Hệ số bán kính vùng đo.

    Returns:
        Giá trị mean-darkness (0 là tối nhất, 255 là sáng nhất).
    """
    if gray_image is None or gray_image.size == 0:
        return 255.0

    x, y, w, h = cv2.boundingRect(box)
    if w <= 0 or h <= 0:
        return 255.0

    cx = x + (w // 2)
    cy = y + (h // 2)
    # Bán kính đo darkness nhỏ hơn bán kính bubble một chút để bớt dính viền.
    rr = max(2, int(round(min(w, h) * float(np.clip(radius_scale, 0.15, 0.48)))))

    x1 = max(0, cx - rr)
    y1 = max(0, cy - rr)
    x2 = min(gray_image.shape[1], cx + rr + 1)
    y2 = min(gray_image.shape[0], cy + rr + 1)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return 255.0

    mask = np.zeros(roi.shape, dtype=np.uint8)
    local_center = (cx - x1, cy - y1)
    # Chặn bán kính theo ROI để không vượt biên trong trường hợp box sát mép ảnh.
    local_radius = max(1, min(rr, min(roi.shape[0], roi.shape[1]) // 2))
    cv2.circle(mask, local_center, local_radius, 255, -1)

    pixels = roi[mask > 0]
    if pixels.size == 0:
        return 255.0
    return float(np.mean(pixels))


def evaluate_digit_rows_mean_darkness(
    gray_image: np.ndarray,
    aligned_rows: List[Optional[List[np.ndarray]]],
    expected_cols: int,
    radius_scale: float = 0.36,
    abs_darkness_threshold: float = 180.0,
    min_second_gap: float = 12.0,
    min_median_gap: float = 18.0,
    min_abs_median_gap: float = 8.0,
    min_abs_second_gap: float = 12.0,
) -> Dict[str, object]:
    # Đánh giá và trả về chỉ số/kết quả quyết định cho bước này.
    """
    Chọn một bubble tô cho mỗi cột dựa trên giá trị mean-darkness nhỏ nhất theo hàng.

    Quy trình:
    1. Với mỗi cột, tính mean-darkness cho tất cả hàng hợp lệ.
    2. Lấy hàng tối nhất làm ứng viên.
    3. Kiểm tra độ cách biệt với ứng viên thứ hai/median để giảm nhầm lẫn.
    4. Trả về chuỗi giải mã và metadata quyết định theo từng cột.

    Args:
        gray_image: Ảnh xám đầu vào.
        aligned_rows: Danh sách hàng box đã căn chỉnh theo chỉ số hàng.
        expected_cols: Số cột cần giải mã.
        radius_scale: Hệ số bán kính vùng đo darkness.
        abs_darkness_threshold: Ngưỡng darkness tuyệt đối.
        min_second_gap: Chênh lệch tối thiểu so với lựa chọn thứ hai.
        min_median_gap: Chênh lệch tối thiểu so với median của cột.
        min_abs_median_gap: Ngưỡng median-gap khi dùng điều kiện tuyệt đối.
        min_abs_second_gap: Ngưỡng second-gap khi dùng điều kiện tuyệt đối.

    Returns:
        Dictionary gồm chuỗi decode, danh sách đánh giá, và quyết định theo cột.
    """
    evaluations: List[Dict[str, object]] = []
    decoded_chars: List[str] = []
    column_decisions: List[Dict[str, object]] = []

    if expected_cols <= 0:
        return {
            "decoded": "",
            "evaluations": evaluations,
        }

    for col in range(expected_cols):
        # So sánh theo cột: mỗi cột chỉ được chọn tối đa 1 hàng (1 chữ số).
        col_items: List[Dict[str, object]] = []
        for row_idx, row in enumerate(aligned_rows):
            if row is None or col >= len(row):
                col_items.append(
                    {
                        "row": row_idx,
                        "col": col,
                        "mean_darkness": 255.0,
                        "filled": False,
                        "box": None,
                        "valid": False,
                    }
                )
                continue

            box = row[col]
            darkness = _mean_darkness_in_box_circle(gray_image, box, radius_scale=radius_scale)
            col_items.append(
                {
                    "row": row_idx,
                    "col": col,
                    "mean_darkness": darkness,
                    "filled": False,
                    "box": box,
                    "valid": True,
                }
            )

        valid_items = [it for it in col_items if it["valid"]]
        if valid_items:
            # Hàng có darkness thấp nhất là ứng viên được tô đậm nhất.
            sorted_items = sorted(valid_items, key=lambda it: float(it["mean_darkness"]))
            best_item = sorted_items[0]
            best_dark = float(best_item["mean_darkness"])
            second_dark = float(sorted_items[1]["mean_darkness"]) if len(sorted_items) > 1 else 255.0
            median_dark = float(np.median([float(it["mean_darkness"]) for it in sorted_items]))

            has_abs_dark = best_dark <= float(abs_darkness_threshold)
            second_gap = second_dark - best_dark
            median_gap = median_dark - best_dark
            # Cần có khoảng cách đủ lớn so với phần còn lại để tránh nhận nhầm cột nhiễu.
            has_gap = second_gap >= float(min_second_gap) and median_gap >= float(min_median_gap)
            has_context_for_abs = (
                median_gap >= float(min_abs_median_gap)
                and second_gap >= float(min_abs_second_gap)
            )
            # Tránh gán filled khi cả cột gần như đồng đều (quá sáng hoặc quá tối đồng loạt).
            col_filled = bool(has_gap or (has_abs_dark and has_context_for_abs))

            if col_filled:
                best_item["filled"] = True
                decoded_chars.append(str(int(best_item["row"])))
            else:
                decoded_chars.append("?")

            column_decisions.append(
                {
                    "col": col,
                    "filled": col_filled,
                    "best_row": int(best_item["row"]),
                    "best_darkness": best_dark,
                    "second_darkness": second_dark,
                    "median_darkness": median_dark,
                    "second_gap": second_gap,
                    "median_gap": median_gap,
                }
            )
        else:
            decoded_chars.append("?")
            column_decisions.append(
                {
                    "col": col,
                    "filled": False,
                    "best_row": None,
                    "best_darkness": 255.0,
                    "second_darkness": 255.0,
                    "median_darkness": 255.0,
                    "second_gap": 0.0,
                    "median_gap": 0.0,
                }
            )

        evaluations.extend(col_items)

    return {
        "decoded": "".join(decoded_chars),
        "evaluations": evaluations,
        "column_decisions": column_decisions,
    }


def _count_filled_digit_columns(result: Dict[str, object]) -> int:
    # Hàm hỗ trợ đếm nhanh số cột đã giải mã hợp lệ.
    decisions = result.get("column_decisions", [])
    if not isinstance(decisions, list):
        return 0
    return sum(1 for item in decisions if bool(item.get("filled", False)))


def evaluate_digit_rows_with_binary_fallback(
    gray_image: np.ndarray,
    aligned_rows: List[Optional[List[np.ndarray]]],
    expected_cols: int,
    binary_image: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    # Decode chính bằng ảnh xám; fallback sang binary nếu thiếu quá nhiều cột hợp lệ.
    primary = evaluate_digit_rows_mean_darkness(
        gray_image,
        aligned_rows,
        expected_cols=expected_cols,
    )
    primary_filled = _count_filled_digit_columns(primary)

    selected = dict(primary)
    selected["decode_source"] = "gray"
    selected["gray_filled_cols"] = int(primary_filled)
    selected["binary_filled_cols"] = 0

    if binary_image is None or not isinstance(binary_image, np.ndarray) or binary_image.size == 0:
        return selected

    binary_gray = binary_image
    if binary_gray.ndim == 3:
        binary_gray = cv2.cvtColor(binary_gray, cv2.COLOR_BGR2GRAY)

    # `binary` trong pipeline được tạo bằng THRESH_BINARY_INV:
    # vùng tô đậm là pixel trắng (255). Trong khi decoder darkness chọn giá trị thấp hơn.
    # Vì vậy cần đảo cực tính để bubble tô đậm trở thành "tối" như ảnh xám.
    binary_for_darkness = cv2.bitwise_not(binary_gray)

    binary_result = evaluate_digit_rows_mean_darkness(
        binary_for_darkness,
        aligned_rows,
        expected_cols=expected_cols,
    )
    binary_filled = _count_filled_digit_columns(binary_result)

    # Chỉ đổi sang binary khi có cải thiện thực sự để tránh ảnh hưởng mẫu vốn đang tốt.
    use_binary = binary_filled > primary_filled
    if use_binary:
        selected = dict(binary_result)
        selected["decode_source"] = "binary"
        selected["gray_filled_cols"] = int(primary_filled)
        selected["binary_filled_cols"] = int(binary_filled)
    else:
        selected["binary_filled_cols"] = int(binary_filled)

    return selected


def _print_digit_darkness_summary(title: str, result: Dict[str, object], limit: int = 20) -> None:
    # In tóm tắt nhanh để theo dõi chất lượng xử lý khi chạy.
    """
    In tóm tắt ngắn kết quả giải mã chữ số theo mean-darkness.

    Args:
        title: Tên section (SoBaoDanh/MaDe).
        result: Kết quả trả về từ hàm giải mã darkness.
        limit: Số dòng chi tiết tối đa cần in.

    Returns:
        Không trả về giá trị.
    """
    decoded = str(result.get("decoded", ""))
    evals = [e for e in result.get("evaluations", []) if bool(e.get("filled", False))]
    decisions = result.get("column_decisions", [])
    if isinstance(decisions, list):
        filled_cols = sum(1 for d in decisions if bool(d.get("filled", False)))
        print(f"{title} filled columns: {filled_cols}/{len(decisions)}")
    print(f"{title} decoded: {decoded}")
    if isinstance(decisions, list) and decisions:
        filled_decisions = [d for d in decisions if bool(d.get("filled", False))]
        for idx, item in enumerate(filled_decisions):
            if idx >= limit:
                print(f"  ... and {len(filled_decisions) - limit} more")
                break
            print(
                f"  col={int(item.get('col', -1))}"
                f" digit={int(item.get('best_row', -1))}"
                f" darkness={float(item.get('best_darkness', 255.0)):.1f}"
                f" second_gap={float(item.get('second_gap', 0.0)):.1f}"
                f" median_gap={float(item.get('median_gap', 0.0)):.1f}"
            )
    else:
        for idx, item in enumerate(evals):
            if idx >= limit:
                print(f"  ... and {len(evals) - limit} more")
                break
            print(
                f"  col={item['col']} digit={item['row']} darkness={item['mean_darkness']:.1f}"
            )


def draw_digit_darkness_overlay(
    image: np.ndarray,
    result: Dict[str, object],
    color: Tuple[int, int, int],
    alpha: float = 0.40,
) -> np.ndarray:
    # Vẽ lớp hiển thị phục vụ debug hoặc trực quan kết quả.
    """
    Vẽ vòng tròn trong suốt cho các bubble được chọn bởi bộ giải mã mean-darkness.

    Args:
        image: Ảnh gốc để vẽ.
        result: Kết quả giải mã mean-darkness.
        color: Màu phủ vòng tròn bubble được chọn.
        alpha: Độ trong suốt lớp phủ.

    Returns:
        Ảnh đã thêm overlay cho các bubble được chọn.
    """
    if image is None or image.size == 0:
        return image

    evaluations = result.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        return image

    overlay = image.copy()
    for item in evaluations:
        if not bool(item.get("filled", False)):
            continue
        box = item.get("box")
        if box is None:
            continue

        x, y, w, h = cv2.boundingRect(np.array(box))
        cx = x + (w // 2)
        cy = y + (h // 2)
        rr = max(2, int(round(min(w, h) * 0.36)))
        cv2.circle(overlay, (cx, cy), rr, color, -1)
        cv2.circle(overlay, (cx, cy), rr, tuple(max(0, c - 80) for c in color), 1)

    out = image.copy()
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0, out)
    return out


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
    # Trích xuất vùng/lưới theo cấu hình hình học của phần tương ứng.
    """
    Vẽ lưới cho từng box với pattern cột tùy biến theo từng hàng.

    Quy trình:
    1. Chuẩn hóa contour box về tứ giác ổn định.
    2. Co vùng làm việc theo start/end offset để tránh dính viền.
    3. Nếu có `row_col_patterns`, chỉ vẽ những ô thuộc cột được chỉ định ở mỗi hàng.
    4. Thu thập `grid_info` để dùng lại cho bước chấm fill-ratio.

    Args:
        image: Ảnh đầu vào (hàm sẽ vẽ trực tiếp lên bản sao của ảnh này).
        boxes: Danh sách box dạng contour polygon (numpy array).
        grid_cols: Số cột của lưới (mặc định 4).
        grid_rows: Số hàng của lưới (mặc định 12).
        start_offset_ratio_x: Tỉ lệ lùi từ biên trái theo trục X (mặc định 0.2 = 20%).
        start_offset_ratio_y: Tỉ lệ lùi từ biên trên theo trục Y (mặc định 0.1 = 10%).
        end_offset_ratio_x: Tỉ lệ lùi từ biên phải theo trục X (mặc định 0.0).
        end_offset_ratio_y: Tỉ lệ lùi từ biên dưới theo trục Y (mặc định 0.0).
        grid_color: Màu đường lưới theo định dạng BGR (mặc định xanh lá).
        grid_thickness: Độ dày đường lưới (mặc định 1).
        row_col_patterns:
            Danh sách pattern cột cho từng hàng.
            Ví dụ: [[0], [1, 2], [0, 1, 2, 3], ...].
            Nếu None thì vẽ đủ tất cả cột ở mọi hàng.

    Returns:
        Dictionary gồm:
        - 'image_with_grid': Ảnh đã vẽ lưới.
        - 'grid_info': Danh sách metadata của từng box/lưới để chấm đáp án.
    """
    # Tạo bản sao để không làm thay đổi ảnh gốc truyền vào.
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
    # Trích xuất vùng/lưới theo cấu hình hình học của phần tương ứng.
    """
    Vẽ lưới cho từng box với offset riêng theo từng box.

    Với mỗi box:
    1. Lấy offset bắt đầu từ `start_offset_ratios[box_idx]` hoặc dùng mặc định (0.2, 0.1).
    2. Suy ra vùng làm việc từ offset đầu/cuối theo trục X, Y.
    3. Vẽ lưới `grid_cols x grid_rows` theo phối cảnh của tứ giác box.
    4. Lưu metadata lưới để phục vụ bước đánh giá ô tô.

    Args:
        image: Ảnh đầu vào (hàm sẽ vẽ lên bản sao của ảnh này).
        boxes: Danh sách box dạng contour polygon (numpy array).
        grid_cols: Số cột lưới (mặc định 4).
        grid_rows: Số hàng lưới (mặc định 10).
        start_offset_ratios:
            Danh sách tuple `(offset_x, offset_y)` cho từng box.
            Nếu thiếu hoặc None thì dùng mặc định `(0.2, 0.1)`.
        end_offset_ratios_x:
            Danh sách tỉ lệ offset cuối theo trục X (tính từ biên phải).
            Nếu thiếu hoặc None thì dùng `0.0`.
        end_offset_ratios_y:
            Danh sách tỉ lệ offset cuối theo trục Y (tính từ biên dưới).
            Nếu thiếu hoặc None thì dùng `0.0`.
        grid_color: Màu đường lưới theo định dạng BGR.
        grid_thickness: Độ dày đường lưới.

    Returns:
        Dictionary gồm:
        - 'image_with_grid': Ảnh đã vẽ lưới.
        - 'grid_info': Danh sách metadata của từng box/lưới.
    """
    # Tạo bản sao để không làm thay đổi ảnh gốc truyền vào.
    output_image = image.copy()
    grid_info = []
    
    default_offset = (0.2, 0.1)
    
    for box_idx, box in enumerate(boxes):
        # Lấy offset đầu cho box hiện tại.
        if start_offset_ratios and box_idx < len(start_offset_ratios):
            offset_x, offset_y = start_offset_ratios[box_idx]
        else:
            offset_x, offset_y = default_offset
        
        # Lấy offset cuối theo trục X, Y nếu có cung cấp riêng.
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
    # Trích xuất vùng/lưới theo cấu hình hình học của phần tương ứng.
    """
    Vẽ lưới chuẩn cho từng box trên ảnh.

    Hàm này dùng một bộ offset chung cho tất cả box.
    Lưới được vẽ theo phối cảnh của contour (không giả định hình chữ nhật trục chuẩn).

    Args:
        image: Ảnh đầu vào (hàm sẽ vẽ lên bản sao của ảnh này).
        boxes: Danh sách box dạng contour polygon.
        grid_cols: Số cột lưới (mặc định 4).
        grid_rows: Số hàng lưới (mặc định 10).
        start_offset_ratio_x: Tỉ lệ lùi đầu theo trục X (mặc định 0.2).
        start_offset_ratio_y: Tỉ lệ lùi đầu theo trục Y (mặc định 0.1).
        end_offset_ratio_x: Tỉ lệ lùi cuối theo trục X từ biên phải (mặc định 0.0).
        end_offset_ratio_y: Tỉ lệ lùi cuối theo trục Y từ biên dưới (mặc định 0.0).
        grid_color: Màu đường lưới (BGR).
        grid_thickness: Độ dày đường lưới.

    Returns:
        Dictionary gồm:
        - 'image_with_grid': Ảnh đã vẽ lưới.
        - 'grid_info': Metadata của từng vùng lưới để phục vụ chấm.
    """
    # Tạo bản sao để giữ nguyên ảnh đầu vào.
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
    # Hàm phụ lọc nhiễu hoặc loại bỏ phần tử không đạt điều kiện.
    """
    Chỉ giữ các thành phần đường liên thông có trục chính dài hơn ngưỡng.

    Args:
        line_mask: Ảnh nhị phân chứa các line đã tách.
        min_length: Độ dài tối thiểu của thành phần cần giữ.
        orientation: Hướng line cần xét (`vertical` hoặc `horizontal`).

    Returns:
        Ảnh mask sau lọc thành phần ngắn.
    """
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
    # Hàm phụ căn chỉnh dữ liệu để tăng tính ổn định đầu ra.
    """
    Kéo dài các line dọc trong cùng hàng về chiều dài tương đương để tăng ổn định.

    Args:
        vertical_mask: Ảnh mask line dọc.
        row_tolerance: Ngưỡng lệch tâm Y để gom cùng một hàng line.
        min_group_size: Số line tối thiểu để thực hiện căn chỉnh.

    Returns:
        Ảnh mask line dọc sau khi căn chỉnh chiều dài.
    """
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
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện giao điểm lưới từ các đường dọc/ngang thu được bằng morphology.

    Quy trình:
    1. Nhị phân hóa ảnh bằng adaptive threshold.
    2. Tách line dọc và line ngang bằng kernel morphology.
    3. Lấy giao điểm giữa hai mask line.
    4. Lọc connected components nhỏ để giữ các giao điểm hợp lệ.

    Args:
        image: Ảnh đầu vào.
        vertical_scale: Tỉ lệ chiều dài kernel dọc theo chiều cao ảnh.
        horizontal_scale: Tỉ lệ chiều dài kernel ngang theo chiều rộng ảnh.
        min_point_area: Diện tích tối thiểu của giao điểm.
        block_size: Kích thước block cho adaptive threshold.
        block_offset: Hệ số C của adaptive threshold.
        debug_prefix: Tiền tố lưu ảnh debug (nếu cần).

    Returns:
        Dictionary chứa các mask trung gian, overlay giao điểm và danh sách tọa độ điểm.
    """
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

def _split_merged_boxes_for_grouping(
    boxes: List[np.ndarray],
    split_wide: bool = False,
    split_tall: bool = False,
    min_area: int = 400,
    max_area: int = 10000,
) -> List[np.ndarray]:
    # Hàm phụ tách phần tử gộp để phục hồi cấu trúc mong muốn.
    """
    Tách các box bubble có khả năng bị dính để nhóm hàng/cột chính xác hơn.

    Args:
        boxes: Danh sách contour box đầu vào.
        split_wide: Bật tách box dính theo chiều ngang.
        split_tall: Bật tách box dính theo chiều dọc.
        min_area: Diện tích nhỏ nhất của box xét tách.
        max_area: Diện tích lớn nhất của box xét tách.

    Returns:
        Danh sách box sau khi tách các contour dính.
    """
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
    # Hàm phụ chia dữ liệu thành các nhánh xử lý độc lập.
    """
    Tách box vùng ID phía trên thành SoBaoDanh (trái) và Mã đề (phải).

    Args:
        boxes: Danh sách box ứng viên vùng phía trên.
        part_i_boxes: Danh sách box phần Part I để suy ra ngưỡng Y phía trên.
        top_margin: Biên an toàn tính từ đỉnh Part I.
        min_area: Diện tích tối thiểu cho box ID.
        max_area: Diện tích tối đa cho box ID.
        row_tolerance: Ngưỡng gom các box cùng hàng theo Y.

    Returns:
        Tuple `(sbd_boxes, ma_de_boxes, split_x)`.
    """
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
    # Gom nhóm dữ liệu để tạo cấu trúc phục vụ xử lý phía sau.
    """
    Gom nhóm box phát hiện được thành ba phần chính của phiếu: Part I, Part II, Part III.

    Chiến lược:
    1. Lọc các box ứng viên theo diện tích và loại trùng bằng IoU.
    2. Gom box theo hàng theo trục Y.
    3. Nhận diện lần lượt Part I (4), Part II (8), Part III (6) theo đặc trưng hình học.
    4. Áp dụng nhiều nhánh fallback để phục hồi khi scan thiếu/méo contour.

    Args:
        boxes: Danh sách contour box phát hiện từ bước morphology.
        row_tolerance: Dung sai gom nhóm theo hàng.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong một nhóm.
        min_boxes_per_group: Số box tối thiểu để coi là một nhóm hợp lệ.

    Returns:
        Dictionary gồm `part_i`, `part_ii`, `part_iii`, và `all_parts`.
    """
    if not boxes:
        return {"part_i": [], "part_ii": [], "part_iii": [], "all_parts": []}

    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
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
        # Hàm phụ chuyển đổi biểu diễn hình học giữa các dạng.
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
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        areas = [int(b["area"]) for b in group]
        if not areas:
            return False
        mean_area = float(np.mean(areas))
        if mean_area <= 0:
            return True
        max_rel = max(abs(a - mean_area) / mean_area for a in areas)
        return max_rel <= tol

    def _select_best_subset(group: List[Dict[str, object]], expected_count: int) -> List[Dict[str, object]]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        if len(group) < expected_count:
            return []
        sorted_group = sorted(group, key=lambda b: int(b["x"]))
        if len(sorted_group) == expected_count:
            return sorted_group

        best_subset: List[Dict[str, object]] = []
        best_score = float("inf")

        def _score_subset(subset: List[Dict[str, object]]) -> float:
            # Chấm subset theo độ đồng đều kích thước và khoảng cách cột.
            areas = [float(b["area"]) for b in subset]
            mean_area = float(np.mean(areas))
            if mean_area <= 0:
                return float("inf")
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
            return float(size_var + 0.25 * gap_var)

        n = len(sorted_group)
        # Với nhóm quá lớn, tránh brute-force tổ hợp toàn phần (rất chậm).
        if n > 14:
            # 1) Quét cửa sổ liên tiếp chuẩn.
            for start in range(0, n - expected_count + 1):
                subset = sorted_group[start : start + expected_count]
                score = _score_subset(subset)
                if score < best_score:
                    best_score = score
                    best_subset = subset

            # 2) Quét cửa sổ lớn hơn 1 phần tử để cho phép bỏ 1 outlier.
            window_size = expected_count + 1
            if n >= window_size:
                for start in range(0, n - window_size + 1):
                    window = sorted_group[start : start + window_size]
                    for drop_idx in range(window_size):
                        subset = window[:drop_idx] + window[drop_idx + 1 :]
                        score = _score_subset(subset)
                        if score < best_score:
                            best_score = score
                            best_subset = subset
            return best_subset

        # Nhóm nhỏ vẫn dùng exhaustive search để giữ chất lượng chọn subset.
        from itertools import combinations
        for idx_tuple in combinations(range(n), expected_count):
            subset = [sorted_group[idx] for idx in idx_tuple]
            score = _score_subset(subset)

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
        # Hàm phụ gom nhóm dữ liệu theo quy tắc hình học.
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
    # Hàm phụ chuyển đổi biểu diễn hình học giữa các dạng.
    """
    Chuyển bounding rectangle sang contour polygon 4 đỉnh.

    Args:
        x: Tọa độ trái.
        y: Tọa độ trên.
        w: Chiều rộng.
        h: Chiều cao.

    Returns:
        Contour đa giác hình chữ nhật dạng OpenCV.
    """
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
    # Hàm phụ dựng cấu trúc dữ liệu trung gian dùng lại nhiều nơi.
    """
    Dựng metadata cơ bản cho từng box để thuận tiện cho các bước gom nhóm.

    Args:
        boxes: Danh sách contour box.

    Returns:
        Danh sách dictionary chứa vị trí, kích thước, tâm Y và diện tích mỗi box.
    """
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
    # Hàm phụ gom nhóm dữ liệu theo quy tắc hình học.
    """
    Gom danh sách box-info thành các hàng dựa trên độ gần nhau theo trục Y.

    Args:
        box_info: Danh sách metadata box.
        row_tolerance: Ngưỡng lệch theo Y để coi cùng một hàng.

    Returns:
        Danh sách nhóm hàng, mỗi phần tử là một list box-info.
    """
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
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Kiểm tra một nhóm box có đồng đều kích thước theo diện tích hay không.

    Args:
        group: Nhóm box-info cần kiểm tra.
        size_tolerance_ratio: Ngưỡng lệch tương đối tối đa cho phép.

    Returns:
        `True` nếu nhóm có độ đồng đều đạt yêu cầu.
    """
    areas = [b["area"] for b in group]
    mean_area = np.mean(areas)
    return max([abs(a - mean_area) / mean_area for a in areas]) <= size_tolerance_ratio if mean_area > 0 else True


def _filter_rows_by_global_size_consistency(
    rows: List[List[np.ndarray]],
    size_tolerance_ratio: float,
    debug: bool = False,
) -> List[List[np.ndarray]]:
    # Hàm phụ lọc nhiễu hoặc loại bỏ phần tử không đạt điều kiện.
    """
    Lọc các hàng có kích thước trung bình lệch quá xa so với mặt bằng chung.

    Args:
        rows: Danh sách hàng box.
        size_tolerance_ratio: Ngưỡng lệch tương đối cho phép.
        debug: Bật log chi tiết khi lọc.

    Returns:
        Danh sách hàng sau lọc nhiễu kích thước.
    """
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
    # Hàm phụ cắt/chọn cửa sổ dữ liệu ổn định nhất.
    """
    Chọn cửa sổ liên tiếp có hình học ổn định nhất khi số hàng vượt giới hạn.

    Args:
        rows: Danh sách hàng box đầu vào.
        max_rows: Số hàng tối đa cần giữ.

    Returns:
        Danh sách hàng đã được cắt về cửa sổ ổn định nhất.
    """
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
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện và gom nhóm các box của vùng Số báo danh.

    Đặc trưng vùng SoBaoDanh:
    - Mỗi hàng có 6 box.
    - Tối đa 10 hàng liên tiếp.
    - Kích thước box trong cùng một hàng tương đối đồng đều
      (cho phép dung sai để chịu được ảnh scan chất lượng khác nhau).

    Args:
        boxes: Danh sách box ứng viên vùng Số báo danh.
        boxes_per_row: Số box kỳ vọng trên mỗi hàng.
        max_rows: Số hàng tối đa cần giữ.
        row_tolerance: Dung sai gom nhóm theo trục Y.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong hàng.
        debug: Bật/tắt log debug.

    Returns:
        Dictionary gồm:
        - 'sobao_danh': Danh sách toàn bộ box SoBaoDanh đã phát hiện.
        - 'sobao_danh_rows': Danh sách các hàng, mỗi hàng chứa 6 box.
        - 'row_count': Số hàng đã phát hiện.
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
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        return _is_uniform_size_group(group, size_tolerance_ratio)

    def has_excessive_x_overlap(
        row_boxes: List[np.ndarray],
        max_overlap_pairs: int = 1,
        overlap_tol_px: int = 1,
    ) -> bool:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Một hàng SBD hợp lệ gần như không có box chồng nhau theo trục X.
        rects = sorted([cv2.boundingRect(b) for b in row_boxes], key=lambda r: r[0])
        overlap_pairs = 0
        for i in range(len(rects) - 1):
            x0, y0, w0, h0 = rects[i]
            x1, y1, w1, h1 = rects[i + 1]
            if x0 + w0 > x1 + overlap_tol_px:
                overlap_pairs += 1
                if overlap_pairs > max_overlap_pairs:
                    return True
        return False

    def _try_recover_merged_row(group: List[Dict[str, object]]) -> Optional[List[np.ndarray]]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Recovery cho trường hợp một hoặc nhiều contour bị dính theo chiều ngang,
        # khiến hàng 6 ô chỉ còn 5 hoặc 4 box.
        missing_count = boxes_per_row - len(group)
        if missing_count <= 0 or missing_count > 2:
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

        splittable_indices: List[int] = []
        for idx_item, (ww, hh) in enumerate(zip(widths, heights)):
            # Chỉ tách contour đủ rộng và có chiều cao gần chuẩn để tránh tách nhầm nhiễu.
            if ww >= median_w * 1.45 and (0.70 * median_h <= hh <= 1.45 * median_h):
                splittable_indices.append(idx_item)

        if len(splittable_indices) < missing_count:
            return None

        split_indices = set(
            sorted(
                splittable_indices,
                key=lambda idx_item: widths[idx_item],
                reverse=True,
            )[:missing_count]
        )

        candidate_boxes: List[np.ndarray] = []
        for i, item in enumerate(sorted_group):
            if i in split_indices:
                merged_w = widths[i]
                merged_h = heights[i]
                mx = int(item["x"])
                my = int(item["y"])
                mh = int(item["h"]) if "h" in item else merged_h

                left_w = merged_w // 2
                right_w = merged_w - left_w
                if left_w <= 0 or right_w <= 0:
                    return None

                left_poly = _rect_to_poly(mx, my, left_w, mh)
                right_poly = _rect_to_poly(mx + left_w, my, right_w, mh)
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
    short_groups: List[List[Dict[str, object]]] = []
    
    for idx, group in enumerate(groups):
        # Only accept rows with exactly boxes_per_row boxes and uniform size
        if len(group) == boxes_per_row and is_uniform_size(group):
            sorted_boxes = [b["box"] for b in sorted(group, key=lambda b: b["x"])]
            if has_excessive_x_overlap(sorted_boxes):
                if debug:
                    print(f"  ✗ Group {idx} rejected: excessive X overlap in 6-box row")
            else:
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
                    if has_excessive_x_overlap(sorted_boxes):
                        if debug:
                            print(
                                f"  ✗ Group {idx} sub-row {start_idx} rejected: excessive X overlap"
                            )
                        continue
                    sobao_danh_rows.append(sorted_boxes)
                    # Mark these boxes as used
                    for i in range(start_idx, start_idx + boxes_per_row):
                        used_indices.add(i)
                    if debug:
                        print(f"  ✓ Group {idx} sub-row extracted as SoBaoDanh row {len(sobao_danh_rows)}")

            if debug and len(group) <= 15:
                print(f"  ✗ Group {idx} rejected: len={len(group)} (need 6), no valid non-overlapping 6-box sub-rows")
        elif boxes_per_row - 2 <= len(group) <= boxes_per_row - 1:
            short_groups.append(group)
            recovered = _try_recover_merged_row(group)
            if recovered is not None:
                sobao_danh_rows.append(recovered)
                if debug:
                    print(
                        f"  ✓ Group {idx} recovered as SoBaoDanh row {len(sobao_danh_rows)} "
                        f"(split merged box, len={len(group)})"
                    )
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
    sobao_danh_rows = [
        row for row in sobao_danh_rows
        if not has_excessive_x_overlap(row)
    ]
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    sobao_danh_rows = _trim_rows_to_consistent_window(sobao_danh_rows, max_rows)
    sobao_danh = [box for row in sobao_danh_rows for box in row]

    # Handle a common pattern: one unrelated header-like top row plus one valid
    # row detected thiếu box do contour bị dính.
    if len(sobao_danh_rows) == max_rows and short_groups:
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
                        f"median tail gap={median_tail_gap:.1f}); attempting replacement from short row"
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
                            # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
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
                        for g in short_groups:
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
                                print("  ✓ Replaced top outlier SBD row using short-row recovery")
                        else:
                            sobao_danh_rows = kept_rows
                            sobao_danh = [box for row in sobao_danh_rows for box in row]
                            if debug:
                                print("  • Dropped top outlier SBD row (no suitable short-row replacement)")

    # Normalize malformed top row where one or more boxes are abnormally wide/shifted.
    # This appears on some scans (e.g. 0003) where first-row middle boxes merge visually.
    if len(sobao_danh_rows) >= 2:
        row_with_y = []
        for row in sobao_danh_rows:
            ys = [cv2.boundingRect(box)[1] for box in row]
            row_with_y.append((float(np.mean(ys)) if ys else 0.0, row))
        row_with_y.sort(key=lambda t: t[0])

        top_row = row_with_y[0][1]
        ref_rows = [r for _, r in row_with_y[1:] if len(r) == boxes_per_row]

        if len(top_row) == boxes_per_row and ref_rows:
            top_rects = sorted([cv2.boundingRect(b) for b in top_row], key=lambda r: r[0])

            col_x_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            col_w_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            h_list: List[int] = []

            for row in ref_rows:
                rects = sorted([cv2.boundingRect(b) for b in row], key=lambda r: r[0])
                if len(rects) != boxes_per_row:
                    continue
                for c, (rx, ry, rw, rh) in enumerate(rects):
                    col_x_lists[c].append(int(rx))
                    col_w_lists[c].append(int(rw))
                    h_list.append(int(rh))

            if all(col_x_lists[c] for c in range(boxes_per_row)) and h_list:
                col_x = [int(round(float(np.median(col_x_lists[c])))) for c in range(boxes_per_row)]
                col_w = [max(1, int(round(float(np.median(col_w_lists[c]))))) for c in range(boxes_per_row)]
                row_h = max(1, int(round(float(np.median(h_list)))))

                wide_outliers = 0
                shifted_outliers = 0
                severe_small_outliers = 0
                overlap_count = 0
                for c, (rx, ry, rw, rh) in enumerate(top_rects):
                    width_tol_hi = col_w[c] * 1.35
                    width_tol_lo = col_w[c] * 0.65
                    if rw > width_tol_hi or rw < width_tol_lo:
                        wide_outliers += 1

                    # Bắt các contour bị co cụm thành chấm/vòng tròn nhỏ thay vì box bubble.
                    severe_small_width = rw < (col_w[c] * 0.58)
                    severe_small_height = rh < (row_h * 0.60)
                    severe_small_area = (rw * rh) < (col_w[c] * row_h * 0.45)
                    if severe_small_width or severe_small_height or severe_small_area:
                        severe_small_outliers += 1

                    shift_tol = max(6.0, col_w[c] * 0.35)
                    if abs(rx - col_x[c]) > shift_tol:
                        shifted_outliers += 1

                for c in range(boxes_per_row - 1):
                    x0, y0, w0, h0 = top_rects[c]
                    x1, y1, w1, h1 = top_rects[c + 1]
                    if x0 + w0 > x1:
                        overlap_count += 1

                top_row_is_malformed = (
                    (wide_outliers >= 1 and shifted_outliers >= 1)
                    or overlap_count >= 1
                    or severe_small_outliers >= 1
                )
                if top_row_is_malformed:
                    row_y = int(round(float(np.mean([r[1] for r in top_rects]))))
                    normalized_top = [
                        _rect_to_poly(col_x[c], row_y, col_w[c], row_h)
                        for c in range(boxes_per_row)
                    ]
                    row_with_y[0] = (row_with_y[0][0], normalized_top)
                    sobao_danh_rows = [row for _, row in row_with_y]
                    sobao_danh = [box for row in sobao_danh_rows for box in row]
                    if debug:
                        print(
                            "  ✓ Normalized top SBD row geometry "
                            f"(wide={wide_outliers}, shifted={shifted_outliers}, "
                            f"small={severe_small_outliers}, overlap={overlap_count})"
                        )

    # Fallback chuyên biệt: thiếu đúng 1 hàng đầu (thường do hàng đầu bị dính contour ngang).
    if len(sobao_danh_rows) == max_rows - 1 and len(sobao_danh_rows) >= 2:
        row_with_y = []
        for row in sobao_danh_rows:
            ys = [cv2.boundingRect(box)[1] for box in row]
            row_with_y.append((float(np.mean(ys)) if ys else 0.0, row))
        row_with_y.sort(key=lambda t: t[0])

        ys_only = [t[0] for t in row_with_y]
        gaps = [ys_only[i + 1] - ys_only[i] for i in range(len(ys_only) - 1)]
        median_gap = float(np.median(gaps)) if gaps else 0.0

        if median_gap > 1.0:
            expected_top_y = ys_only[0] - median_gap

            valid_rows = [row for _, row in row_with_y if len(row) == boxes_per_row]
            col_x_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            col_w_lists: List[List[int]] = [[] for _ in range(boxes_per_row)]
            h_list: List[int] = []

            for row in valid_rows:
                rects = sorted([cv2.boundingRect(b) for b in row], key=lambda r: r[0])
                if len(rects) != boxes_per_row:
                    continue
                for c, (rx, ry, rw, rh) in enumerate(rects):
                    col_x_lists[c].append(int(rx))
                    col_w_lists[c].append(int(rw))
                    h_list.append(int(rh))

            if all(col_x_lists[c] for c in range(boxes_per_row)) and h_list:
                col_x = [int(round(float(np.median(col_x_lists[c])))) for c in range(boxes_per_row)]
                col_w = [max(1, int(round(float(np.median(col_w_lists[c]))))) for c in range(boxes_per_row)]
                row_h = max(1, int(round(float(np.median(h_list)))) )

                best_short_group: Optional[List[Dict[str, object]]] = None
                best_dist = float("inf")
                top_band_limit = ys_only[0] + max(20.0, median_gap * 0.45)
                for group in short_groups:
                    if not group:
                        continue
                    gy = float(np.mean([int(it["y"]) for it in group]))
                    if gy > top_band_limit:
                        continue
                    dist = abs(gy - expected_top_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_short_group = group

                recover_tol = max(45.0, median_gap * 1.15)
                if best_short_group is not None and best_dist <= recover_tol:
                    assigned: List[Optional[np.ndarray]] = [None] * boxes_per_row
                    used_cols = set()
                    group_sorted = sorted(best_short_group, key=lambda it: int(it["x"]))

                    for it in group_sorted:
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

                    row_y = int(round(float(np.mean([int(it["y"]) for it in group_sorted]))))
                    for c in range(boxes_per_row):
                        # Chuẩn hóa các box recover để tránh contour quá rộng chèn sang cột kế bên.
                        if assigned[c] is None:
                            assigned[c] = _rect_to_poly(col_x[c], row_y, col_w[c], row_h)
                            continue

                        rx, ry, rw, rh = cv2.boundingRect(assigned[c])
                        width_ok = (0.68 * col_w[c]) <= rw <= (1.42 * col_w[c])
                        height_ok = (0.68 * row_h) <= rh <= (1.42 * row_h)
                        center_x = rx + (rw / 2.0)
                        expected_center_x = col_x[c] + (col_w[c] / 2.0)
                        center_ok = abs(center_x - expected_center_x) <= max(8.0, col_w[c] * 0.55)

                        if not (width_ok and height_ok and center_ok):
                            assigned[c] = _rect_to_poly(col_x[c], row_y, col_w[c], row_h)

                    # Đảm bảo các cột trái -> phải không đè nhau do sai hình học cục bộ.
                    for c in range(boxes_per_row - 1):
                        x0, y0, w0, h0 = cv2.boundingRect(assigned[c])
                        x1, y1, w1, h1 = cv2.boundingRect(assigned[c + 1])
                        if x0 + w0 > x1:
                            assigned[c] = _rect_to_poly(col_x[c], row_y, col_w[c], row_h)
                            assigned[c + 1] = _rect_to_poly(col_x[c + 1], row_y, col_w[c + 1], row_h)

                    recovered_top = [box for box in assigned if box is not None]
                    if len(recovered_top) == boxes_per_row:
                        sobao_danh_rows = [recovered_top] + [row for _, row in row_with_y]
                        sobao_danh_rows = sobao_danh_rows[:max_rows]
                        sobao_danh = [box for row in sobao_danh_rows for box in row]
                        if debug:
                            print(
                                "  ✓ Recovered missing top SBD row from short top group "
                                f"(gap~{median_gap:.1f}, y={row_y}, short_len={len(best_short_group)})"
                            )
    
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
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện và gom nhóm các box của vùng Mã đề.

    Đặc trưng vùng Mã đề:
    - Mỗi hàng có 3 box.
    - Tối đa 10 hàng.
    - Kích thước box trong cùng hàng tương đối đồng đều.

    Args:
        boxes: Danh sách box ứng viên vùng Mã đề.
        boxes_per_row: Số box kỳ vọng trên mỗi hàng.
        max_rows: Số hàng tối đa cần giữ.
        row_tolerance: Dung sai gom nhóm theo trục Y.
        size_tolerance_ratio: Dung sai đồng đều kích thước trong hàng.
        debug: Bật/tắt log debug.

    Returns:
        Dictionary gồm:
        - 'ma_de': Danh sách toàn bộ box Mã đề đã phát hiện.
        - 'ma_de_rows': Danh sách các hàng, mỗi hàng chứa 3 box.
        - 'row_count': Số hàng đã phát hiện.
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
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
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


def build_fill_scoring_binary(
    image: np.ndarray,
    block_size: int = 31,
    block_offset: int = 9,
) -> np.ndarray:
    # Dựng binary riêng cho bước chấm tô để giảm hiện tượng nét trắng "nở" quá dày.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, int(block_size))

    binary_score = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(block_offset),
    )

    # Opening nhẹ để làm mảnh biên trắng và loại hạt nhiễu nhỏ.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary_score = cv2.morphologyEx(binary_score, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return binary_score


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
    # Phát hiện dữ liệu/đối tượng theo tiêu chí của bước này.
    """
    Phát hiện các ô kín tạo bởi mạng đường dọc/ngang từ morphology.

    Các bước chính:
    1. Tách mask đường dọc/ngang.
    2. Ghép line mask và đóng các khe hở nhỏ.
    3. Flood-fill nền ngoài để tách vùng kín bên trong.
    4. Tách connected components để lấy contour ô.

    Args:
        image: Ảnh đầu vào.
        vertical_scale: Tỉ lệ chiều dài kernel dọc.
        horizontal_scale: Tỉ lệ chiều dài kernel ngang.
        min_line_length: Ngưỡng độ dài tối thiểu của line thành phần.
        align_vertical_rows: Bật căn chỉnh độ dài line dọc theo hàng.
        vertical_row_tolerance: Dung sai gom line dọc theo hàng.
        block_size: Kích thước block của adaptive threshold.
        block_offset: Hệ số C của adaptive threshold.
        min_box_area: Diện tích tối thiểu để giữ contour box.
        min_box_width: Chiều rộng tối thiểu của box.
        min_box_height: Chiều cao tối thiểu của box.
        close_kernel_size: Kích thước kernel đóng khe hở line.
        debug_prefix: Tiền tố lưu ảnh debug trung gian.

    Returns:
        Dictionary chứa ảnh trung gian, overlay và danh sách contour box phát hiện.
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

    # Chỉ giữ các đoạn line đủ dài để tránh sinh box từ nhiễu ngắn.
    vertical = _filter_line_components_by_length(vertical, min_line_length, "vertical")
    horizontal = _filter_line_components_by_length(horizontal, min_line_length, "horizontal")

    # Căn lại các line dọc song song trong cùng hàng để biên trên/dưới đồng đều hơn.
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

    # Lấy contour đa giác thật thay vì chỉ dùng bounding box chữ nhật trục chuẩn.
    contours, _ = cv2.findContours(enclosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[np.ndarray] = []  # Lưu polygon contour để giữ đúng hình học phối cảnh.
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_box_area:
            continue
        
        # Xấp xỉ contour thành polygon với epsilon nhỏ để giữ biên dạng tốt hơn.
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Dùng boundingRect để lọc nhanh các contour quá nhỏ.
        x, y, bw, bh = cv2.boundingRect(approx)
        
        if bw < min_box_width or bh < min_box_height:
            continue

        boxes.append(approx)

    # Sắp theo vị trí để đầu ra ổn định giữa các lần chạy.
    boxes.sort(key=lambda b: (int(b[0, 0, 1]), int(b[0, 0, 0])))

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for poly in boxes:
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)

    result = {
        "binary": grid["binary"],
        "binary_score": build_fill_scoring_binary(
            image,
            block_size=max(21, int(block_size) - 4),
            block_offset=int(block_offset) + 2,
        ),
        "vertical": vertical,
        "horizontal": horizontal,
        "lines": lines,
        "lines_closed": lines_closed,
        "enclosed": enclosed,
        "boxes_overlay": overlay,
        "boxes": boxes,
    }

    if debug_prefix:
        # cv2.imwrite(f"{debug_prefix}_vertical.jpg", vertical)
        # cv2.imwrite(f"{debug_prefix}_horizontal.jpg", horizontal)
        # cv2.imwrite(f"{debug_prefix}_lines.jpg", lines)
        # cv2.imwrite(f"{debug_prefix}_lines_closed.jpg", lines_closed)
        # cv2.imwrite(f"{debug_prefix}_enclosed.jpg", enclosed)
        cv2.imwrite(f"{debug_prefix}_boxes.jpg", overlay)

    return result


def detect_black_corner_markers(
    image: np.ndarray,
    debug_prefix: Optional[str] = None,
) -> Dict[str, object]:
    # Phát hiện 4 marker đen ở 4 góc ảnh để định vị khung phiếu.
    h_img, w_img = image.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return {
            "corners": {
                "top_left": None,
                "top_right": None,
                "bottom_right": None,
                "bottom_left": None,
            },
            "ordered_corners": [],
            "found_count": 0,
            "all_found": False,
            "candidate_count": 0,
            "debug_image_path": None,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Marker góc là vùng tối: lấy ngưỡng tối từ Otsu nhưng chặn trên để tránh mask trắng toàn vùng.
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_thresh = int(np.clip(0.62 * float(otsu_val), 60.0, 130.0))
    _, binary_inv = cv2.threshold(blur, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    img_area = float(h_img * w_img)
    diag = float(np.hypot(w_img, h_img))

    selected: Dict[str, Optional[Tuple[int, int]]] = {
        "top_left": None,
        "top_right": None,
        "bottom_right": None,
        "bottom_left": None,
    }
    selected_bboxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {
        "top_left": None,
        "top_right": None,
        "bottom_right": None,
        "bottom_left": None,
    }
    candidates: List[Dict[str, object]] = []

    def _pick_corner_from_roi(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        target: Tuple[float, float],
    ) -> Optional[Dict[str, object]]:
        x0 = int(np.clip(x0, 0, w_img - 1))
        y0 = int(np.clip(y0, 0, h_img - 1))
        x1 = int(np.clip(x1, x0 + 1, w_img))
        y1 = int(np.clip(y1, y0 + 1, h_img))

        roi = binary_inv[y0:y1, x0:x1]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_min_area = max(8.0, 0.000002 * img_area)
        local_max_area = max(local_min_area + 1.0, 0.030 * img_area)

        best = None
        best_score = -1e9
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < local_min_area or area > local_max_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw < 3 or rh < 3:
                continue

            gx, gy = x0 + rx, y0 + ry
            if gx <= 2 or gy <= 2 or (gx + rw) >= (w_img - 2) or (gy + rh) >= (h_img - 2):
                continue

            extent = area / float(max(1, rw * rh))
            aspect = float(rw) / float(rh)
            if extent < 0.10 or not (0.03 <= aspect <= 35.0):
                continue

            roi_gray = gray[gy:gy + rh, gx:gx + rw]
            if roi_gray.size == 0:
                continue
            mean_intensity = float(np.mean(roi_gray))
            if mean_intensity > 235.0:
                continue

            cx = float(gx + 0.5 * rw)
            cy = float(gy + 0.5 * rh)
            distance_norm = float(np.hypot(cx - target[0], cy - target[1])) / max(diag, 1.0)
            area_norm = area / max(img_area, 1.0)
            area_boost = min(1.0, area_norm / 0.0015)

            # Ưu tiên gần góc là chính, chỉ dùng extent/area như tín hiệu phụ.
            score = (0.35 * extent) + (0.10 * area_boost) - (4.0 * distance_norm)
            if score > best_score:
                best_score = score
                best = {
                    "bbox": (int(gx), int(gy), int(rw), int(rh)),
                    "center": (int(round(cx)), int(round(cy))),
                    "area": area,
                    "extent": extent,
                    "score": score,
                }

        return best

    roi_x = max(40, int(round(0.24 * w_img)))
    roi_y = max(40, int(round(0.24 * h_img)))
    corner_targets = {
        "top_left": (0.0, 0.0),
        "top_right": (float(w_img - 1), 0.0),
        "bottom_right": (float(w_img - 1), float(h_img - 1)),
        "bottom_left": (0.0, float(h_img - 1)),
    }

    corner_rois = {
        "top_left": (0, 0, roi_x, roi_y),
        "top_right": (w_img - roi_x, 0, w_img, roi_y),
        "bottom_right": (w_img - roi_x, h_img - roi_y, w_img, h_img),
        "bottom_left": (0, h_img - roi_y, roi_x, h_img),
    }

    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        target = corner_targets[name]
        x0, y0, x1, y1 = corner_rois[name]

        picked = _pick_corner_from_roi(x0, y0, x1, y1, target)
        if picked is None:
            # ROI lớn hơn nếu patch góc đầu tiên không thấy marker.
            roi_x2 = max(roi_x, int(round(0.34 * w_img)))
            roi_y2 = max(roi_y, int(round(0.34 * h_img)))
            if name == "top_left":
                picked = _pick_corner_from_roi(0, 0, roi_x2, roi_y2, target)
            elif name == "top_right":
                picked = _pick_corner_from_roi(w_img - roi_x2, 0, w_img, roi_y2, target)
            elif name == "bottom_right":
                picked = _pick_corner_from_roi(w_img - roi_x2, h_img - roi_y2, w_img, h_img, target)
            else:
                picked = _pick_corner_from_roi(0, h_img - roi_y2, roi_x2, h_img, target)

        if picked is not None:
            selected[name] = picked["center"]
            selected_bboxes[name] = picked["bbox"]
            candidates.append(picked)

    # Fallback toàn ảnh cho các góc chưa bắt được trong ROI.
    missing_names = [name for name, pt in selected.items() if pt is None]
    if missing_names:
        global_contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        global_candidates: List[Dict[str, object]] = []
        min_a = max(8.0, 0.000002 * img_area)
        max_a = max(min_a + 1.0, 0.035 * img_area)

        for contour in global_contours:
            area = float(cv2.contourArea(contour))
            if area < min_a or area > max_a:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue
            if x <= 2 or y <= 2 or (x + w) >= (w_img - 2) or (y + h) >= (h_img - 2):
                continue
            extent = area / float(max(1, w * h))
            aspect = float(w) / float(h)
            if extent < 0.10 or not (0.03 <= aspect <= 35.0):
                continue

            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.size == 0:
                continue
            if float(np.mean(roi_gray)) > 235.0:
                continue

            cx = float(x + 0.5 * w)
            cy = float(y + 0.5 * h)
            global_candidates.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center": (int(round(cx)), int(round(cy))),
                    "extent": extent,
                }
            )

        used_centers = {pt for pt in selected.values() if pt is not None}
        for name in missing_names:
            target = corner_targets[name]
            best = None
            best_score = float("inf")
            for cand in global_candidates:
                center = cand["center"]
                if center in used_centers:
                    continue

                cx, cy = float(center[0]), float(center[1])
                dist_norm = float(np.hypot(cx - target[0], cy - target[1])) / max(diag, 1.0)

                if name == "top_left":
                    quadrant_penalty = 0.0 if (cx <= 0.5 * w_img and cy <= 0.5 * h_img) else 0.7
                elif name == "top_right":
                    quadrant_penalty = 0.0 if (cx >= 0.5 * w_img and cy <= 0.5 * h_img) else 0.7
                elif name == "bottom_right":
                    quadrant_penalty = 0.0 if (cx >= 0.5 * w_img and cy >= 0.5 * h_img) else 0.7
                else:
                    quadrant_penalty = 0.0 if (cx <= 0.5 * w_img and cy >= 0.5 * h_img) else 0.7

                score = dist_norm + quadrant_penalty
                if score < best_score:
                    best_score = score
                    best = cand

            if best is not None:
                selected[name] = best["center"]
                selected_bboxes[name] = best["bbox"]
                candidates.append(best)
                used_centers.add(best["center"])

    # Loại các điểm quá xa góc mục tiêu để tránh nhận nhầm marker giữa cạnh.
    max_corner_distance_ratio = 0.32
    for name, pt in list(selected.items()):
        if pt is None:
            continue
        target = corner_targets[name]
        dist_norm = float(np.hypot(float(pt[0]) - target[0], float(pt[1]) - target[1])) / max(diag, 1.0)
        if dist_norm > max_corner_distance_ratio:
            selected[name] = None
            selected_bboxes[name] = None

    # Kiểm tra hình học vùng hợp lệ để tránh góc bị kéo vào giữa trang.
    for name, pt in list(selected.items()):
        if pt is None:
            continue
        px, py = float(pt[0]), float(pt[1])
        invalid = False
        if name in ("top_left", "top_right") and py > 0.40 * h_img:
            invalid = True
        if name in ("bottom_left", "bottom_right") and py < 0.88 * h_img:
            invalid = True
        if name in ("top_left", "bottom_left") and px > 0.25 * w_img:
            invalid = True
        if name in ("top_right", "bottom_right") and px < 0.75 * w_img:
            invalid = True

        if invalid:
            selected[name] = None
            selected_bboxes[name] = None

    # Nội suy góc thiếu từ các góc còn lại (giả định ảnh gần song song trục).
    tl = selected["top_left"]
    tr = selected["top_right"]
    br = selected["bottom_right"]
    bl = selected["bottom_left"]

    if br is None and tr is not None and bl is not None:
        selected["bottom_right"] = (int(tr[0]), int(bl[1]))
    if bl is None and tl is not None and br is not None:
        selected["bottom_left"] = (int(tl[0]), int(br[1]))
    if tr is None and tl is not None and br is not None:
        selected["top_right"] = (int(br[0]), int(tl[1]))
    if tl is None and tr is not None and bl is not None:
        selected["top_left"] = (int(bl[0]), int(tr[1]))

    # Sửa corner outlier khi đã đủ 4 góc nhưng một góc lệch mạnh so với hình học cạnh.
    if all(selected[name] is not None for name in ("top_left", "top_right", "bottom_right", "bottom_left")):
        def _corner_dist_ratio(name: str, pt: Tuple[int, int]) -> float:
            target = corner_targets[name]
            return float(np.hypot(float(pt[0]) - target[0], float(pt[1]) - target[1])) / max(diag, 1.0)

        def _clip_point(x: int, y: int) -> Tuple[int, int]:
            return (
                int(np.clip(x, 0, w_img - 1)),
                int(np.clip(y, 0, h_img - 1)),
            )

        # Dung sai hình học theo kích thước ảnh để không quá nhạy với góc chụp nghiêng nhẹ.
        y_edge_tol = max(22, int(round(0.06 * h_img)))
        x_edge_tol = max(18, int(round(0.06 * w_img)))

        tl_pt = selected["top_left"]
        tr_pt = selected["top_right"]
        br_pt = selected["bottom_right"]
        bl_pt = selected["bottom_left"]

        dist = {
            "top_left": _corner_dist_ratio("top_left", tl_pt),
            "top_right": _corner_dist_ratio("top_right", tr_pt),
            "bottom_right": _corner_dist_ratio("bottom_right", br_pt),
            "bottom_left": _corner_dist_ratio("bottom_left", bl_pt),
        }

        def _is_corner_outlier(name: str, peer: str) -> bool:
            return dist[name] > max(0.09, dist[peer] * 1.8)

        top_y_delta = abs(int(tl_pt[1]) - int(tr_pt[1]))
        bottom_y_delta = abs(int(bl_pt[1]) - int(br_pt[1]))
        left_x_delta = abs(int(tl_pt[0]) - int(bl_pt[0]))
        right_x_delta = abs(int(tr_pt[0]) - int(br_pt[0]))

        if top_y_delta > y_edge_tol:
            if _is_corner_outlier("top_left", "top_right"):
                selected["top_left"] = _clip_point(int(bl_pt[0]), int(tr_pt[1]))
                selected_bboxes["top_left"] = None
            elif _is_corner_outlier("top_right", "top_left"):
                selected["top_right"] = _clip_point(int(br_pt[0]), int(tl_pt[1]))
                selected_bboxes["top_right"] = None

        # Re-read points after possible top-edge correction.
        tl_pt = selected["top_left"]
        tr_pt = selected["top_right"]
        br_pt = selected["bottom_right"]
        bl_pt = selected["bottom_left"]

        if bottom_y_delta > y_edge_tol:
            if _is_corner_outlier("bottom_left", "bottom_right"):
                selected["bottom_left"] = _clip_point(int(tl_pt[0]), int(br_pt[1]))
                selected_bboxes["bottom_left"] = None
            elif _is_corner_outlier("bottom_right", "bottom_left"):
                selected["bottom_right"] = _clip_point(int(tr_pt[0]), int(bl_pt[1]))
                selected_bboxes["bottom_right"] = None

        tl_pt = selected["top_left"]
        tr_pt = selected["top_right"]
        br_pt = selected["bottom_right"]
        bl_pt = selected["bottom_left"]

        if left_x_delta > x_edge_tol:
            if _is_corner_outlier("top_left", "bottom_left"):
                selected["top_left"] = _clip_point(int(bl_pt[0]), int(tr_pt[1]))
                selected_bboxes["top_left"] = None
            elif _is_corner_outlier("bottom_left", "top_left"):
                selected["bottom_left"] = _clip_point(int(tl_pt[0]), int(br_pt[1]))
                selected_bboxes["bottom_left"] = None

        tl_pt = selected["top_left"]
        tr_pt = selected["top_right"]
        br_pt = selected["bottom_right"]
        bl_pt = selected["bottom_left"]

        if right_x_delta > x_edge_tol:
            if _is_corner_outlier("top_right", "bottom_right"):
                selected["top_right"] = _clip_point(int(br_pt[0]), int(tl_pt[1]))
                selected_bboxes["top_right"] = None
            elif _is_corner_outlier("bottom_right", "top_right"):
                selected["bottom_right"] = _clip_point(int(tr_pt[0]), int(bl_pt[1]))
                selected_bboxes["bottom_right"] = None

    ordered_corners: List[Tuple[int, int]] = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        pt = selected[name]
        if pt is not None:
            ordered_corners.append(pt)

    debug_image_path = None
    if debug_prefix:
        debug_img = image.copy() if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cand in candidates:
            x, y, w, h = cand["bbox"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 180, 0), 2)

        draw_colors = {
            "top_left": (0, 255, 0),
            "top_right": (0, 255, 255),
            "bottom_right": (0, 128, 255),
            "bottom_left": (255, 0, 0),
        }
        for name, pt in selected.items():
            if pt is None:
                continue
            color = draw_colors.get(name, (0, 255, 0))
            cv2.circle(debug_img, pt, 8, color, -1)
            cv2.putText(debug_img, name, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        debug_image_path = f"{debug_prefix}_corner_markers.jpg"
        cv2.imwrite(debug_image_path, debug_img)

    found_count = sum(1 for pt in selected.values() if pt is not None)
    return {
        "corners": selected,
        "ordered_corners": ordered_corners,
        "found_count": found_count,
        "all_found": found_count == 4,
        "candidate_count": len(candidates),
        "debug_image_path": debug_image_path,
    }


def extrapolate_missing_rows(
    detection_results: Dict[str, object],
    target_rows: int = 10,
    expected_top_y: Optional[int] = None,
    expected_row_step: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, object]:
    # Nội suy/ngoại suy dữ liệu thiếu dựa trên mốc tham chiếu.
    """
    Nội suy/ngoại suy các hàng còn thiếu cho SoBaoDanh và Mã đề dựa trên khoảng cách hàng.

    Giả định các hàng của hai vùng được căn thẳng theo trục dọc và có khoảng cách gần đều.
    Hàm sẽ điền các hàng còn thiếu để đạt đủ `target_rows`.

    Args:
        detection_results: Dictionary chứa 'sobao_danh_rows' và 'ma_de_rows'.
        target_rows: Số hàng mục tiêu (mặc định 10).
        debug: Bật in thông tin debug khi xử lý.

    Returns:
        Dictionary kết quả đã bổ sung các hàng nội suy/ngoại suy.
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
    
    sobao_is_synthetic = bool(detection_results.get("sobao_is_synthetic", False))
    ma_de_is_synthetic = bool(detection_results.get("ma_de_is_synthetic", False))

    if debug:
        print(f"\n[DEBUG] SoBaoDanh Y positions: {sobao_y_positions}")
        print(f"[DEBUG] MaDe Y positions: {ma_de_y_positions}")
        print(f"[DEBUG] SoBaoDanh synthetic: {sobao_is_synthetic}")
        print(f"[DEBUG] MaDe synthetic: {ma_de_is_synthetic}")
    
    # Use SoBaoDanh Y positions as reference (since it has all 10 rows detected)
    # If SoBaoDanh has 10 rows, use them directly
    if len(sobao_y_positions) == target_rows and not sobao_is_synthetic:
        reference_positions = sobao_y_positions
        if debug:
            print(f"[DEBUG] Using SoBaoDanh Y positions directly: {reference_positions}")
    else:
        # If SoBaoDanh doesn't have all rows, calculate expected positions
        # Calculate average row spacing from the section with more rows
        if sobao_is_synthetic and not ma_de_is_synthetic and ma_de_y_positions:
            spacing_positions = ma_de_y_positions
        elif ma_de_is_synthetic and not sobao_is_synthetic and sobao_y_positions:
            spacing_positions = sobao_y_positions
        else:
            spacing_positions = sobao_y_positions if len(sobao_y_positions) >= len(ma_de_y_positions) else ma_de_y_positions
        
        if len(spacing_positions) >= 2:
            spacings = [spacing_positions[i + 1] - spacing_positions[i] for i in range(len(spacing_positions) - 1)]
            avg_spacing = int(round(float(np.median(spacings)))) if spacings else 0
        elif expected_row_step is not None and int(expected_row_step) > 0:
            avg_spacing = int(expected_row_step)
        else:
            avg_spacing = 0
        
        if spacing_positions:
            first_detected_y = int(spacing_positions[0])
            first_row_y = first_detected_y

            if avg_spacing > 0 and expected_top_y is not None:
                expected_top = int(expected_top_y)
                best_offset = 0
                best_score = float("inf")
                for offset in range(target_rows):
                    candidate_top = first_detected_y - (offset * avg_spacing)
                    score = abs(candidate_top - expected_top)
                    if score < best_score:
                        best_score = score
                        best_offset = offset
                first_row_y = first_detected_y - (best_offset * avg_spacing)

                clamp_margin = max(30, int(round(avg_spacing * 2.0)))
                first_row_y = int(np.clip(first_row_y, expected_top - clamp_margin, expected_top + clamp_margin))

            if avg_spacing > 0:
                reference_positions = [first_row_y + i * avg_spacing for i in range(target_rows)]
            else:
                reference_positions = [first_row_y for _ in range(target_rows)]
            if debug:
                print(f"[DEBUG] Calculated reference positions: {reference_positions}")
        else:
            if expected_top_y is not None and expected_row_step is not None and int(expected_row_step) > 0:
                y0 = int(expected_top_y)
                step = int(expected_row_step)
                reference_positions = [y0 + i * step for i in range(target_rows)]
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
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        """Căn các hàng phát hiện được vào các vị trí tham chiếu theo trục Y."""
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
    sobao_tolerance = 30
    ma_de_tolerance = 90
    if expected_row_step is not None and int(expected_row_step) > 0:
        base_tol = int(round(float(expected_row_step) * 2.2))
        if sobao_is_synthetic:
            sobao_tolerance = max(sobao_tolerance, base_tol)
        if ma_de_is_synthetic:
            ma_de_tolerance = max(ma_de_tolerance, base_tol)

    aligned_sobao = align_rows_to_reference_positions(
        sobao_danh_rows, sobao_y_positions, reference_positions, "SoBaoDanh", tolerance=sobao_tolerance)
    aligned_ma_de = align_rows_to_reference_positions(
        ma_de_rows, ma_de_y_positions, reference_positions, "MaDe", tolerance=ma_de_tolerance)
    
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
    # Chuẩn hóa đầu vào về định dạng ổn định trước khi xử lý.
    """
    Chuẩn hóa tham số ảnh người dùng nhập về dạng `PhieuQG.XXXX`.

    Args:
        image_arg: Chuỗi người dùng nhập (id, tên file, hoặc đường dẫn).

    Returns:
        Tên stem chuẩn hóa để dùng thống nhất trong pipeline.
    """
    if not image_arg:
        return "PhieuQG.0015"

    raw = image_arg.strip()
    if not raw:
        return "PhieuQG.0015"

    token = Path(raw).name
    token_lower = token.lower()
    if token_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
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


def _draw_rows_contours(
    image: np.ndarray,
    rows: List[List[np.ndarray]],
    color: Tuple[int, int, int],
    thickness: int,
) -> int:
    # Hàm hỗ trợ vẽ hình học/phần tử phụ trợ trong pipeline.
    """
    Vẽ contour cho danh sách hàng box và trả về số box đã vẽ.

    Args:
        image: Ảnh cần vẽ.
        rows: Danh sách hàng box.
        color: Màu vẽ contour.
        thickness: Độ dày nét vẽ.

    Returns:
        Tổng số box đã được vẽ lên ảnh.
    """
    drawn = 0
    for row in rows:
        for poly in row:
            cv2.polylines(image, [poly], True, color, thickness)
            drawn += 1
    return drawn


def _print_grid_info(
    grid_info: List[Dict[str, object]],
    detail_formatter: Optional[Callable[[Dict[str, object]], str]] = None,
) -> None:
    # In tóm tắt nhanh để theo dõi chất lượng xử lý khi chạy.
    """
    In thông tin grid theo định dạng thống nhất cho log debug.

    Args:
        grid_info: Danh sách metadata lưới.
        detail_formatter: Hàm tùy chọn để format thêm thông tin theo từng box.

    Returns:
        Không trả về giá trị.
    """
    print(f"Grid drawn on {len(grid_info)} boxes")
    for info in grid_info:
        detail = detail_formatter(info) if detail_formatter is not None else ""
        suffix = f", {detail}" if detail else ""
        print(
            f"  Box {info['box_idx']}: region {info['region_size']}, "
            f"cell_size ~{info['cell_size'][0]:.1f}x{info['cell_size'][1]:.1f}{suffix}"
        )


def _evaluate_section_fill(
    section_name: str,
    binary_threshold: Optional[np.ndarray],
    grid_info: List[Dict[str, object]],
    fill_ratio_thresh: float,
    inner_margin_ratio: float,
    circle_radius_scale: float,
    circle_border_exclude_ratio: float,
) -> List[Dict[str, object]]:
    # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
    """
    Đánh giá ô tô đậm cho một section với cấu hình mask vòng tròn thống nhất.

    Args:
        section_name: Tên section dùng cho log.
        binary_threshold: Ảnh nhị phân đầu vào.
        grid_info: Metadata lưới của section.
        fill_ratio_thresh: Ngưỡng phân loại tô/không tô.
        inner_margin_ratio: Tỉ lệ co vùng chấm bên trong mỗi ô.
        circle_radius_scale: Hệ số bán kính bubble.
        circle_border_exclude_ratio: Tỉ lệ bỏ viền ngoài bubble.

    Returns:
        Danh sách kết quả chấm cho section.
    """
    if binary_threshold is None or not grid_info:
        return []

    total_cells = 0
    for info in grid_info:
        rows, cols = info.get("grid_shape", (0, 0))
        rows = int(rows)
        cols = int(cols)
        pattern = info.get("pattern")

        if isinstance(pattern, list) and rows > 0 and cols > 0:
            count = 0
            for r in range(rows):
                if r < len(pattern) and isinstance(pattern[r], list):
                    count += len([c for c in pattern[r] if 0 <= int(c) < cols])
                else:
                    count += cols
            total_cells += count
        else:
            total_cells += max(0, rows * cols)

    hough_budget = max(64, min(220, int(round(total_cells * 0.45))))

    evals = evaluate_grid_fill_from_binary(
        binary_image=binary_threshold,
        grid_info=grid_info,
        fill_ratio_thresh=fill_ratio_thresh,
        inner_margin_ratio=inner_margin_ratio,
        mask_mode="hough-circle",
        circle_radius_scale=circle_radius_scale,
        circle_border_exclude_ratio=circle_border_exclude_ratio,
        hough_only_ambiguous=True,
        hough_ambiguity_margin=0.10,
        hough_max_cells=hough_budget,
    )
    _print_fill_summary(section_name, evals)
    return evals


def _build_synthetic_id_rows_from_part_i(
    image_shape: Tuple[int, int],
    part_i_boxes: List[np.ndarray],
    cols: int,
    rows: int,
    x_range_ratio: Tuple[float, float],
    distance_from_part_i_ratio: float,
    row_step_ratio: float,
) -> List[List[np.ndarray]]:
    # Hàm hỗ trợ dựng lưới ID tổng hợp khi detect quá ít hàng.
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    if h_img <= 0 or w_img <= 0 or cols <= 0 or rows <= 0:
        return []

    # Neo theo mép trên Part I; nếu thiếu Part I thì dùng mốc gần đúng theo layout.
    if part_i_boxes:
        part_i_top = min(cv2.boundingRect(b)[1] for b in part_i_boxes)
    else:
        part_i_top = int(round(h_img * 0.3))

    distance_ratio = float(np.clip(distance_from_part_i_ratio, 0.05, 0.60))
    row_step_ratio = float(np.clip(row_step_ratio, 0.010, 0.080))

    distance_px = max(1, int(round(distance_ratio * h_img)))
    step_y = max(6, int(round(row_step_ratio * h_img)))
    # Stack rows directly with no vertical gap.
    row_h = step_y

    # Tính Y hàng đầu từ khoảng cách tới Part I và chặn trong ảnh.
    y0 = part_i_top - distance_px
    grid_h = rows * row_h
    y0 = min(y0, h_img - grid_h - 1)
    y0 = max(0, y0)

    xr0 = float(np.clip(x_range_ratio[0], 0.0, 0.99))
    xr1 = float(np.clip(x_range_ratio[1], xr0 + 1e-4, 1.0))
    x0 = int(round(xr0 * w_img))
    x1 = int(round(xr1 * w_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + cols)

    # Build contiguous column edges so boxes touch each other horizontally.
    x_edges = np.round(np.linspace(x0, x1, cols + 1)).astype(np.int32)
    x_edges[0] = x0
    x_edges[-1] = x1
    for i in range(1, len(x_edges)):
        min_allowed = x_edges[i - 1] + 1
        max_allowed = x1 - (cols - i)
        x_edges[i] = int(np.clip(x_edges[i], min_allowed, max_allowed))

    out_rows: List[List[np.ndarray]] = []
    for r in range(rows):
        y = y0 + (r * row_h)
        y_top = int(y)

        row_boxes: List[np.ndarray] = []
        for c in range(cols):
            x_left = int(x_edges[c])
            x_right = int(x_edges[c + 1])
            box_w = max(1, x_right - x_left)
            row_boxes.append(_rect_to_poly(x_left, y_top, box_w, row_h))

        out_rows.append(row_boxes)

    return out_rows


def _build_synthetic_id_rows_fixed_image_position(
    image_shape: Tuple[int, int],
    cols: int,
    rows: int,
    x_range_ratio: Tuple[float, float],
    top_y_ratio: float,
    row_step_ratio: float,
) -> List[List[np.ndarray]]:
    # Dựng lưới ID cố định theo vị trí ảnh (không phụ thuộc Part I).
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    if h_img <= 0 or w_img <= 0 or cols <= 0 or rows <= 0:
        return []

    top_ratio = float(np.clip(top_y_ratio, 0.0, 0.95))
    row_step_ratio = float(np.clip(row_step_ratio, 0.010, 0.080))

    y0 = int(round(top_ratio * h_img))
    row_h = max(6, int(round(row_step_ratio * h_img)))
    grid_h = rows * row_h
    y0 = min(y0, h_img - grid_h - 1)
    y0 = max(0, y0)

    xr0 = float(np.clip(x_range_ratio[0], 0.0, 0.99))
    xr1 = float(np.clip(x_range_ratio[1], xr0 + 1e-4, 1.0))
    x0 = int(round(xr0 * w_img))
    x1 = int(round(xr1 * w_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + cols)

    x_edges = np.round(np.linspace(x0, x1, cols + 1)).astype(np.int32)
    x_edges[0] = x0
    x_edges[-1] = x1
    for i in range(1, len(x_edges)):
        min_allowed = x_edges[i - 1] + 1
        max_allowed = x1 - (cols - i)
        x_edges[i] = int(np.clip(x_edges[i], min_allowed, max_allowed))

    out_rows: List[List[np.ndarray]] = []
    for r in range(rows):
        y_top = int(y0 + (r * row_h))
        row_boxes: List[np.ndarray] = []
        for c in range(cols):
            x_left = int(x_edges[c])
            x_right = int(x_edges[c + 1])
            box_w = max(1, x_right - x_left)
            row_boxes.append(_rect_to_poly(x_left, y_top, box_w, row_h))
        out_rows.append(row_boxes)

    return out_rows


def _build_synthetic_id_rows_from_reference_positions(
    image_shape: Tuple[int, int],
    cols: int,
    x_range_ratio: Tuple[float, float],
    reference_positions: List[int],
    row_height: Optional[int] = None,
) -> List[List[np.ndarray]]:
    # Dựng lưới ID tổng hợp với trục Y neo theo các vị trí tham chiếu đã biết.
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    if h_img <= 0 or w_img <= 0 or cols <= 0 or not reference_positions:
        return []

    ref = [int(y) for y in reference_positions]
    ref.sort()

    if row_height is None or int(row_height) <= 0:
        if len(ref) >= 2:
            diffs = [max(1, ref[i + 1] - ref[i]) for i in range(len(ref) - 1)]
            row_h = int(round(float(np.median(diffs))))
        else:
            row_h = max(6, int(round(0.018 * h_img)))
    else:
        row_h = max(1, int(row_height))

    xr0 = float(np.clip(x_range_ratio[0], 0.0, 0.99))
    xr1 = float(np.clip(x_range_ratio[1], xr0 + 1e-4, 1.0))
    x0 = int(round(xr0 * w_img))
    x1 = int(round(xr1 * w_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + cols)

    x_edges = np.round(np.linspace(x0, x1, cols + 1)).astype(np.int32)
    x_edges[0] = x0
    x_edges[-1] = x1
    for i in range(1, len(x_edges)):
        min_allowed = x_edges[i - 1] + 1
        max_allowed = x1 - (cols - i)
        x_edges[i] = int(np.clip(x_edges[i], min_allowed, max_allowed))

    out_rows: List[List[np.ndarray]] = []
    for y_ref in ref:
        y_top = int(np.clip(y_ref, 0, max(0, h_img - row_h - 1)))
        row_boxes: List[np.ndarray] = []
        for c in range(cols):
            x_left = int(x_edges[c])
            x_right = int(x_edges[c + 1])
            box_w = max(1, x_right - x_left)
            row_boxes.append(_rect_to_poly(x_left, y_top, box_w, row_h))
        out_rows.append(row_boxes)

    return out_rows


def _apply_affine_from_corner_markers(
    image: np.ndarray,
    corners: Dict[str, Optional[Tuple[int, int]]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # Hiệu chỉnh topview từ 4 góc marker bằng perspective transform.
    if image is None or corners is None:
        return None, None

    tl = corners.get("top_left")
    tr = corners.get("top_right")
    br = corners.get("bottom_right")
    bl = corners.get("bottom_left")
    if tl is None or tr is None or br is None or bl is None:
        return None, None

    h_img, w_img = image.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return None, None

    src = np.array(
        [
            [float(tl[0]), float(tl[1])],
            [float(tr[0]), float(tr[1])],
            [float(br[0]), float(br[1])],
            [float(bl[0]), float(bl[1])],
        ],
        dtype=np.float32,
    )

    width_top = float(np.hypot(src[1, 0] - src[0, 0], src[1, 1] - src[0, 1]))
    width_bottom = float(np.hypot(src[2, 0] - src[3, 0], src[2, 1] - src[3, 1]))
    height_left = float(np.hypot(src[3, 0] - src[0, 0], src[3, 1] - src[0, 1]))
    height_right = float(np.hypot(src[2, 0] - src[1, 0], src[2, 1] - src[1, 1]))

    if min(width_top, width_bottom, height_left, height_right) < 20.0:
        return None, None

    # Chuẩn hóa toàn trang về topview chiếm toàn khung ảnh đầu ra.
    dst = np.array(
        [
            [0.0, 0.0],
            [float(w_img - 1), 0.0],
            [float(w_img - 1), float(h_img - 1)],
            [0.0, float(h_img - 1)],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix


def _demo(image_arg: Optional[str] = None, topview_debug: bool = False) -> None:
    # Pipeline demo đầy đủ: detect, chấm, vẽ và xuất debug.
    # Chuẩn hóa tên ảnh đầu vào để hỗ trợ nhiều kiểu tham số (0015, PhieuQG.0015, ...).
    base_image_name = _normalize_image_stem(image_arg)
    candidate_paths = [
        Path("PhieuQG") / f"{base_image_name}.jpg",
        Path("PhieuQG") / f"{base_image_name}.jpeg",
        Path("PhieuQG") / f"{base_image_name}.png",
        Path("PhieuQG") / f"{base_image_name}.BMP",
        Path("PhieuQG") / f"{base_image_name}.tiff",
    ]
    image_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    out_dir = Path("output/detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        tried = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Cannot read image. Tried: {tried}")

    orig_h, orig_w = img.shape[:2]
    img, resize_scale = resize_image_for_inference(img, max_side=2200)
    if resize_scale < 0.9999:
        resized_h, resized_w = img.shape[:2]
        print(
            f"[Resize] {orig_w}x{orig_h} -> {resized_w}x{resized_h} "
            f"(scale={resize_scale:.3f})"
        )
    img_original = img.copy()

    def _run_detection_pipeline(src_img: np.ndarray, prefix: Optional[str]) -> Tuple[Dict[str, object], Dict[str, object]]:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Pipeline phát hiện box + gom phần I/II/III.
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
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Tăng tương phản cục bộ để cứu các scan mờ/thiếu nét.
        lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _parts_score(parts_local: Dict[str, object], page_h: int) -> float:
        # Hàm hỗ trợ một bước xử lý trong pipeline chấm phiếu.
        # Điểm heuristic để chọn kết quả detect tốt hơn giữa base và CLAHE.
        p1 = len(parts_local["part_i"])
        p2 = len(parts_local["part_ii"])
        p3 = len(parts_local["part_iii"])
        score = 0.0
        score += 3.0 if p1 == 4 else (p1 / 4.0)
        score += 4.0 * min(1.0, p2 / 8.0)
        score += 3.0 * min(1.0, p3 / 6.0)

        # Ưu tiên cấu hình có Part III nằm ở vùng thấp của trang (đúng bố cục phiếu).
        if p3 and page_h > 0:
            y_vals = [cv2.boundingRect(b)[1] for b in parts_local["part_iii"]]
            p3_ratio = float(np.mean(y_vals)) / float(page_h)
            score += max(0.0, min(1.0, (p3_ratio - 0.55) / 0.20))
        return score

    debug_prefix = str(out_dir / image_path.stem)
    corner_markers = detect_black_corner_markers(img, debug_prefix=debug_prefix)
    print(
        f"Corner markers: {corner_markers['found_count']}/4 "
        f"(candidates={corner_markers['candidate_count']})"
    )
    if corner_markers.get("debug_image_path"):
        print(f"Corner debug image: {corner_markers['debug_image_path']}")
    for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
        print(f"  {key}: {corner_markers['corners'][key]}")

    data, parts = _run_detection_pipeline(img, debug_prefix)

    preprocess_mode = "base"
    topview_applied = False
    page_h = img.shape[0] if img is not None else 0

    # Fallback cho ảnh khó: thử CLAHE rồi chọn kết quả có điểm bố cục tốt hơn.
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
    
    # In thống kê số box từng phần để theo dõi chất lượng detect.
    print(f"Part I boxes: {len(parts['part_i'])}")
    print(f"Part II boxes: {len(parts['part_ii'])}")
    print(f"Part III boxes: {len(parts['part_iii'])}")
    
    # Lấy các box còn lại sau khi đã tách Part I/II/III để tìm vùng SBD/Mã đề.
    part_box_set = set(id(box) for box in parts["all_parts"])
    remaining_boxes = [box for box in data["boxes"] if id(box) not in part_box_set]

    # Một số scan dính 2 bubble thành 1 contour rộng, cần tách trước khi nhóm.
    remaining_for_upper = _split_merged_boxes_for_grouping(
        remaining_boxes,
        split_wide=True,
        split_tall=False,
    )

    # Tách vùng ID phía trên thành SBD bên trái và Mã đề bên phải theo trục X.
    sbd_candidates, ma_de_candidates, split_x = _separate_upper_id_boxes(
        remaining_for_upper,
        parts["part_i"],
    )
    active_sbd_candidates = sbd_candidates
    active_ma_de_candidates = ma_de_candidates

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
    
    # Với Mã đề, có trường hợp dính 2 hàng theo chiều dọc thành box cao -> tách trước.
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

    def _evaluate_id_decode_quality(
        src_img: np.ndarray,
        data_local: Dict[str, object],
        sbd_rows: List[List[np.ndarray]],
        ma_de_rows: List[List[np.ndarray]],
    ) -> Dict[str, int]:
        # Chấm nhanh chất lượng ID: số cột có bubble hợp lệ sau decoder.
        gray_local = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        binary_local = data_local.get("binary")

        if sbd_rows:
            sbd_eval_local = evaluate_digit_rows_with_binary_fallback(
                gray_local,
                sbd_rows,
                expected_cols=6,
                binary_image=binary_local,
            )
            sbd_filled_local = int(_count_filled_digit_columns(sbd_eval_local))
        else:
            sbd_filled_local = 0

        if ma_de_rows:
            ma_de_eval_local = evaluate_digit_rows_with_binary_fallback(
                gray_local,
                ma_de_rows,
                expected_cols=3,
                binary_image=binary_local,
            )
            ma_de_filled_local = int(_count_filled_digit_columns(ma_de_eval_local))
        else:
            ma_de_filled_local = 0

        return {
            "sbd_filled": sbd_filled_local,
            "ma_de_filled": ma_de_filled_local,
        }

    base_id_quality = _evaluate_id_decode_quality(
        img,
        data,
        sobao_danh["sobao_danh_rows"],
        ma_de["ma_de_rows"],
    )
    current_id_quality = dict(base_id_quality)

    # Cho phép dùng topview khi một trong hai vùng ID bị thiếu hàng rõ rệt.
    affine_retry_threshold = 4
    no_valid_id_parts = (
        int(base_id_quality.get("sbd_filled", 0)) == 0
        and int(base_id_quality.get("ma_de_filled", 0)) == 0
    )
    topview_allowed_by_rows = (
        sobao_danh["row_count"] <= affine_retry_threshold
        or ma_de["row_count"] <= affine_retry_threshold
        or no_valid_id_parts
    )
    if topview_allowed_by_rows:
        affine_img, affine_matrix = _apply_affine_from_corner_markers(img, corner_markers.get("corners", {}))
        if affine_img is not None and affine_matrix is not None:
            data_affine, parts_affine = _run_detection_pipeline(affine_img, None)

            part_box_set_affine = set(id(box) for box in parts_affine["all_parts"])
            remaining_boxes_affine = [box for box in data_affine["boxes"] if id(box) not in part_box_set_affine]

            remaining_for_upper_affine = _split_merged_boxes_for_grouping(
                remaining_boxes_affine,
                split_wide=True,
                split_tall=False,
            )

            sbd_candidates_affine, ma_de_candidates_affine, split_x_affine = _separate_upper_id_boxes(
                remaining_for_upper_affine,
                parts_affine["part_i"],
            )

            sobao_danh_affine = detect_sobao_danh_boxes(
                sbd_candidates_affine,
                boxes_per_row=6,
                max_rows=10,
                row_tolerance=30,
                size_tolerance_ratio=0.35,
                debug=False,
            )

            remaining_for_ma_de_affine = _split_merged_boxes_for_grouping(
                ma_de_candidates_affine,
                split_wide=False,
                split_tall=True,
            )
            ma_de_affine = detect_ma_de_boxes(
                remaining_for_ma_de_affine,
                boxes_per_row=3,
                max_rows=10,
                row_tolerance=20,
                size_tolerance_ratio=0.3,
                debug=False,
            )

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

            base_sbd_filled = int(base_id_quality.get("sbd_filled", 0))
            base_ma_de_filled = int(base_id_quality.get("ma_de_filled", 0))
            affine_sbd_filled = int(affine_id_quality.get("sbd_filled", 0))
            affine_ma_de_filled = int(affine_id_quality.get("ma_de_filled", 0))

            force_topview_for_missing_id_parts = (
                int(sobao_danh["row_count"]) == 0 and int(ma_de["row_count"]) == 0
            )
            force_topview_for_invalid_id_parts = (
                no_valid_id_parts and new_quality_score >= old_quality_score
            )
            force_topview_for_single_invalid_side = (
                (base_sbd_filled == 0 and affine_sbd_filled > base_sbd_filled)
                or (base_ma_de_filled == 0 and affine_ma_de_filled > base_ma_de_filled)
            )

            # Guard: không áp topview nếu làm tụt mạnh Part II/III đang detect tốt ở ảnh gốc.
            base_part_ii_count = len(parts.get("part_ii", []))
            base_part_iii_count = len(parts.get("part_iii", []))
            affine_part_ii_count = len(parts_affine.get("part_ii", []))
            affine_part_iii_count = len(parts_affine.get("part_iii", []))

            topview_preserves_sections = True
            section_guard_reasons: List[str] = []

            if base_part_ii_count >= 6:
                min_part_ii_after = max(4, int(round(0.70 * float(base_part_ii_count))))
                if affine_part_ii_count < min_part_ii_after:
                    topview_preserves_sections = False
                    section_guard_reasons.append(
                        f"PartII {base_part_ii_count}->{affine_part_ii_count}"
                    )

            if base_part_iii_count >= 5:
                min_part_iii_after = max(4, int(round(0.70 * float(base_part_iii_count))))
                if affine_part_iii_count < min_part_iii_after:
                    topview_preserves_sections = False
                    section_guard_reasons.append(
                        f"PartIII {base_part_iii_count}->{affine_part_iii_count}"
                    )

            topview_candidate_improved = (
                new_score > old_score
                or new_quality_score > old_quality_score
                or force_topview_for_missing_id_parts
                or force_topview_for_invalid_id_parts
                or force_topview_for_single_invalid_side
            )

            if topview_candidate_improved and topview_preserves_sections:
                img = affine_img
                data = data_affine
                parts = parts_affine
                split_x = split_x_affine
                sobao_danh = sobao_danh_affine
                ma_de = ma_de_affine
                active_sbd_candidates = sbd_candidates_affine
                active_ma_de_candidates = ma_de_candidates_affine
                current_id_quality = dict(affine_id_quality)
                preprocess_mode = f"{preprocess_mode}+topview-id"
                topview_applied = True
                if new_score > old_score or new_quality_score > old_quality_score:
                    print(
                        f"[Topview] Retry improved ID detection: score {old_score} -> {new_score}, "
                        f"filled {old_quality_score} -> {new_quality_score} "
                        f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                    )
                elif force_topview_for_missing_id_parts:
                    print(
                        f"[Topview] Forced for missing ID parts: score {old_score} -> {new_score} "
                        f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                    )
                elif force_topview_for_single_invalid_side:
                    print(
                        f"[Topview] Forced for single-side invalid ID quality: score {old_score} -> {new_score}, "
                        f"filled {old_quality_score} -> {new_quality_score} "
                        f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                    )
                else:
                    print(
                        f"[Topview] Forced for invalid ID quality: score {old_score} -> {new_score}, "
                        f"filled {old_quality_score} -> {new_quality_score} "
                        f"(SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']})"
                    )
            elif topview_candidate_improved and not topview_preserves_sections:
                print(
                    "[Topview] Retry blocked by section guard: "
                    + ", ".join(section_guard_reasons)
                )
            else:
                print(
                    f"[Topview] Retry no improvement: score {old_score} -> {new_score}, "
                    f"filled {old_quality_score} -> {new_quality_score} "
                    f"(SBD={sobao_danh_affine['row_count']}, MaDe={ma_de_affine['row_count']})"
                )
        else:
            print("[Topview] Retry skipped: not enough reliable corner markers")
    else:
        print(
            "[Topview] Retry blocked by row gate: "
            f"SBD={sobao_danh['row_count']}, MaDe={ma_de['row_count']} (>4 on both sides)"
        )

    # Cấu hình fallback ID (không cần truyền argument CLI).
    # Chế độ fixed sẽ đặt lưới theo vị trí tuyệt đối trên ảnh.
    id_fallback_row_threshold = 4
    id_grid_top_ratio = 0.0725
    id_grid_row_step_ratio = 0.0211
    id_sbd_x_range_ratio_fixed = (0.74, 0.865)
    id_made_x_range_ratio_fixed = (0.90, 0.96)

    sbd_x_range_ratio, made_x_range_ratio = (
        id_sbd_x_range_ratio_fixed,
        id_made_x_range_ratio_fixed,
    )

    sobao_rows_are_synthetic = False
    ma_de_rows_are_synthetic = False

    def _estimate_reference_positions_from_rows(rows: List[List[np.ndarray]], target_rows: int = 10) -> List[int]:
        # Suy ra 10 mốc Y từ các hàng detect được và neo gần vị trí top mặc định.
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
        expected_top = int(round(float(id_grid_top_ratio) * float(img.shape[0])))
        expected_step = max(6, int(round(float(id_grid_row_step_ratio) * float(img.shape[0]))))

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
        # Ước lượng miền X theo candidate thật để giảm lệch grid fixed ở các scan khó.
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
        img.shape[1],
        id_sbd_x_range_ratio_fixed,
        min_boxes=18,
        min_width_ratio=0.08,
        max_width_ratio=0.22,
    )
    made_x_range_ratio = _estimate_id_x_range_ratio_from_candidates(
        active_ma_de_candidates,
        img.shape[1],
        id_made_x_range_ratio_fixed,
        min_boxes=9,
        min_width_ratio=0.04,
        max_width_ratio=0.14,
    )

    # Debug topview cho ID chỉ chạy khi bật cờ vì khá tốn thời gian.
    if topview_debug and topview_allowed_by_rows and (
        sobao_danh["row_count"] <= id_fallback_row_threshold
        or ma_de["row_count"] <= id_fallback_row_threshold
    ):
        topview_img, _ = _apply_affine_from_corner_markers(img_original, corner_markers.get("corners", {}))
        if topview_img is not None:
            topview_sbd_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=topview_img.shape[:2],
                cols=6,
                rows=10,
                x_range_ratio=sbd_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )
            topview_made_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=topview_img.shape[:2],
                cols=3,
                rows=10,
                x_range_ratio=made_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )

            gray_topview = cv2.cvtColor(topview_img, cv2.COLOR_BGR2GRAY)
            topview_sbd_digits = evaluate_digit_rows_mean_darkness(
                gray_topview,
                topview_sbd_rows,
                expected_cols=6,
            )
            topview_made_digits = evaluate_digit_rows_mean_darkness(
                gray_topview,
                topview_made_rows,
                expected_cols=3,
            )

            print("\n=== Topview Mean-Darkness Decode ===")
            _print_digit_darkness_summary("Topview SoBaoDanh", topview_sbd_digits)
            _print_digit_darkness_summary("Topview MaDe", topview_made_digits)

            topview_debug = topview_img.copy()
            _draw_rows_contours(topview_debug, topview_sbd_rows, (255, 128, 0), thickness=1)
            _draw_rows_contours(topview_debug, topview_made_rows, (255, 255, 0), thickness=1)

            topview_debug = draw_digit_darkness_overlay(
                topview_debug,
                topview_sbd_digits,
                color=(0, 220, 255),
                alpha=0.40,
            )
            topview_debug = draw_digit_darkness_overlay(
                topview_debug,
                topview_made_digits,
                color=(0, 255, 255),
                alpha=0.40,
            )

            cv2.putText(
                topview_debug,
                f"Topview SBD: {topview_sbd_digits.get('decoded', '')}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                topview_debug,
                f"Topview MaDe: {topview_made_digits.get('decoded', '')}",
                (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            topview_raw_path = f"{debug_prefix}_topview.jpg"
            topview_id_path = f"{debug_prefix}_topview_id_mean_darkness.jpg"
            cv2.imwrite(topview_raw_path, topview_img)
            cv2.imwrite(topview_id_path, topview_debug)
            print(f"Topview image saved to {topview_raw_path}")
            print(f"Topview ID debug saved to {topview_id_path}")
        else:
            print("Topview debug skipped: affine transform unavailable (missing reliable corners)")

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

    if not allow_ratio_id_fallback and (
        sbd_needs_fallback
        or ma_de_needs_fallback
    ):
        print(
            "[Fallback] Skipped synthetic ID ratio-grid: topview not applied "
            f"(sbd_filled={current_id_quality.get('sbd_filled', 0)}, "
            f"ma_de_filled={current_id_quality.get('ma_de_filled', 0)})"
        )
    # Nếu detect quá ít hàng ID, dựng lưới tổng hợp 10 hàng theo vị trí cố định trên ảnh.
    if allow_ratio_id_fallback and sbd_needs_fallback:
        sbd_fallback_reason = (
            f"rows <= {fallback_threshold}"
            if sobao_danh["row_count"] <= fallback_threshold
            else "filled cols = 0"
        )
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
            sobao_rows = _build_synthetic_id_rows_from_reference_positions(
                image_shape=img.shape[:2],
                cols=6,
                x_range_ratio=sbd_x_range_ratio,
                reference_positions=reference_positions,
                row_height=ma_de_row_h,
            )
            fallback_mode = "from-ma_de"
        else:
            sobao_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=img.shape[:2],
                cols=6,
                rows=10,
                x_range_ratio=sbd_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )
            fallback_mode = "fixed"

        if sobao_rows:
            sobao_danh["sobao_danh_rows"] = sobao_rows
            sobao_danh["sobao_danh"] = [b for row in sobao_rows for b in row]
            sobao_danh["row_count"] = len(sobao_rows)
            sobao_rows_are_synthetic = True
            print(
                f"[Fallback] SoBaoDanh {sbd_fallback_reason}, "
                f"drew synthetic 6x10 grid (mode={fallback_mode})"
            )

    if allow_ratio_id_fallback and ma_de_needs_fallback:
        ma_de_fallback_reason = (
            f"rows <= {fallback_threshold}"
            if ma_de["row_count"] <= fallback_threshold
            else "filled cols = 0"
        )
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
            ma_de_rows = _build_synthetic_id_rows_from_reference_positions(
                image_shape=img.shape[:2],
                cols=3,
                x_range_ratio=made_x_range_ratio,
                reference_positions=reference_positions,
                row_height=sbd_row_h,
            )
            fallback_mode = "from-sobao"
        else:
            ma_de_rows = _build_synthetic_id_rows_fixed_image_position(
                image_shape=img.shape[:2],
                cols=3,
                rows=10,
                x_range_ratio=made_x_range_ratio,
                top_y_ratio=id_grid_top_ratio,
                row_step_ratio=id_grid_row_step_ratio,
            )
            fallback_mode = "fixed"

        if ma_de_rows:
            ma_de["ma_de_rows"] = ma_de_rows
            ma_de["ma_de"] = [b for row in ma_de_rows for b in row]
            ma_de["row_count"] = len(ma_de_rows)
            ma_de_rows_are_synthetic = True
            print(
                f"[Fallback] MaDe {ma_de_fallback_reason}, "
                f"drew synthetic 3x10 grid (mode={fallback_mode})"
            )

    # Bù thiếu MaDe lên đủ 10 hàng bằng cách neo theo trục Y của SoBaoDanh.
    # Ý tưởng: dùng hình học cột MaDe đã phát hiện tốt để nội suy những hàng bị mất.
    if ma_de["row_count"] < 10 and ma_de["ma_de_rows"]:
        if sobao_danh["sobao_danh_rows"] and not sobao_rows_are_synthetic:
            ref_positions = [
                int(round(float(np.mean([cv2.boundingRect(box)[1] for box in row]))))
                for row in sobao_danh["sobao_danh_rows"][:10]
                if row
            ]
        else:
            ref_positions = _estimate_reference_positions_from_rows(ma_de["ma_de_rows"], target_rows=10)

        ma_de_rect_rows = []
        for row in ma_de["ma_de_rows"]:
            rects = [cv2.boundingRect(box) for box in row]
            rects = sorted(rects, key=lambda r: r[0])
            if len(rects) == 3:
                ma_de_rect_rows.append(rects)

        if len(ref_positions) == 10 and ma_de_rect_rows:
            # Dựng template hình học ổn định cho 3 cột Mã đề.
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

                row_polys = [_rect_to_poly(rx, ry, rw, rh) for rx, ry, rw, rh in rects]
                aligned_rows[chosen_idx] = row_polys
                used_ref_indices.add(chosen_idx)

            # Nội suy các hàng còn thiếu bằng box tổng hợp tại các mốc Y tham chiếu.
            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx]
                synthetic_row: List[np.ndarray] = []
                for c in range(3):
                    x = col_x[c]
                    w = max(1, col_w[c])
                    synthetic_row.append(_rect_to_poly(x, y_ref, w, row_h))
                aligned_rows[idx] = synthetic_row

            ma_de_completed_rows = [row for row in aligned_rows if row is not None]
            if len(ma_de_completed_rows) == 10:
                ma_de["ma_de_rows"] = ma_de_completed_rows
                ma_de["ma_de"] = [box for row in ma_de_completed_rows for box in row]
                ma_de["row_count"] = 10
                if len(ma_de_rect_rows) < 10:
                    ma_de_rows_are_synthetic = True

    # Bù thiếu SoBaoDanh lên đủ 10 hàng bằng cách neo theo trục Y của Mã đề.
    # Hữu ích cho các mẫu có SBD mất nhiều hàng nhưng Mã đề vẫn đủ 10 hàng (ví dụ 0018).
    if (
        sobao_danh["row_count"] < 10
        and sobao_danh["sobao_danh_rows"]
        and ma_de["ma_de_rows"]
        and not ma_de_rows_are_synthetic
    ):
        ref_positions = [
            int(round(float(np.mean([cv2.boundingRect(box)[1] for box in row]))))
            for row in ma_de["ma_de_rows"][:10]
            if row
        ]

        sobao_rect_rows = []
        for row in sobao_danh["sobao_danh_rows"]:
            rects = [cv2.boundingRect(box) for box in row]
            rects = sorted(rects, key=lambda r: r[0])
            if len(rects) == 6:
                sobao_rect_rows.append(rects)

        if len(ref_positions) == 10 and sobao_rect_rows:
            # Dựng template hình học ổn định cho 6 cột SoBaoDanh.
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

                row_polys = [_rect_to_poly(rx, ry, rw, rh) for rx, ry, rw, rh in rects]
                aligned_rows[chosen_idx] = row_polys
                used_ref_indices.add(chosen_idx)
                matched_rows.append((chosen_idx, rects, row_y))

            # Ước lượng xu hướng x theo y cho từng cột để giảm lệch phối cảnh ở hàng nội suy.
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

                # Chặn miền dự đoán để tránh x trôi quá xa khi ngoại suy hàng trên cùng.
                margin = max(8.0, float(col_w[col_idx]) * 2.0)
                x_min = float(np.min(xs)) - margin
                x_max = float(np.max(xs)) + margin
                pred = float(np.clip(pred, x_min, x_max))
                return int(round(pred))

            # Ước lượng lệch Y giữa 2 cụm SBD/Mã đề từ các hàng đã match.
            y_bias = 0
            if matched_rows:
                y_deltas = [int(row_y - ref_positions[idx]) for idx, _, row_y in matched_rows]
                y_bias = int(round(float(np.median(y_deltas))))

            # Bù lệch X theo từng cột để giảm drift khi ngoại suy sang vùng thiếu hàng.
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

            # Nội suy các hàng còn thiếu bằng box tổng hợp tại các mốc Y tham chiếu.
            for idx in range(10):
                if aligned_rows[idx] is not None:
                    continue
                y_ref = ref_positions[idx] + y_bias
                synthetic_row: List[np.ndarray] = []
                for c in range(6):
                    x = _predict_col_x(c, float(y_ref)) + col_x_bias[c]
                    w = max(1, col_w[c])
                    synthetic_row.append(_rect_to_poly(x, y_ref, w, row_h))
                aligned_rows[idx] = synthetic_row

            sobao_completed_rows = [row for row in aligned_rows if row is not None]
            if len(sobao_completed_rows) == 10:
                sobao_danh["sobao_danh_rows"] = sobao_completed_rows
                sobao_danh["sobao_danh"] = [box for row in sobao_completed_rows for box in row]
                sobao_danh["row_count"] = 10
                if len(sobao_rect_rows) < 10:
                    sobao_rows_are_synthetic = True

    print(f"MaDe rows: {ma_de['row_count']}")
    print(f"MaDe boxes: {len(ma_de['ma_de'])}")
    print(f"Upper split X: {split_x:.1f}")
    print(f"SoBaoDanh rows (final): {sobao_danh['row_count']}")
    print(f"MaDe rows (final): {ma_de['row_count']}")
    
    # Căn thẳng theo trục Y và ngoại suy để cả SBD/Mã đề đều đủ 10 hàng logic.
    expected_id_top_y = int(round(float(id_grid_top_ratio) * float(img.shape[0])))
    expected_id_row_step = max(6, int(round(float(id_grid_row_step_ratio) * float(img.shape[0]))))
    combined_results = {
        "sobao_danh_rows": sobao_danh["sobao_danh_rows"],
        "ma_de_rows": ma_de["ma_de_rows"],
        "sobao_is_synthetic": sobao_rows_are_synthetic,
        "ma_de_is_synthetic": ma_de_rows_are_synthetic,
    }
    extrapolated = extrapolate_missing_rows(
        combined_results,
        target_rows=10,
        expected_top_y=expected_id_top_y,
        expected_row_step=expected_id_row_step,
        debug=False,
    )
    
    print(f"\nExtrapolation Summary:")
    print(f"  SoBaoDanh: {extrapolated['sobao_detected_count']}/10 detected, {extrapolated['sobao_missing_count']} missing")
    print(f"  MaDe: {extrapolated['ma_de_detected_count']}/10 detected, {extrapolated['ma_de_missing_count']} missing")
    
    # In vị trí các hàng Mã đề còn thiếu để tiện debug theo Y.
    aligned_ma_de = extrapolated.get("ma_de_rows_aligned", [])
    reference_positions = extrapolated.get("reference_positions", [])
    
    if aligned_ma_de and None in aligned_ma_de:
        print(f"\n  Missing MaDe rows at positions:")
        for idx, row in enumerate(aligned_ma_de):
            if row is None:
                y_pos = reference_positions[idx] if idx < len(reference_positions) else "?"
                print(f"    Row {idx + 1}: Y ≈ {y_pos}")
    
    gray_for_digits = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobao_rows_aligned = extrapolated.get("sobao_danh_rows_aligned", sobao_danh["sobao_danh_rows"])
    ma_de_rows_aligned = extrapolated.get("ma_de_rows_aligned", ma_de["ma_de_rows"])

    binary_for_digits = data.get("binary")

    sbd_digits = evaluate_digit_rows_with_binary_fallback(
        gray_for_digits,
        sobao_rows_aligned,
        expected_cols=6,
        binary_image=binary_for_digits,
    )
    made_digits = evaluate_digit_rows_with_binary_fallback(
        gray_for_digits,
        ma_de_rows_aligned,
        expected_cols=3,
        binary_image=binary_for_digits,
    )

    print("\n=== Mean-Darkness Digit Decode ===")
    _print_digit_darkness_summary("SoBaoDanh", sbd_digits)
    _print_digit_darkness_summary("MaDe", made_digits)

    # Vẽ viền Part I/II/III bằng màu khác nhau để kiểm tra tách phần.
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1.5, 2
    
    # Cấu hình màu và nhãn cho từng phần.
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
    
    # Vẽ contour cho vùng SBD/Mã đề đã phát hiện thật (không vẽ hàng nội suy ở ảnh parts).
    sobao_color = (255, 128, 0)  # Màu cam cho SoBaoDanh
    _draw_rows_contours(overlay, sobao_danh["sobao_danh_rows"], sobao_color, thickness=2)
    
    ma_de_color = (255, 255, 0)  # Màu cyan cho MaDe
    _draw_rows_contours(overlay, ma_de["ma_de_rows"], ma_de_color, thickness=2)
    
    # Ở ảnh parts, chỉ hiển thị box phát hiện thật; không hiển thị box nội suy.
    
    cv2.imwrite(f"{debug_prefix}_parts.jpg", overlay)
    print(f"Parts visualization saved to {debug_prefix}_parts.jpg")
    
    # Vẽ grid toàn bộ phần trắc nghiệm lên cùng một ảnh để debug/đối chiếu nhanh.
    print(f"\n=== Drawing grids on all parts ===")
    combined_grid_image = img.copy()
    
    binary_threshold = data.get("binary_score", data.get("binary"))
    if binary_threshold is not None:
        print("\nUsing binary threshold image for fill-ratio classification")

    # Phần I: lưới 4x10 chuẩn, ưu tiên giữ sát khung thật để giảm lệch ô.
    part_i_evals: List[Dict[str, object]] = []
    if parts["part_i"]:
        print(f"\n=== Part I: 4x10 grid (18.5% left, 3.0% right, 10% y) ===")
        grid_result = extract_grid_from_boxes(
            combined_grid_image,
            boxes=parts["part_i"],
            grid_cols=4,
            grid_rows=10,
            start_offset_ratio_x=0.185,
            start_offset_ratio_y=0.1,
            end_offset_ratio_x=0.03,
            end_offset_ratio_y=0.015,
            grid_color=(0, 255, 0),  # Màu xanh lá
            grid_thickness=1,
        )
        combined_grid_image = grid_result["image_with_grid"]

        _print_grid_info(grid_result["grid_info"])
        part_i_evals = _evaluate_section_fill(
            section_name="Part I",
            binary_threshold=binary_threshold,
            grid_info=grid_result["grid_info"],
            fill_ratio_thresh=0.55,
            inner_margin_ratio=0.01,
            circle_radius_scale=0.6,
            circle_border_exclude_ratio=0.12,
        )

        if part_i_evals:
            part_i_evals_refined = suppress_false_positive_by_relative_dominance(
                part_i_evals,
                group_keys=("box_idx", "row"),
                ratio_key="fill_ratio",
                min_dominant_ratio=0.52,
                min_gap=0.12,
                min_ratio_scale=1.35,
            )
            suppressed_count = sum(
                1
                for item in part_i_evals_refined
                if bool(item.get("suppressed_by_dominance", False))
            )
            if suppressed_count > 0:
                print(
                    f"[Part I] Dominance filter suppressed {suppressed_count} false-positive cells"
                )
                _print_fill_summary("Part I (after dominance)", part_i_evals_refined)
            part_i_evals = part_i_evals_refined
    
    # Phần II: mỗi cụm có offset xen kẽ trái/phải để bám đúng bố cục phiếu.
    part_ii_evals: List[Dict[str, object]] = []
    if parts["part_ii"]:
        print(f"\n=== Part II: 2x4 grid (alternating offsets, 30% y, -5% bottom) ===")
        # Part II gồm 8 box, dùng offset xen kẽ để bám đúng vị trí bubble thực tế.
        part_ii_count = len(parts["part_ii"])
        offset_ratios = [
            (0.3, 0.33) if (box_idx % 2 == 0) else (0.0, 0.33)
            for box_idx in range(part_ii_count)
        ]
        end_offset_x = [0.0] * part_ii_count  # 0% từ biên phải cho tất cả box
        end_offset_y = [0.03] * part_ii_count  # 3% từ đáy để tránh dính viền đậm
        
        grid_result_ii = extract_grid_from_boxes_variable_offsets(
            combined_grid_image,
            boxes=parts["part_ii"],
            grid_cols=2,
            grid_rows=4,
            start_offset_ratios=offset_ratios,
            end_offset_ratios_x=end_offset_x,
            end_offset_ratios_y=end_offset_y,
            grid_color=(0, 165, 255),  # Màu cam
            grid_thickness=1,
        )
        combined_grid_image = grid_result_ii["image_with_grid"]

        _print_grid_info(
            grid_result_ii["grid_info"],
            detail_formatter=lambda info: (
                f"offset {info['offset_ratios']}, end_y -{info['end_offset_y'] * 100:.0f}%"
            ),
        )
        part_ii_evals = _evaluate_section_fill(
            section_name="Part II",
            binary_threshold=binary_threshold,
            grid_info=grid_result_ii["grid_info"],
            fill_ratio_thresh=0.52,
            inner_margin_ratio=0.01,
            circle_radius_scale=0.6,
            # Part II dễ bị false-positive do viền bubble đậm; loại bớt viền để ưu tiên vùng lõi tô.
            circle_border_exclude_ratio=0.18,
        )

        if part_ii_evals:
            part_ii_evals_refined = suppress_false_positive_by_relative_dominance(
                part_ii_evals,
                group_keys=("box_idx", "row"),
                ratio_key="fill_ratio",
                min_dominant_ratio=0.52,
                min_gap=0.10,
                min_ratio_scale=1.25,
            )
            suppressed_count = sum(
                1
                for item in part_ii_evals_refined
                if bool(item.get("suppressed_by_dominance", False))
            )
            if suppressed_count > 0:
                print(
                    f"[Part II] Dominance filter suppressed {suppressed_count} false-positive cells"
                )
                _print_fill_summary("Part II (after dominance)", part_ii_evals_refined)
            part_ii_evals = part_ii_evals_refined
    
    # Phần III: lưới 4x12 với pattern hàng đầu đặc biệt theo mẫu đề.
    part_iii_evals: List[Dict[str, object]] = []
    if parts["part_iii"]:
        print(f"\n=== Part III: 4x12 grid with custom pattern (20% x, 10% y) ===")
        # Pattern Part III: 2 hàng đầu đặc biệt, các hàng sau dùng đủ 4 cột.
        custom_pattern = [[0], [1, 2]] + [[0, 1, 2, 3] for _ in range(10)]
        
        grid_result_iii = extract_grid_from_boxes_custom_pattern(
            combined_grid_image,
            boxes=parts["part_iii"],
            grid_cols=4,
            grid_rows=12,
            start_offset_ratio_x=0.22,
            start_offset_ratio_y=0.16,
            end_offset_ratio_x=0.1,
            end_offset_ratio_y=0.015,
            grid_color=(255, 0, 0),  # BGR: hiển thị màu đỏ
            grid_thickness=1,
            row_col_patterns=custom_pattern,
        )
        combined_grid_image = grid_result_iii["image_with_grid"]

        _print_grid_info(
            grid_result_iii["grid_info"],
            detail_formatter=lambda info: f"pattern={info['pattern'][:2]}... (12 rows total)",
        )
        part_iii_evals = _evaluate_section_fill(
            section_name="Part III",
            binary_threshold=binary_threshold,
            grid_info=grid_result_iii["grid_info"],
            fill_ratio_thresh=0.52,
            inner_margin_ratio=0.05,
            circle_radius_scale=0.6,
            circle_border_exclude_ratio=0.12,
        )

        if part_iii_evals:
            part_iii_evals_refined = suppress_false_positive_by_relative_dominance(
                part_iii_evals,
                group_keys=("box_idx", "col"),
                ratio_key="fill_ratio",
                min_dominant_ratio=0.52,
                min_gap=0.10,
                min_ratio_scale=1.20,
            )
            suppressed_count = sum(
                1
                for item in part_iii_evals_refined
                if bool(item.get("suppressed_by_dominance", False))
            )
            if suppressed_count > 0:
                print(
                    f"[Part III] Dominance filter suppressed {suppressed_count} false-positive cells"
                )
                _print_fill_summary("Part III (after dominance)", part_iii_evals_refined)
            part_iii_evals = part_iii_evals_refined

    # Vẽ lưới contour của SBD/Mã đề trên ảnh tổng hợp để đối chiếu toàn cục.
    sobao_grid_color = (255, 128, 0)  # Cam
    ma_de_grid_color = (255, 255, 0)  # Cyan

    if sobao_danh["sobao_danh_rows"]:
        print(f"\n=== SoBaoDanh: drawing detected box grid ===")
        sobao_count = _draw_rows_contours(
            combined_grid_image,
            sobao_danh["sobao_danh_rows"],
            sobao_grid_color,
            thickness=1,
        )
        print(f"Grid drawn on {sobao_count} SoBaoDanh boxes")

    if ma_de["ma_de_rows"]:
        print(f"\n=== MaDe: drawing detected box grid ===")
        ma_de_count = _draw_rows_contours(
            combined_grid_image,
            ma_de["ma_de_rows"],
            ma_de_grid_color,
            thickness=1,
        )
        print(f"Grid drawn on {ma_de_count} MaDe boxes")

    combined_grid_image = draw_digit_darkness_overlay(
        combined_grid_image,
        sbd_digits,
        color=(0, 220, 255),
        alpha=0.40,
    )
    combined_grid_image = draw_digit_darkness_overlay(
        combined_grid_image,
        made_digits,
        color=(0, 255, 255),
        alpha=0.40,
    )

    all_evals = part_i_evals + part_ii_evals + part_iii_evals
    if all_evals:
        combined_grid_image = draw_filled_cells_overlay(
            combined_grid_image,
            all_evals,
            color=(0, 255, 0),
            alpha=0.35,
        )

        if binary_threshold is not None:
            binary_fillratio_path = f"{debug_prefix}_binary_fillratio_grid.jpg"
            draw_binary_fillratio_debug(binary_threshold, all_evals, binary_fillratio_path)
            print(f"✓ Binary fill-ratio debug image saved to: {binary_fillratio_path}")
    
    # Lưu ảnh tổng hợp cuối: gồm lưới, overlay ô tô và kết quả debug.
    combined_grid_path = f"{debug_prefix}_all_parts_with_grid.jpg"
    cv2.imwrite(combined_grid_path, combined_grid_image)
    print(f"\n✓ Combined grid image saved to: {combined_grid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát hiện và vẽ lưới đáp án từ ảnh phiếu.")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Định danh ảnh, ví dụ: 0015, 31, PhieuQG.0031, hoặc PhieuQG/PhieuQG.0031.jpg",
    )
    parser.add_argument(
        "--image",
        dest="image_opt",
        default=None,
        help="Định danh ảnh (chấp nhận cùng định dạng như tham số vị trí image).",
    )
    parser.add_argument(
        "--topview-debug",
        action="store_true",
        help="Bật nhánh debug topview cho ID (tăng thời gian xử lý).",
    )
    args = parser.parse_args()
    _demo(image_arg=args.image_opt or args.image, topview_debug=bool(args.topview_debug))
