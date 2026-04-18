# Tài Liệu Kỹ Thuật detect.py

Tài liệu này mô tả chi tiết, theo đúng thứ tự thực thi, toàn bộ flow xử lý của file `detect.py`.
Mục tiêu là để bạn có thể:

- Hiểu chính xác pipeline đang làm gì ở từng bước.
- Biết vì sao có các nhánh fallback trong code.
- Dễ debug khi kết quả nhận dạng lệch.
- Nắm rõ tham số nào ảnh hưởng mạnh đến độ chính xác.

## 1) Mục tiêu của pipeline

`detect.py` xử lý ảnh phiếu trắc nghiệm theo hướng **morphology + hình học contour + suy luận theo bố cục phiếu** để:

- Tách 3 vùng trắc nghiệm chính: `Part I`, `Part II`, `Part III`.
- Tách vùng trên của phiếu thành `SoBaoDanh` (6 cột, 10 hàng) và `MaDe` (3 cột, 10 hàng).
- Chấm ô tô ở Part I/II/III bằng `fill-ratio` trên mask vòng tròn.
- Giải mã chữ số SBD/Mã đề bằng `mean darkness` theo cột.
- Xuất ảnh debug trực quan để kiểm tra từng pha.

## 2) Tóm tắt flow end-to-end

Luồng chính nằm trong hàm `_demo()` của `detect.py`.

```text
Input image arg
  -> normalize image stem
  -> load image from PhieuQG/
  -> detect_boxes_from_morph_lines()
  -> group_boxes_into_parts()  [Part I/II/III]
  -> fallback CLAHE (nếu detect yếu)
  -> tách box còn lại vùng trên thành SBD + MaDe
  -> detect_sobao_danh_boxes()
  -> detect_ma_de_boxes()
  -> extrapolate_missing_rows()
  -> evaluate_digit_rows_mean_darkness() [SBD, MaDe]
  -> extract_grid_from_boxes*() cho Part I/II/III
  -> evaluate_grid_fill_from_binary() cho Part I/II/III
  -> vẽ overlay + lưu ảnh debug output/detection/
```

## 3) Phân tích chi tiết từng bước xử lý

### 3.1 Chuẩn hóa input ảnh

Hàm: `_normalize_image_stem(image_arg)`

- Nhận nhiều kiểu input: `0030`, `PhieuQG.0030`, `PhieuQG/PhieuQG.0030.jpg`, ...
- Chuẩn hóa về dạng chuẩn: `PhieuQG.XXXX`.
- Mặc định fallback: `PhieuQG.0015` nếu không truyền tham số.

Sau đó `_demo()` thử đọc lần lượt:

- `PhieuQG/<stem>.jpg`
- `PhieuQG/<stem>.jpeg`
- `PhieuQG/<stem>.png`

Nếu không đọc được sẽ raise `FileNotFoundError` với danh sách path đã thử.

### 3.2 Trích xuất mạng line và box kín (morphology)

Hàm chính: `detect_boxes_from_morph_lines()`

#### Bước nội bộ cốt lõi

1. `detect_grid_points()`:
   - Chuyển xám + Gaussian blur.
   - Adaptive threshold (`THRESH_BINARY_INV`).
   - Morphology để tách line dọc/ngang bằng kernel tỷ lệ theo kích thước ảnh.
   - Lấy giao điểm line (`bitwise_and`) và lọc component nhỏ.

2. `_filter_line_components_by_length()`:
   - Loại line quá ngắn theo `min_line_length` để giảm nhiễu.

3. `_align_vertical_lengths_by_row()` (tùy chọn):
   - Căn line dọc trong cùng hàng về chiều dài gần nhau để ổn định hơn khi đóng kín ô.

4. Tạo vùng kín:
   - Hợp line dọc + ngang.
   - Morph close để bịt khe hở.
   - Flood-fill nền ngoài rồi lấy phần kín bên trong.

5. Trích contour box:
   - Lọc theo diện tích, chiều rộng, chiều cao.
   - `approxPolyDP` để có polygon ổn định.
   - Sắp theo `(y, x)` để output có tính lặp.

#### Output của bước này

- `boxes`: danh sách contour box kín.
- Ảnh debug morphology: `vertical`, `horizontal`, `lines`, `boxes_overlay`.

### 3.3 Gom box thành Part I/II/III

Hàm: `group_boxes_into_parts()`

Đây là bước quan trọng nhất về logic bố cục:

1. Giữ các container lớn (lọc theo percentile area) để tránh trộn bubble nhỏ.
2. Khử trùng contour chồng nhau bằng IoU cao.
3. Gom theo hàng (trục Y) với tolerance thích nghi theo chiều cao box.
4. Nhận diện từng phần theo vị trí + hình học:
   - `Part I`: 4 box
   - `Part II`: ưu tiên 8 box; fallback từ 4 box gộp (tách đôi trái/phải) hoặc tái tạo từ box con
   - `Part III`: 6 box 

Pipeline này có nhiều nhánh recovery để giữ ổn định khi contour bị dính/mất.

### 3.4 Fallback CLAHE khi detect phần chính chưa đủ

Trong `_demo()`:

- Nếu `Part II < 8` hoặc `Part III < 6`, hệ thống thử ảnh CLAHE.
- Chạy lại detect đầy đủ trên ảnh CLAHE.
- So sánh điểm heuristic bố cục (`_parts_score`) giữa bản gốc và CLAHE.
- Chọn kết quả tốt hơn.

Điểm lợi: cứu được nhiều ảnh scan mờ mà không ép luôn dùng CLAHE cho mọi ảnh.

### 3.5 Tách vùng trên phiếu thành SBD và Mã đề

Sau khi có Part I/II/III:

1. Lấy `remaining_boxes` (box chưa thuộc 3 part).
2. `_split_merged_boxes_for_grouping()` để tách box dính ngang trước khi tách SBD/Mã đề.
3. `_separate_upper_id_boxes()` chia danh sách box theo trục X thành:
   - Nhóm SBD (bên trái).
   - Nhóm Mã đề (bên phải).

### 3.6 Gom hàng SBD/Mã đề và ngoại suy hàng thiếu

#### Số báo danh

Hàm: `detect_sobao_danh_boxes()`

- Kỳ vọng: mỗi hàng 6 box, tối đa 10 hàng.
- Gom theo hàng Y, kiểm tra đồng đều kích thước.
- Có logic cứu row bị dính (5 box nhưng 1 box quá rộng).
- Có nhánh sửa hình học hàng đầu nếu méo/bị chồng box.

#### Mã đề

Hàm: `detect_ma_de_boxes()`

- Kỳ vọng: mỗi hàng 3 box, tối đa 10 hàng.
- Gom nhóm tương tự SBD nhưng tham số phù hợp Mã đề.

#### Bù thiếu hàng

Hàm: `extrapolate_missing_rows()`

- Căn SBD/Mã đề theo trục Y tham chiếu.
- Tạo cấu trúc aligned đủ 10 vị trí (có thể chứa `None`).
- Trả thống kê thiếu/phát hiện theo từng vùng.

Ngoài ra trong `_demo()` còn có một nhánh bổ sung để **dựng lại 10 hàng Mã đề** từ template hình học nếu Mã đề thiếu hàng nhưng SBD đầy đủ mốc Y.

### 3.7 Giải mã chữ số SBD/Mã đề bằng mean darkness

Hàm: `evaluate_digit_rows_mean_darkness()`

Với mỗi cột:

1. Tính `mean_darkness` trong vòng tròn trung tâm từng hàng.
2. Chọn hàng tối nhất làm ứng viên.
3. Áp điều kiện độ chênh với hạng 2 và median để tránh false positive.
4. Nếu đạt điều kiện thì cột đó có digit, ngược lại trả `?`.

Kết quả:

- `decoded`: chuỗi số (ví dụ `012345` hoặc `01?3?5`).
- `column_decisions`: metadata quyết định từng cột.

### 3.8 Chia lưới và chấm Part I/II/III

Trong `_demo()`:

- `Part I`: `extract_grid_from_boxes()` với lưới `4 x 10`.
- `Part II`: `extract_grid_from_boxes_variable_offsets()` với lưới `2 x 4`, offset xen kẽ từng box.
- `Part III`: `extract_grid_from_boxes_custom_pattern()` với lưới `4 x 12` và pattern hàng đặc thù.

Sau đó `_evaluate_section_fill()` gọi `evaluate_grid_fill_from_binary()` với:

- `mask_mode="hough-circle"`.
- `fill_ratio_thresh=0.54`.
- Bán kính và viền loại trừ để bám sát bubble tròn thật.

Cuối cùng hệ thống vẽ overlay ô tô (`draw_filled_cells_overlay`) và ảnh fill-ratio (`draw_binary_fillratio_debug`).

### 3.9 Xuất ảnh debug

Thư mục đầu ra: `output/detection/`

Mỗi ảnh đầu vào có prefix `PhieuQG.XXXX_...`.

Các file thường gặp:

- `<prefix>_vertical.jpg`
- `<prefix>_horizontal.jpg`
- `<prefix>_lines.jpg`
- `<prefix>_boxes.jpg`
- `<prefix>_parts.jpg`
- `<prefix>_binary_fillratio_grid.jpg`
- `<prefix>_all_parts_with_grid.jpg`

## 4) Tham số quan trọng trong flow hiện tại

Các giá trị dưới đây đang được hard-code trong `_demo()`:

### 4.1 Detect box morphology

- `vertical_scale = 0.015`
- `horizontal_scale = 0.015`
- `min_line_length = 50`
- `vertical_row_tolerance = 10`
- `block_size = 35`
- `block_offset = 7`
- `min_box_area = 200`
- `min_box_width = 15`
- `min_box_height = 15`
- `close_kernel_size = 3`

### 4.2 Nhận dạng SBD/Mã đề

- SBD: `boxes_per_row=6`, `max_rows=10`, `row_tolerance=30`, `size_tolerance_ratio=0.35`
- Mã đề: `boxes_per_row=3`, `max_rows=10`, `row_tolerance=20`, `size_tolerance_ratio=0.3`
- Extrapolate: `target_rows=10`

### 4.3 Chấm Part I/II/III

- Ngưỡng tô chung: `fill_ratio_thresh = 0.54`
- `mask_mode = hough-circle`
- `circle_radius_scale = 0.6`
- `circle_border_exclude_ratio = 0.1`

Offset lưới:

- Part I: start `(0.2, 0.1)`, end `(0.015, 0.015)`
- Part II: start xen kẽ `(0.3, 0.33)` / `(0.0, 0.33)`, end_y `0.03`
- Part III: start `(0.22, 0.16)`, end `(0.1, 0.015)`, pattern hàng đặc thù

## 5) Cách cài đặt và chạy

## 5.1 Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5.2 Chạy pipeline

```bash
python detect.py 0030
```

Bạn có thể truyền linh hoạt:

- `python detect.py 30`
- `python detect.py PhieuQG.0030`
- `python detect.py PhieuQG/PhieuQG.0030.jpg`
- `python detect.py --image 0030`

## 6) Cách đọc log runtime nhanh

Các dòng chính khi chạy:

- `Detected boxes`: số box kín tìm được sau morphology.
- `Preprocess mode`: `base` hoặc `clahe`.
- `Part I/II/III boxes`: số container lớn từng phần.
- `SoBaoDanh rows`, `MaDe rows`: số hàng ID phát hiện.
- `Extrapolation Summary`: mức thiếu hàng sau căn chỉnh.
- `filled x/y`: số ô được phân loại là tô ở từng phần.

Nếu `SoBaoDanh rows = 0` và `MaDe rows = 0` nhưng Part I/II/III vẫn đủ, nghĩa là pipeline vẫn chấm được phần trắc nghiệm nhưng vùng ID phía trên chưa tách được.

## 7) Checklist debug theo thứ tự ưu tiên

1. Kiểm tra ảnh `*_boxes.jpg`: box kín có đủ và sạch không.
2. Kiểm tra `Part I/II/III boxes` trên log có đạt kỳ vọng 4/8/6 không.
3. Nếu thiếu part, xem pipeline có chuyển `Preprocess mode: clahe` chưa.
4. Kiểm tra `*_parts.jpg` để xác nhận phân vùng đúng vị trí.
5. Nếu chấm sai ô, xem `*_binary_fillratio_grid.jpg` (ratio và mask circle).
6. Nếu sai SBD/Mã đề, xem thống kê `Mean-Darkness Digit Decode` và `Extrapolation Summary`.

## 8) Cấu trúc thư mục hiện tại

```text
XLA_DeThiTHPT/
├── detect.py
├── app.py
├── requirements.txt
├── README.md
├── PhieuQG/
└── output/
    └── detection/
```

## 9) Ghi chú vận hành

- `detect.py` là luồng CLI cốt lõi để kiểm thử thuật toán.
- Đầu ra debug ảnh được thiết kế để truy vết lỗi theo từng pha, nên nên giữ lại khi tuning tham số.
- Khi điều chỉnh ngưỡng, nên kiểm tra đồng thời 3 nhóm chỉ số: số box detect, số hàng ID detect, số ô filled theo part để tránh tối ưu lệch một khu vực.
