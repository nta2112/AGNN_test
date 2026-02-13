# Hướng Dẫn Dự Đoán (Inference) với AGNN

## 1. Chuẩn bị dữ liệu

### Thư mục Support (Kho dữ liệu mẫu)
Bạn có thể trỏ thẳng vào thư mục chứa dữ liệu gốc của bạn! Code sẽ tự động lấy ngẫu nhiên 5 ảnh (hoặc số lượng bạn tùy chỉnh) mỗi lớp để làm mẫu. không cần phải copy thủ công nữa.

Cấu trúc thư mục:
```
data/
    class_A/
        (hàng trăm ảnh...)
    class_B/
        (hàng trăm ảnh...)
    ...
```

### Thư mục Query (Ảnh cần dự đoán)
Chứa các ảnh bạn muốn dự đoán.
Ví dụ:
```
query_data/
    test_01.jpg
    test_02.jpg
    ...
```

## 2. Cách chạy script

Sử dụng lệnh sau trong terminal:

```bash
python predict_custom.py \
    --config ./configs/train_meta_agnn_resnet50.yaml \
    --checkpoint ./save/path/to/your/checkpoint/max-va.pth \
    --support-path /path/to/your/full/dataset/images \
    --query-path ./path/to/query_data \
    --n-shot 5
```

**Giải thích các tham số:**
- `--support-path`: Giờ đây bạn có thể trỏ vào thư mục gốc chứa tất cả các lớp. Code sẽ tự lo liệu việc lấy mẫu.
- `--n-shot`: Số lượng ảnh mẫu lấy ngẫu nhiên cho mỗi lớp (Mặc định là 5).
- `--query-path`: Thư mục chứa các ảnh cần dự đoán.

## 3. Lưu ý
Nếu một lớp nào đó có ít ảnh hơn số `n-shot` bạn yêu cầu, script sẽ tự động nhân bản ảnh để đủ số lượng, đảm bảo code không bị lỗi.
