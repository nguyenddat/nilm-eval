# NILM

## Chuẩn bị dữ liệu sau khi clone

Cần Python 3.14 và 7-Zip (`7z` hoặc `7zz`). Trên Ubuntu/Debian:

```bash
sudo apt install p7zip-full
python data/scripts/download_datasets.py
```

Script chỉ dùng thư viện chuẩn Python, không cần cài các thư viện huấn luyện để tải dữ liệu.
Mặc định tải cả REFIT và PLEGMA bản cleaned cùng README/metadata, kiểm tra SHA-256
và giải nén vào `data/cleaned/REFIT/` và `data/cleaned/PLEGMA/` để các notebook hiện có đọc trực tiếp.
Archive được lưu trong `data/raw/<bộ dữ liệu>/`.

Các tùy chọn:

```bash
# Chỉ chuẩn bị một bộ dữ liệu
python data/scripts/download_datasets.py --dataset refit
python data/scripts/download_datasets.py --dataset plegma

# Tải thêm cả hai archive raw gốc, giữ nguyên dạng nén
python data/scripts/download_datasets.py --include-raw

# Chỉ tải, không giải nén; không cần 7-Zip
python data/scripts/download_datasets.py --download-only
```

Có thể chạy script từ bất kỳ thư mục nào; đường dẫn mặc định được xác định theo vị trí script.
Dùng `--data-dir /đường/dẫn/data` nếu muốn lưu ở nơi khác; các notebook hiện tại vẫn dùng `data/` của repository.
Chạy lại sẽ bỏ qua tệp tải về có SHA-256 hợp lệ và thư mục cleaned đã có đủ tệp đúng kích thước.
Tệp tải dở có đuôi `.part`, được tải lại từ đầu ở lần chạy sau; script tự thử tối đa ba lần khi tải lỗi.
Nếu máy chủ trả HTTP 403, có thể tải bằng trình duyệt theo URL trong `data/scripts/datasets.json`,
đặt tệp vào `data/raw/<bộ dữ liệu>/` rồi chạy lại script để kiểm tra và giải nén.

Lần chuẩn bị đầu cần tải khoảng 0,8 GB archive cleaned và khoảng 8,3 GB dữ liệu sau giải nén.
Tùy chọn `--include-raw` tải thêm khoảng 1,4 GB archive raw gốc.
Git bỏ qua archive, tệp tải dở, toàn bộ `data/cleaned/` và `data/processed/`;
script, notebook và README/metadata nguồn nhỏ vẫn được giữ lại khi clone.
`data/processed/` được tạo lại bằng notebook `data/scripts/Preprocess_REFIT_NILMFormer.ipynb`, không được tải bởi script.

Nguồn công bố và giấy phép: [REFIT cleaned](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/),
[REFIT raw](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) và
[PLEGMA](https://pureportal.strath.ac.uk/en/datasets/plegma-dataset/) (CC BY 4.0).
URL từng tệp và SHA-256 của bản dữ liệu đang dùng được lưu trong `data/scripts/datasets.json`.
