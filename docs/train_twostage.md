# Chạy thực nghiệm TwoStageNILMFormer

Từ thư mục gốc project:

```bash
uv run python -m training.train_twostage_nilmformer
```

Runner mặc định đọc CSV từ `data/processed/REFIT/temporal_7aea0ef86866`, dùng
20 hộ, 5 seed và hai nhánh pooled/per_house, tổng cộng **105 lượt train**.
Mặc định dùng CUDA và báo lỗi nếu CUDA không khả dụng; không tự chuyển sang CPU.
Có thể đặt rõ `--device cpu` để chạy kiểm tra trên CPU.
Đầu ra mặc định là thư mục có timestamp trong `training/runs/`.

Mỗi mô hình × seed dùng đúng một đối tượng tqdm. Thanh được reset và tái sử dụng
theo từng epoch, tăng sau mỗi batch train/validation; postfix hiển thị epoch,
pha, batch và loss đang chạy. Phần trăm và ETA trên thanh áp dụng cho epoch hiện
tại. Cuối epoch hiển thị loss/metrics tổng hợp và trạng thái early stopping.

```bash
uv run python -m training.train_twostage_nilmformer \
    --data-dir data/processed/REFIT/temporal_7aea0ef86866 \
    --output-dir training/runs/twostage_full \
    --epochs 50 --patience 10 --num-seeds 5 --batch-size 64
```

Có thể chọn một nhóm hộ bằng `--houses 1 2 3` hoặc chỉ một nhánh bằng
`--protocols pooled` / `--protocols per_house`. Thư mục kết quả phải là thư mục
chưa có study.json, tránh ghi đè kết quả đã chạy.

## Validation và mất cân bằng nhãn

Giữ nguyên 20% cuối làm test. Lấy 20% thời gian cuối bên trong phần train gốc
làm validation, xấp xỉ tỷ lệ toàn kỳ 64/16/20. Cấu hình bằng
`--validation-fraction 0.2`; mốc được chọn trước khi biết nhãn test. Cửa sổ cắt
ngang train/validation bị loại. Lưu mốc và số cửa sổ trong validation_split.csv.

Early stopping theo validation loss với patience 10, min_delta 0 mặc định.
best.pt ghi mọi cải thiện validation tuyệt đối; last.pt ghi trạng thái epoch
cuối. Nạp lại best.pt trước test. Tập test không được dùng cho early stopping.

Trọng số BCE lớp bật tính riêng từng thiết bị từ các nhãn train hợp lệ:
`(off_count + 1)/(on_count + 1)`, chặn `[0.01, 100]` mặc định. Đổi giới hạn trên
bằng `--max-positive-weight`. Kênh không có mẫu bật dùng trọng số 1 và lưu cờ
has_positive_train_labels=False; kênh thiếu nhãn không đóng góp vào thống kê.
Không dùng một tỷ lệ bật/tắt chung cho mọi thiết bị hoặc mọi mô hình riêng.
Giới hạn và trọng số các loss có thể cần hiệu chỉnh trên validation.

## Đầu ra

- study.json: dữ liệu nguồn, hash manifest, danh sách hộ, seed và tham số chạy.
- validation_split.csv: ranh giới train/validation và số cửa sổ từng hộ.
- `pooled/seed_<seed>/` và `house_<id>/seed_<seed>/`: best.pt, last.pt,
  class_balance.csv, history.json, epoch_metrics.csv, training_curves.png,
  training_curves.svg và test_metrics.csv.
- test_metrics.csv ở thư mục gốc: kết quả theo protocol/seed/hộ/thiết bị.
- summary_by_house.csv: trung bình, độ lệch chuẩn và số seed có giá trị.
- macro_by_seed.csv: trung bình theo hộ trong từng seed, tránh để hộ dài chi phối.

Metrics công suất/điện năng dùng đơn vị W/kWh sau hoàn nguyên normalize.
F1 chính so công suất cuối với ngưỡng thiết bị; detector_f1 đánh giá riêng mask.
Nhãn bật/tắt thực tế giữ nguyên quy tắc tạo trước resample của pipeline; đây
là F1 theo mẫu phút, không phải F1 ghép sự kiện. Kênh không có nhãn được ghi
evaluated=False và score NaN. Không đổi NaN thành 0 khi tổng hợp.

Checkpoint lưu cấu hình hai nhánh, trọng số BCE, seed, optimizer và lịch sử.
last.pt là bản lưu trạng thái, chưa có chức năng resume hoàn chỉnh qua CLI.

## Đồ thị và metrics trong quá trình train

Sau mỗi lượt model × seed, runner lưu line plot gồm sáu ô: loss tổng, MAE,
RMSE, F1, precision và recall. Mỗi ô có đường train/validation và vạch đánh dấu
epoch validation tốt nhất. Đồ thị lưu PNG và SVG để xem hoặc đưa vào báo cáo.

Mỗi epoch lưu cả metrics từng thiết bị và macro theo các thiết bị có nhãn hợp
lệ vào history.json và epoch_metrics.csv, gồm thêm NDE, SAE, MR và sai số điện
năng. Metrics dùng công suất đã hoàn nguyên W và nhãn bật/tắt cuối pipeline.
Không dùng test để vẽ đường theo epoch hoặc chọn mô hình.

Các tổng sai số và TP/FP/FN/TN được cộng trên toàn epoch trước khi chia; không
lấy trung bình F1/SAE từng batch. Chỉ số macro trên hình lấy trung bình các
thiết bị hợp lệ, không phải macro theo hộ; bảng test cuối vẫn báo riêng từng hộ.
Train metrics là thống kê dự đoán trong quá trình cập nhật trọng số, với dropout
và mask straight-through bật. Validation metrics dùng eval và mask cứng. Vì vậy
hai đường không phải hai phép đánh giá cùng một trạng thái trọng số cuối epoch.

## Chạy thử kết nối

```bash
uv run python -m training.train_twostage_nilmformer --smoke-test --device cpu
```

Chế độ này chỉ dùng tối đa hai hộ, một seed, một epoch, hai cửa sổ/tập và mô hình
nhỏ. Kết quả có smoke_test=True, không dùng làm benchmark và không được trộn
với kết quả 105 lượt train đầy đủ. Xem `--help` để biết các tham số còn lại.
