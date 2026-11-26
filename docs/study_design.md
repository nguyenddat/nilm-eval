# Thiết kế nghiên cứu: đánh giá theo từng hộ

## Mục tiêu và đơn vị đánh giá

Đánh giá NILMFormer và TwoStageNILMFormer trên từng hộ, từng thiết bị và từng
seed. Đơn vị kết quả là `model × protocol × seed × house × device`.
Notebook triển khai: `data/scripts/Preprocess_REFIT_NILMFormer.ipynb`.

## Hai nhánh thực nghiệm

| Nhánh | Dữ liệu train | Dữ liệu đánh giá |
| --- | --- | --- |
| Mô hình chung | Phần train của tất cả các hộ | Báo riêng phần test của từng hộ |
| Mỗi hộ một mô hình | Chỉ phần train của hộ h | Chỉ phần test của cùng hộ h |

Mô hình riêng được khởi tạo mới, không lấy trọng số mô hình chung. Nếu khởi tạo
từ mô hình chung, đó là nhánh fine-tuning cần tách thành thí nghiệm khác.

## Chia theo thời gian

- Mỗi hộ được xét độc lập theo thứ tự timestamp. Lấy khoảng thời gian lịch từ
  đầu phút chứa mẫu đầu đến hết phút chứa mẫu cuối, chọn mốc 80% và làm tròn lên phút.
- 80% đầu thuộc train, 20% cuối thuộc test. Tỷ lệ này áp dụng trước lọc dữ liệu;
  số cửa sổ hợp lệ không bắt buộc có tỷ lệ đúng 80/20.
- Tách dữ liệu trước forward-fill, resample và cắt cửa sổ. Không điền qua ranh giới,
  không có cửa sổ chồng lấn train/test và không ghép hai đoạn cách nhau bởi khoảng mất dữ liệu.
- CSV tiền xử lý giữ nguyên 80/20. Runner TwoStage tách 20% thời gian cuối của
  phần train gốc làm validation để early stopping; xấp xỉ 64% train, 16% validation,
  20% test theo thời gian toàn kỳ. Loại cửa sổ cắt ngang mốc train/validation.
- Chọn checkpoint theo validation loss, khôi phục trọng số tốt nhất trước khi
  đánh giá test; không dùng test chọn epoch hoặc siêu tham số.
- Không truyền test_loader vào tham số val_loader của các hàm train.
- Mặc định tối đa 50 epoch, patience 10 và min_delta=0; có thể cấu hình qua CLI.

Thiết kế này đo khả năng dự đoán ở thời gian tương lai của hộ đã có dữ liệu train.
Nó không thay thế đánh giá hộ mới chưa từng xuất hiện trong train. Yêu cầu gốc về
hộ chưa thấy cần thêm một nhánh giữ hộ hoàn toàn ngoài train khi triển khai sau.

## Các hộ và nhãn thiết bị

Pipeline mới xét toàn bộ 20 CSV hộ REFIT trong repo, giữ thứ tự đích máy giặt,
máy rửa bát, ấm đun nước và lò vi sóng. Không lọc theo danh sách hộ thực nghiệm
từng thiết bị của tác giả. Ánh xạ mặc định dùng bảng kênh NILMFormer và lưu cạnh
mô tả trong README REFIT để kiểm tra chênh lệch.

Phạm vi thiết bị đã chốt cố định đúng bốn loại trên, theo thứ tự
`WashingMachine`, `Dishwasher`, `Kettle`, `Microwave`; cả hai mô hình dùng
`c_out=4`. Notebook kiểm tra thứ tự này bằng assert. Giữ ngưỡng tạo nhãn và các
bước xử lý tín hiệu theo cấu hình đã đối chiếu với mã nghiên cứu; chia thời gian
80/20 và hai nhánh train vẫn theo thiết kế dự án đã chốt.

Bản cũ còn sáu hộ vì yêu cầu đồng thời: đủ bốn nhãn khớp tên và thuộc giao danh
sách thực nghiệm của bốn thiết bị. Đây là lựa chọn phục vụ đối chứng nhiều đầu
ra đầy đủ nhãn, không có nghĩa các hộ khác không có dữ liệu hoặc không train được.
Ví dụ hộ 20 bị loại khỏi giao cũ vì không nằm trong danh sách máy giặt của cấu
hình tác giả, dù bảng kênh có máy giặt. Hộ 1 không có nhãn ấm/lò vi sóng nhưng
vẫn có thể đóng góp nhãn máy giặt và máy rửa bát.

- Kênh chưa ánh xạ được có valid=False; giá trị 0 lưu trong tensor chỉ là giá trị
  thay thế và không được coi là nhãn tắt trong loss hoặc metrics.
- Giữ cửa sổ có điện tổng và các kênh đã ánh xạ hợp lệ. Kênh chưa ánh xạ không
  làm mất cửa sổ của các thiết bị khác.
- Thiết bị có hậu tố hoặc thiết bị kết hợp không tự được gộp thành một loại;
  inventory phải phản ánh các nhãn chưa ánh xạ. Điều này khác với kết luận thiết bị không tồn tại.
- Nhánh mỗi hộ không đánh giá các đầu ra không có nhãn train/test của hộ đó.
- Chênh lệch ánh xạ nguồn, chẳng hạn Kettle/Hi-Fi tại hộ 15, vẫn được ghi nhận;
  chưa coi các kênh này là nhãn đã xác minh độc lập.

## Seed và so sánh

Giữ seed dữ liệu 42 và cùng mốc chia giữa hai nhánh, hai kiến trúc và 5 seed
khởi tạo. Với 20 hộ, mỗi kiến trúc có `(1 + 20) × 5 = 105` lượt train dự kiến;
hai kiến trúc có tổng 210 lượt. Runner hiện thực thi riêng TwoStageNILMFormer;
baseline vẫn có hàm train riêng, chưa có runner hàng loạt tương ứng.

## Mất cân bằng nhãn bật/tắt

Runner đếm nhãn bật/tắt hợp lệ theo từng thiết bị **chỉ trên các cửa sổ train
thực tế sau tách validation**. Không dùng nhãn validation/test để tính trọng số.
Mô hình chung lấy thống kê mọi hộ train; mô hình riêng lấy thống kê chính hộ đó.

Trọng số lớp dương BCE là `(off_count + 1) / (on_count + 1)`, chặn trong
`[0.01, 100]` mặc định để ổn định gradient; giới hạn trên có thể cấu hình.
Kênh không có mẫu bật dùng trọng số 1 và báo cờ không có mẫu bật để diễn giải
kết quả. Kênh thiếu nhãn có valid=False, không đóng góp vào số đếm hoặc loss.
Các trọng số được broadcast theo kênh trước khi lọc valid để không bị sai thứ tự.
Loss hồi quy sau mask và loss phụ trên các điểm bật vẫn giữ vai trò hiện có.

Lưu class_balance.csv và cấu hình trọng số trong từng checkpoint để truy vết.

Dùng cùng mẫu dữ liệu và mask khi so sánh. Cùng số epoch không đồng nghĩa cùng
chi phí tính toán: mô hình chung thấy nhiều mẫu hơn mỗi mô hình riêng; phải báo
số mẫu, số bước cập nhật, thời gian và chi phí cho từng nhánh.

Báo MAE/RMSE, sai số điện năng và F1 theo hộ/thiết bị; luôn kèm số điểm hợp lệ,
độ phủ nhãn và số mẫu bật. Tổng hợp trung bình/độ lệch chuẩn theo seed; chỉ rõ
cách lấy trung bình theo hộ thay vì để hộ có nhiều mẫu chi phối. Khi phân tích
bật đồng thời, phân tầng theo số kênh được giám sát vì các hộ có độ phủ khác nhau.

Các bước tín hiệu, ngưỡng, TimeRPE và scaling tiếp tục dùng cấu hình ghi trong
notebook. Chia thời gian và mask kênh thiếu là điều chỉnh của nghiên cứu hiện tại,
không gọi là tái lập nguyên thiết lập chia theo hộ trong bài NILMFormer.
