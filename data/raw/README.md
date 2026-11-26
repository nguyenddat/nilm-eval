# Dữ liệu nguồn NILM

Tải và giải nén tự động bằng `python data/scripts/download_datasets.py` từ thư mục gốc dự án.
Thêm `--include-raw` để tải cả archive raw gốc. Xem tùy chọn và hướng dẫn sau khi clone tại [README dự án](../../README.md).
Các archive và tệp tải dở được Git bỏ qua; README/metadata nhỏ vẫn được lưu trong repository.

Dữ liệu được tải ngày 25/11/2026 từ kho của University of Strathclyde. Các tệp nguồn được giữ nguyên như khi công bố; archive `.7z` vẫn được lưu tại đây, bản cleaned đã giải nén nằm trong `data/cleaned/`.

| Bộ dữ liệu | Tệp nén | Nguồn |
| --- | --- | --- |
| REFIT, bản đã làm sạch | `REFIT/CLEAN_REFIT_081116.7z` | https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z |
| REFIT, bản thô gốc | `REFIT/REFIT_RAW_081116.7z` | https://pure.strath.ac.uk/ws/portalfiles/portal/62090177/REFIT_RAW_081116.7z |
| PLEGMA, bản đã làm sạch | `PLEGMA/PlegmaDataset_Clean.7z` | https://pure.strath.ac.uk/ws/portalfiles/portal/180681715/PlegmaDataset_Clean.7z |
| PLEGMA, bản thô gốc | `PLEGMA/PlegmaDataset_Raw.7z` | https://pure.strath.ac.uk/ws/portalfiles/portal/180681717/PlegmaDataset_Raw.7z |

README nguồn và metadata REFIT đi kèm nằm trong thư mục của từng bộ dữ liệu. Trang công bố: [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) và [PLEGMA](https://pureportal.strath.ac.uk/en/datasets/plegma-dataset/). Cả hai được công bố theo giấy phép CC BY 4.0.
