# CSC4005 Lab 3 Report – UrbanSound8K with 1D-CNN

## 1. Thông tin sinh viên

- Họ tên: [Điền họ tên]
- Mã sinh viên: [Điền mã số]
- Lớp: [Điền lớp]
- Link GitHub repo: https://github.com/[your-username]/csc4005-lab3-1dcnn-lein69
- Link W&B run/project: https://wandb.ai/[your-username]/csc4005-lab3-urbansound-1dcnn/runs/...

---

## 2. Mục tiêu thí nghiệm

Mô tả ngắn gọn mục tiêu của lab:

- phân loại âm thanh môi trường trên UrbanSound8K,
- sử dụng MFCC/log-mel làm chuỗi đặc trưng theo thời gian,
- xây dựng và huấn luyện 1D-CNN,
- theo dõi thí nghiệm bằng W&B,
- phân tích lỗi bằng confusion matrix.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

- Dataset: UrbanSound8K
- Số lớp: 10
- Các lớp: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- Fold dùng để train: 1–8
- Fold dùng để validation: 9
- Fold dùng để test: 10

### 3.2. Tiền xử lý audio

Điền cấu hình đã dùng:

| Thành phần | Giá trị |
|---|---|
| Sample rate | 16000 Hz |
| Duration | 4.0 giây |
| Feature type | MFCC |
| n_mfcc / n_mels | 40 |
| n_fft | 1024 |
| hop_length | 512 |
| Augmentation | Time-frequency masking (cho MFCC/log-mel), random shift + noise (cho raw waveform) |

Giải thích ngắn: vì sao cần đưa audio về cùng sample rate và cùng độ dài?

Audio cần cùng sample rate để đảm bảo các features được trích xuất trên cùng một tần số lấy mẫu, và cùng độ dài để input có kích thước cố định cho mô hình. Đồng thời giới hạn độ dài (4s) giúp batch size ổn định và giảm bộ nhớ cần thiết.

---

## 4. Mô hình 1D-CNN

Mô tả kiến trúc mô hình:

```text
Input: [batch, 40, T]  (MFCC features: 40 channels x T time frames)
  → Conv1D Block 1: Conv1d(40→64) + BatchNorm + ReLU + MaxPool1d(k=2)
  → Conv1D Block 2: Conv1d(64→128) + BatchNorm + ReLU + MaxPool1d(k=2)
  → Conv1D Block 3: Conv1d(128→128) + BatchNorm + ReLU + MaxPool1d(k=2)
  → Global Average Pooling: AdaptiveAvgPool1d(1) → [batch, 128]
  → Classifier: Flatten → Dropout(0.35) → Linear(128, 10)
  → Softmax (được áp dụng bởi CrossEntropyLoss)
```

Bảng cấu hình:

| Thành phần | Giá trị |
|---|---|
| model_name | mfcc_1dcnn |
| hidden_channels | [64, 128, 128] |
| dropout | 0.35 |
| optimizer | adamw |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| batch size | 32 |
| epochs | 12 |
| patience | 4 |

---

## 5. Kết quả thực nghiệm

### 5.1. Kết quả chính

| Metric | Giá trị |
|---|---:|
| Best validation accuracy |  |
| Test accuracy |  |
| Average epoch time |  |
| Total parameters |  |
| Trainable parameters |  |

### 5.2. Learning curves

Chèn hình `outputs/<run_name>/curves.png` vào đây.

Nhận xét (mẫu – cần điều chỉnh theo kết quả thực tế):

- Train loss/val loss có giảm đều không? → **Có**, loss giảm mượt từ epoch 1 đến epoch X, sau đó bão hòa.
- Có dấu hiệu overfitting không? → **Không rõ** / **Có nhẹ**: train accuracy tăng tiếp nhưng val accuracy dao động nhỏ, gap nhỏ (~2–3%).
- Early stopping có xảy ra không? → **Có** tại epoch Y khi val_acc không cải thiện trong `patience` epochs.

### 5.3. Confusion matrix

Chèn hình `outputs/<run_name>/confusion_matrix.png` vào đây.

Nhận xét (mẫu – dựa trên pattern thường gặp trên UrbanSound8K):

- **Lớp dễ phân loại**: dog_bark, gun_shot, car_horn (đặc trưng rõ ràng, âm thanh ngắn).
- **Lớp dễ nhầm**: children_playing vs street_music (có nhiễu nền tương tự), engine_idling vs air_conditioner (cả hai đều là âm thanh máy móc tần thấp).
- **Nguyên nhân**: Độ dài clip 4s có thể chưa đủ bắt trọn một sự kiện âm thanh; imbalance nhẹ giữa các lớp; một số lớp có acoustic signatures tương tự (ví dụ engine_idling và air_conditioner).

---

## 6. W&B tracking

**Để tìm run name:**

Run name được đặt trong file config (trường `run_name`) hoặc qua CLI `--run_name`. Mặc định là `debug_run`. Sau khi train xong, outputs được lưu tại:

```
outputs/<run_name>/
```

Ví dụ từ configs có sẵn:

- `configs/baseline_mfcc_1dcnn.json` → `run_name: "mfcc_1dcnn_stable"` → outputs ở `outputs/mfcc_1dcnn_stable/`
- `configs/extension_raw_waveform.json` → `run_name: "extension_raw_waveform_1dcnn"` → outputs ở `outputs/extension_raw_waveform_1dcnn/`

**Link W&B run hoặc project:**

```text
https://wandb.ai/your-username/csc4005-lab3-urbansound-1dcnn/runs/xxxxxxxx
```

**Learning curves (từ local outputs):**

```text
outputs/<run_name>/curves.png
```

Ví dụ:

![Learning Curves](outputs/mfcc_1dcnn_stable/curves.png)

**Confusion matrix (từ local outputs):**

```text
outputs/<run_name>/confusion_matrix.png
```

Ví dụ:

![Confusion Matrix](outputs/mfcc_1dcnn_stable/confusion_matrix.png)

**W&B Dashboard cần log:**

- Train/val loss & accuracy (line charts)
- Test accuracy, best_val_acc (summary)
- Hyperparameter configuration
- Confusion matrix image

**Nếu chạy offline, sync W&B:**

```bash
wandb sync wandb/offline-run-*
```

---

## 7. Phân tích và thảo luận

Trả lời ngắn các câu hỏi:

1. Vì sao dùng 1D-CNN thay vì MLP cho chuỗi đặc trưng audio?
   - 1D-CNN khai thác tính địa phương (local patterns) theo trục thời gian qua kernel sliding, số tham số ít hơn MLP, và invariant với độ dịch chuyển thời gian (translation invariance).
2. Kernel 1D trong bài này đang trượt theo chiều nào?
   - Kernel trượt theo chiều thời gian (time axis) trên chuỗi đặc trưng. Input shape là [batch, channels, time_frames], kernel di chuyển dọc theo dimension cuối cùng.
3. MFCC giúp mô hình học dễ hơn raw waveform ở điểm nào?
   - MFCC đã nén thông tin âm thanh thành các hệ số cepstral, loại bỏ độ cao pitch, tập trung vào đặc trưng phôi (vocal tract) – mô hình học được representation phong phú hơn từ ít feature hơn so với raw waveform thô.
4. Mô hình hiện tại còn hạn chế gì?
   - Mô hình chỉ khai thác context cục bộ qua 3 lớp conv, chưa có cơ chế attention toàn cục; số lớp hạn chế nên capacity có thể thấp với âm thanh phức tạp; cũng có thể underfitting nếu số epoch ít.
5. Có thể cải thiện kết quả bằng cách nào?
   - Tăng độ sâu (thêm Conv1D blocks), dùng separable/depthwiseConv để giảm tham số, thêm residual connections, dùng larger kernel hoặc dilated conv, fine-tune learning rate/scheduler, tăng data augmentation, hoặc ensemble nhiều mô hình (MFCC + log-mel).

---

## 8. Bài mở rộng nếu có

Nếu thử nghiệm thêm pipeline khác (log-mel hoặc raw waveform), điền bảng so sánh:

| Pipeline | Feature/Input | Test accuracy | Nhận xét |
|---|---:|---:|---|
| Baseline | MFCC (40 channels) |  | MFCC là feature phổ biến, ổn định, phù hợp với âm thanh môi trường |
| Extension 1 | log-mel (64 channels) |  | log-mel giữ thông tin phổ đầy đủ hơn, nhưng có thể chứa nhiễu |
| Extension 2 | raw waveform (1 channel) |  | Khó khăn: mô hình phải tự học representation từ tín hiệu thô, cần nhiều dữ liệu hơn |

Chạy từng config tương ứng:

```bash
# MFCC baseline
python -m src.train --config configs/baseline_mfcc_1dcnn.json --data_dir /path/to/UrbanSound8K

# Log-mel (tạo config tương tự với feature_type: "logmel")
python -m src.train --config configs/your_logmel_config.json --data_dir /path/to/UrbanSound8K

# Raw waveform
python -m src.train --config configs/extension_raw_waveform.json --data_dir /path/to/UrbanSound8K
```

**(Lưu ý: Bài mở rộng là tùy chọn, không bắt buộc. Nếu không làm, có thể bỏ qua phần này.)**

---

## 9. Kết luận

Tóm tắt 3–5 ý chính học được từ lab:

1. **Tiền xử lý audio chuẩn hóa** (cùng sample rate và độ dài) là bước quan trọng để đảm bảo input cố định, batch ổn định và giảm bộ nhớ.

2. **MFCC biểu diễn tốt đặc trưng âm thanh** cho bài toán phân loại môi trường, giúp mô hình học nhanh hơn raw waveform và đạt độ chính xác cao.

3. **1D-CNN phù hợp với chuỗi thời gian** nhờ khai thác local patterns qua kernel sliding, số tham số thấp hơn MLP và có tính translation-invariance.

4. **Data augmentation** (time-frequency masking) giúp mô hình tổng quát hóa tốt hơn, giảm overfitting trên dữ liệu hạn chế.

5. **W&B tracking** hỗ trợ theo dõi quá trình train, so sánh hyperparameters và phân tích kết quả thông qua learning curves và confusion matrix.
