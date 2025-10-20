Vietnamese ASR (QuartzNet) for Kria KV260 Deployment
Dự án này chứa các script để chuyển đổi một mô hình nhận dạng tiếng nói tiếng Việt (ASR) QuartzNet đã được huấn luyện trước từ checkpoint PyTorch sang định dạng ONNX đã được lượng tử hóa (quantized), sẵn sàng để triển khai trên các bo mạch Xilinx Kria KV260 (sử dụng DPU).

💡 Tổng quan quy trình (Workflow)
Quy trình tổng thể bao gồm hai bước chính:

Export: Chuyển đổi các checkpoint của mô hình (encoder và decoder) từ định dạng PyTorch (.pt) sang một mô hình ONNX duy nhất với độ chính xác float32 (FP32).

Quantize: Lượng tử hóa mô hình ONNX FP32 sang định dạng INT8 bằng cách sử dụng dữ liệu hiệu chỉnh (calibration data). Quá trình này giúp tối ưu hóa mô hình để chạy hiệu quả trên DPU.

Sơ đồ quy trình: PyTorch Checkpoints (.pt) → quartznet_vi_export.py → FP32 ONNX Model → quan.py → Quantized INT8 ONNX Model

⚙️ Cài đặt môi trường
Clone repo của Vitis AI
```bash
git clone https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI/src/vai_quantizer/vai_q_onnx/
sh build.sh
pip install pkgs/*.whl
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
```

🚀 Hướng dẫn sử dụng
Bước 1: Export Model từ PyTorch sang ONNX (FP32)
Bước này sử dụng script quartznet_vi_export.py để tạo ra một file ONNX FP32 từ các checkpoint của mô hình.

Chuẩn bị Checkpoints: Đặt các file checkpoint của encoder và decoder vào đúng đường dẫn được định nghĩa trong script. Mặc định là:

models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt

models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt

Chạy script: Thực thi script từ terminal:


python quartznet_vi_export.py
Kết quả: Script sẽ tạo ra một file quartznet_vietnamese_mel_fixed.onnx ở thư mục gốc. File này là mô hình ASR của bạn ở định dạng ONNX FP32.

Bước 2: Quantize Model ONNX sang INT8
Bước này sử dụng script quan.py để chuyển đổi file .onnx ở Bước 1 sang định dạng INT8 tối ưu cho DPU.

Chuẩn bị Dữ liệu Hiệu chỉnh (Calibration Data): Lượng tử hóa tĩnh (static quantization) yêu cầu một bộ dữ liệu nhỏ (khoảng 100-1000 mẫu) để "hiệu chỉnh" dải giá trị của các tham số.

Tải một bộ dữ liệu tiếng nói tiếng Việt, ví dụ như VIVOS.

Giải nén và đặt vào một thư mục.

Mở file quan.py và cập nhật biến dataset_path để trỏ đến thư mục chứa dữ liệu của bạn.

Ví dụ trong file quan.py:

```
# ...
if __name__ == "__main__":
    try:
        quantize_quartznet_model(
            model_path="quartznet_vietnamese_mel_fixed.onnx",
            output_path="quan_quartz.onnx",
            # THAY ĐỔI ĐƯỜNG DẪN NÀY
            dataset_path="/path/to/your/vivos/dataset", 
            max_samples=1000,
        )
# ...
```
Chạy script quantize: Quan trọng: Đảm bảo bạn đang ở trong môi trường Vitis AI (ví dụ: Docker container).

Bash

python quan.py
Kết quả: Script sẽ tạo ra file quan_quartz.onnx. Đây là mô hình đã được lượng tử hóa INT8, sẵn sàng cho các bước tiếp theo như biên dịch bằng vai_c_onnx.

📄 Mô tả các file
quartznet_vi_export.py:

Xây dựng lại kiến trúc mô hình QuartzNet trong PyTorch.

Tải trọng số từ các file checkpoint .pt (sau khi đã trích xuất ra .npy).

Thực hiện việc export mô hình sang định dạng ONNX FP32 với input là các đặc trưng mel-spectrogram.

quan.py:

Sử dụng thư viện vai_q_onnx của Vitis AI.

Định nghĩa một AudioCalibrationDataReader để đọc và tiền xử lý các file âm thanh từ bộ dữ liệu hiệu chỉnh.

Thực hiện lượng tử hóa tĩnh (static quantization) để chuyển đổi mô hình ONNX FP32 sang INT8.

