FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir     "onnxruntime>=1.17.0"     "orjson>=3.9.15"     "gradio>=4.19.2"     -r requirements.txt

COPY . .

# Download ONNX models (repo ships without them)
RUN mkdir -p /app/src/model     && curl -sfL -o /app/src/model/face_detection_yunet_2023mar.onnx        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx     && curl -sfL -o /app/src/model/RMBG-1.4-model.onnx        https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx     && curl -sfL -o /app/src/model/yolov8n-pose.onnx        https://huggingface.co/JONNYVERSE/yolov8n-pose/resolve/main/onnx/model.onnx

EXPOSE 7860

CMD ["python", "src/webui/app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--deployment_mode", "server"]
