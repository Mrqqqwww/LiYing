FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir     "onnxruntime>=1.17.0"     "orjson>=3.9.15"     "gradio>=4.19.2"     -r requirements.txt

COPY . .

# Download ONNX models (repo ships without them)
# Background matting uses u2netp (4.4MB, 320px input): RMBG-1.4 needs
# 700MB+ RAM at inference and OOMs the 512MB free instance
RUN mkdir -p /app/src/model     && curl -sfL -o /app/src/model/face_detection_yunet_2023mar.onnx        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx     && curl -sfL -o /app/src/model/RMBG-1.4-model.onnx        https://huggingface.co/BritishWerewolf/U-2-Netp/resolve/main/onnx/model.onnx     && curl -sfL -o /app/src/model/yolov8n-pose.onnx        https://huggingface.co/JONNYVERSE/yolov8n-pose/resolve/main/onnx/model.onnx

EXPOSE 7860

# Chinese UI; memory-saver knobs for the 512MB free instance:
# - u2netp input is fixed 320x320
# - no-arena ORT sessions cut peak RSS ~50%
# - source photos downsampled to 3MP before detection/compositing
ENV LIYING_LANG=zh     LIYING_RMBG_SIZE=320     LIYING_ORT_NOARENA=1     LIYING_MAX_PIXELS=3000000

CMD ["python", "src/webui/app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--deployment_mode", "server"]
