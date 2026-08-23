FROM python:3.11-slim

# OpenCV 在 headless 模式下仍然需要這些系統函式庫
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝 Python 依賴（利用 Docker layer cache，只有依賴有變才重裝）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 Python 原始碼
COPY server.py run_inference.py calibrate.py run_calibration.py ./

EXPOSE 8000

# 預設以靜態圖片模式啟動；實際參數可在 docker-compose.yml 覆寫
CMD ["python", "server.py", \
    "--model", "/app/models/best.pt", \
    "--calibration", "/app/seat_calibration.json", \
    "--image", "/app/assets/pictures/main/main2.png", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
