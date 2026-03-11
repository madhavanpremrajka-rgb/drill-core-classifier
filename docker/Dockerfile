FROM python:3.12-slim

WORKDIR /app
ENV MODELS_DIR=/app/models
ENV METRICS_PATH=/app/Artifacts/metrics.json

COPY inference/requirements.txt .
RUN pip install --no-cache-dir setuptools==75.8.0
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ .
COPY src/ ./src/

COPY models/EfficientNetB0_C-35_R-128_L-0.2_trainable-True_20260220-092325/ ./models/EfficientNetB0_C-35_R-128_L-0.2_trainable-True_20260220-092325/
COPY Artifacts/metrics.json ./Artifacts/metrics.json
COPY Artifacts/class_names.json ./Artifacts/class_names.json

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]