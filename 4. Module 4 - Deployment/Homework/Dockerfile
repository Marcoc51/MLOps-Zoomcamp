FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .
COPY model.bin .

RUN pip install pandas==2.2.3 pyarrow==17.0.0 numpy==2.1.1 scipy==1.13.1 scikit-learn==1.5.0

RUN mkdir -p /app/output

ENTRYPOINT ["python", "starter.py"]
