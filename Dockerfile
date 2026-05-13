FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY benchmark ./benchmark
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "benchmark.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
