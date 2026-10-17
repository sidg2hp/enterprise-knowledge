FROM python:3.10.8-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
