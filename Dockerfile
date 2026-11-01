FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NEXUS_DB_PATH=/app/data/nexus.db HF_HOME=/app/model-cache
WORKDIR /app
COPY requirements-app.txt requirements-retrieval.txt ./
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt -r requirements-retrieval.txt \
    && useradd --create-home --uid 10001 nexus \
    && mkdir -p /app/data /app/model-cache && chown -R nexus:nexus /app
COPY --chown=nexus:nexus nexus ./nexus
COPY --chown=nexus:nexus policies ./policies
COPY --chown=nexus:nexus evaluation ./evaluation
USER nexus
EXPOSE 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8766/health',timeout=3)"
CMD ["python","-m","uvicorn","nexus.api:app","--host","0.0.0.0","--port","8766","--workers","1"]
