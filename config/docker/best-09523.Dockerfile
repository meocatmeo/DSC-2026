FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
LABEL org.opencontainers.image.title="DSC best 0.9523 score-cache recovery"
LABEL org.opencontainers.image.description="Recomputes the frozen 2080 private predictions from saved model scores. Not fresh GPU inference."
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_INDEX=1
WORKDIR /app
COPY wheels/ /wheels/
COPY requirements-recovery.txt /app/requirements-recovery.txt
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels --require-hashes -r requirements-recovery.txt
COPY recovery/ /app/
RUN mkdir /output && chown 65532:65532 /output
USER 65532:65532
ENTRYPOINT ["python", "/app/recover.py"]
CMD ["--out", "/output"]
