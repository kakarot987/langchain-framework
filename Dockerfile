FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY docs ./docs
COPY evals ./evals
ENV PYTHONUNBUFFERED=1 DOCQA_DOCS=/app/docs/sample
USER 65532:65532
ENTRYPOINT ["docqa"]
CMD ["demo"]
