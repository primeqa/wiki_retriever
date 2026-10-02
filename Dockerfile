FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY wiki_retriever ./wiki_retriever
RUN pip install --no-cache-dir '.[local]'
EXPOSE 8000
ENTRYPOINT ["wiki-retriever", "serve", "--host", "0.0.0.0"]
