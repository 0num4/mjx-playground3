FROM python:3.12

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    swig \
    && rm -rf /var/lib/apt/lists/*

# PyTorchのインストール
RUN curl -fsSL https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl -o torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl \
    && pip install torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl \
    && rm torch-2.1.1%2Bcu118-cp39-cp39-linux_x86_64.whl

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.5.0

# Poetryを PATH に追加
ENV PATH="${PATH}:/root/.local/bin"

# ワークディレクトリの設定
WORKDIR /app

# pyproject.tomlとpoetry.lockをコピー
COPY pyproject.toml poetry.lock ./

# 依存関係のインストール
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# プロジェクトファイルをコピー
COPY . .

# コンテナ起動時に実行するコマンド
CMD ["poetry", "run", "python", "main.py"]