# PLUTOプロジェクト用のDockerfile
# ベースイメージ: CUDA対応のPyTorchイメージを使用
# 根拠: README.mdでGPUを使用した学習・評価が行われているため
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /workspace

# 環境変数を設定
# PYTHONUNBUFFERED: Pythonの標準出力をバッファリングしない（ログのリアルタイム表示のため）
# DEBIAN_FRONTEND: aptのインタラクティブモードを無効化
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

# システムパッケージの更新とインストール
# git: nuplan-devkitのクローンに必要
# wget, curl: ファイルのダウンロードに必要
# build-essential: C/C++コンパイラ（一部のPythonパッケージのビルドに必要）
# libglib2.0-0, libsm6, libxext6, libxrender-dev: OpenCVの依存関係
# libgl1-mesa-glx: OpenCVのlibGL.so.1に必要
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python 3.9 環境を作成
# 根拠: README.md の "conda create -n pluto python=3.9" を再現
# 注: PyTorch イメージに conda は既にインストール済み
RUN conda create -n pluto python=3.9 -y

# nuplan-devkitのインストール
RUN cd /workspace && git clone https://github.com/est2mzd/nuplan-devkit.git
RUN cd /workspace/nuplan-devkit && git checkout feature/pluto
RUN cd /workspace/nuplan-devkit && git pull

# setup.pyをインストール（editable mode）
# 根拠: README.mdの手順順序に従い、setup.py → requirements.txt の順
# conda pluto環境内で実行
RUN /opt/conda/bin/conda run -n pluto pip install --no-cache-dir -e /workspace/nuplan-devkit

# requirements.txtをインストール
# 修正版requirements.txtを使用（PyTorch 2.0.1環境との互換性を確保）
# --prefer-binary: バイナリパッケージを優先（インストール高速化）
RUN /opt/conda/bin/conda run -n pluto pip install --no-cache-dir --prefer-binary -r /workspace/nuplan-devkit/requirements.txt

# PLUTOプロジェクトの依存パッケージをインストール
# 根拠: キャッシュ効率のためにrequirements.txtを先にコピーしてインストール
# conda pluto環境内で実行
WORKDIR /workspace/pluto
COPY requirements.txt /tmp/pluto_requirements.txt
RUN /opt/conda/bin/conda run -n pluto pip install --no-cache-dir --prefer-binary -r /tmp/pluto_requirements.txt

# setup_env.sh は不要
# COPY script/setup_env.sh /tmp/pluto_setup_env.sh
# RUN /opt/conda/bin/conda run -n pluto bash -c "sh /tmp/pluto_setup_env.sh"

# 環境変数を追加設定
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV NUPLAN_MAPS_ROOT="/nuplan/dataset/maps"
ENV NUPLAN_DATA_ROOT="/nuplan/dataset"
ENV WANDB_DIR="/workspace/exp"
ENV PATH="/opt/conda/envs/pluto/bin:$PATH"

# conda pluto環境を有効化
# 根拠: PATH を設定することで、python コマンドが常にpluto環境を使用される
ENV PATH="/opt/conda/envs/pluto/bin:$PATH"

# conda を初期化して .bashrc に conda activate を追加
# これにより、インタラクティブなbashシェルで pluto環境が自動的に有効化される
RUN /opt/conda/bin/conda init bash && \
    echo "conda activate pluto" >> /root/.bashrc

# 作業ディレクトリをplutoプロジェクトに設定
WORKDIR /workspace

# デフォルトコマンド: bashシェルを起動
CMD ["/bin/bash"]
