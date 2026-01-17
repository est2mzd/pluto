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
# 根拠: README.mdのSetup手順に記載されている必須コンポーネント
# ローカルのnuplan-devkitをコピー（修正済みrequirements.txtを内包）
# ビルドコンテキスト: プロジェクトルート（COPY パスは相対）
COPY nuplan-devkit /workspace/nuplan-devkit

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
COPY requirements.txt .
RUN /opt/conda/bin/conda run -n pluto pip install --no-cache-dir --prefer-binary -r requirements.txt

# PLUTOプロジェクトのソースコードをコピー
# 根拠: ファイル・ディレクトリを明示的に指定（.dockerignoreで管理）
# 重複を排除し、必要なファイルのみコピー
COPY README.md .
COPY config ./config
COPY script ./script
COPY src ./src
COPY run_simulation.py .
COPY run_training.py .

# setup_env.sh を実行
# 根拠: README.md の最後のステップ "sh ./script/setup_env.sh"
RUN /opt/conda/bin/conda run -n pluto bash -c "cd /workspace/pluto && sh ./script/setup_env.sh"

# PYTHONPATHを設定
# 根拠: README.mdでexport PYTHONPATH=$PYTHONPATH:$(pwd)が使用されている
ENV PYTHONPATH="/workspace/pluto:${PYTHONPATH}"

# nuPlan データセットの環境変数を設定
# 根拠: nuplan-devkit が這些環境変数を使用して、maps と data を検索
# dataset配下に統合されたため、パスを修正
ENV NUPLAN_MAPS_ROOT="/nuplan/dataset/maps"
ENV NUPLAN_DATA_ROOT="/nuplan/dataset"

# conda pluto環境を有効化
# 根拠: PATH を設定することで、python コマンドが常にpluto環境を使用される
ENV PATH="/opt/conda/envs/pluto/bin:$PATH"

# conda を初期化して .bashrc に conda activate を追加
# これにより、インタラクティブなbashシェルで pluto環境が自動的に有効化される
RUN /opt/conda/bin/conda init bash && \
    echo "conda activate pluto" >> /root/.bashrc

# 作業ディレクトリをplutoプロジェクトに設定
WORKDIR /workspace/pluto

# デフォルトコマンド: bashシェルを起動
CMD ["/bin/bash"]
