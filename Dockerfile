# base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04


WORKDIR /app

# 直接覆盖默认的 SHELL（推荐）
SHELL ["/bin/bash", "-c"]

# 使用中科大镜像源替换默认的Debian源
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
        sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list; \
    else \
        mkdir -p /etc/apt && \
        echo "deb https://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list && \
        echo "deb https://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
        echo "deb https://mirrors.ustc.edu.cn/debian-security/ bookworm-security main contrib non-free" >> /etc/apt/sources.list; \
    fi

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps \
        build-essential \
        libpoppler-cpp-dev \
        pkg-config \
        wget \
        bzip2 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*


# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH

## 设置清华镜像源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes


# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 复制应用代码
COPY . /app/
RUN conda env create -f environment.yaml
RUN conda activate gemorna

