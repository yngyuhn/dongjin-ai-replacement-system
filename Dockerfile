FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括Git）
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制云部署专用依赖文件
COPY requirements-cloud.txt .

# 先安装基础依赖
RUN pip install --no-cache-dir flask==2.3.3 \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    Pillow==10.0.1 \
    scikit-learn==1.3.0 \
    scipy==1.11.4 \
    tqdm==4.66.1 \
    requests==2.31.0

# 单独安装segment-anything
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# 复制应用代码
COPY . .

# 创建模型目录并设置权限
RUN mkdir -p /app && chmod 755 /app

# 暴露端口
EXPOSE 5000

# 启动应用（模型将在首次运行时自动下载）
CMD ["python", "app.py"]