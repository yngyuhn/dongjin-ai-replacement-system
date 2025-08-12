FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
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

# 复制依赖文件
COPY requirements-cloud.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-cloud.txt

# 复制应用代码
COPY . .

# 创建模型目录并设置权限
RUN mkdir -p /app && chmod 755 /app

# 暴露端口
EXPOSE 5000

# 启动应用（模型将在首次运行时自动下载）
CMD ["python", "app.py"]