# 模型下载问题排查指南

## 1. 增强的下载机制

已优化 `model_manager.py` 的下载逻辑，现在具备：

### ✅ 自动重试机制
- **3次重试**: 网络超时时自动重试，间隔递增 (5s, 10s, 15s)
- **双重方法**: requests → urllib 备用方案
- **智能超时**: 连接超时30s，读取超时300s
- **大块传输**: 64KB chunk size 提升下载效率

### ✅ 增强的错误处理
- 网络超时检测与重连
- 文件完整性验证
- 进度显示优化（每2秒更新）
- 下载速度监控

## 2. 当前错误分析

从您的终端输出看到：
```
requests下载超时 (尝试 1): HTTPSConnectionPool(host='dl.fbaipublicfiles.com', port=443): Read timeout (300 seconds)
```

这表明：
- 已成功连接到 Facebook 的服务器
- 但在300秒内未能完成2.5GB文件的下载
- 说明网络速度较慢或不稳定

## 3. 解决方案

### 🎯 方案1: 手动下载（推荐）

如果网络环境不佳，建议手动下载并放置文件：

1. **下载SAM模型权重**：
   ```bash
   # 使用迅雷/IDM等下载工具，或者分段下载
   wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

2. **放置到正确位置**：
   ```
   d:\AIdj\model_cache\sam_vit_h_4b8939.pth
   ```

3. **验证文件完整性**：
   - 文件大小应约为 2.5GB (2,564,550,879 字节)
   - MD5校验：`a7bf3b02f3ebf1267aba913ff637d9a2`

### 🎯 方案2: 使用国内镜像

修改 `model_manager.py` 中的下载URL：

```python
# 在第251行替换为国内镜像
sam_url = "https://download.pytorch.org/models/sam_vit_h_4b8939.pth"
# 或其他可靠的镜像源
```

### 🎯 方案3: 配置代理

如果有代理服务，可以配置：

```python
# 在 _download_from_url 方法中添加代理
proxies = {
    'http': 'http://your-proxy:port',
    'https': 'https://your-proxy:port'
}
response = session.get(url, stream=True, proxies=proxies, ...)
```

### 🎯 方案4: 分段下载脚本

创建专用下载脚本：

```python
import requests
import os
from pathlib import Path

def download_with_resume(url, local_path, chunk_size=1024*1024):
    """支持断点续传的下载"""
    headers = {}
    if os.path.exists(local_path):
        headers['Range'] = f'bytes={os.path.getsize(local_path)}-'
    
    response = requests.get(url, headers=headers, stream=True, timeout=(30, 600))
    
    with open(local_path, 'ab' if 'Range' in headers else 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

# 使用示例
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
download_with_resume(url, "d:/AIdj/model_cache/sam_vit_h_4b8939.pth")
```

## 4. 验证修复效果

下载完成后，测试应用：

```python
# 检查文件是否正确
python -c "
import os
from pathlib import Path
model_path = Path('d:/AIdj/model_cache/sam_vit_h_4b8939.pth')
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024*1024)
    print(f'✅ 模型文件存在: {size_mb:.1f}MB')
else:
    print('❌ 模型文件不存在')
"
```

```bash
# 启动应用测试
python app.py
```

## 5. 生产环境优化

### Render部署优化：
1. **预构建镜像**: 将模型文件打包到Docker镜像中
2. **CDN加速**: 使用更快的CDN分发模型文件
3. **分层缓存**: 利用Render的持久化存储

### S3集成：
```bash
# 配置环境变量
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export S3_BUCKET="your-model-bucket"
```

## 6. 应急处理

如果仍无法下载，可以：

1. **临时禁用SAM功能**: 修改路由跳过mask生成
2. **使用轻量化模型**: 替换为更小的SAM变体
3. **本地开发**: 先在本地完成开发，生产环境再配置

---

**下一步操作建议**：
1. 先尝试手动下载文件到 `model_cache` 目录
2. 重新启动应用验证
3. 如果问题持续，考虑配置S3或使用镜像源