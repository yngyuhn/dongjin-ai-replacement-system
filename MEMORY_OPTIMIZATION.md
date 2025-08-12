# 内存优化和外部存储解决方案

## 问题背景

在Render等云平台部署时，遇到"Out of memory (used over 512Mi)"错误，主要原因：
1. SAM和DINOv2模型文件较大（总计约3GB）
2. 模型加载到内存后占用大量RAM
3. Render免费版本内存限制为512MB

## 解决方案

### 1. 按需加载 (Lazy Loading)
- 模型不在启动时预加载
- 仅在首次使用时下载和加载
- 支持模型卸载以释放内存

### 2. 外部存储支持
- 支持AWS S3存储模型文件
- 本地缓存机制，避免重复下载
- 自动回退到原始URL下载

### 3. 内存管理
- 实时监控内存使用
- 自动卸载未使用的模型
- 可配置的内存阈值

### 4. 优化参数
- 减少SAM的points_per_side (64→32)
- 减少points_per_batch (32→16)
- 降低内存占用约50%

## 配置说明

### 环境变量配置

复制`.env.example`为`.env`并配置：

```bash
# AWS S3配置（可选）
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name

# 模型S3路径
SAM_MODEL_S3_PATH=models/sam_vit_b_01ec64.pth
DINOV2_MODEL_S3_PATH=models/dinov2_vitb14_pretrain.pth

# 内存管理
MAX_MEMORY_USAGE_MB=400
ENABLE_MODEL_UNLOADING=true
LOCAL_CACHE_DIR=./model_cache
```

### 本地部署

如果不配置S3，系统将：
1. 从原始URL下载模型
2. 缓存到本地目录
3. 按需加载和卸载

### S3部署

1. 将模型文件上传到S3：
```bash
aws s3 cp sam_vit_b_01ec64.pth s3://your-bucket/models/
aws s3 cp dinov2_vitb14_pretrain.pth s3://your-bucket/models/
```

2. 配置环境变量
3. 部署应用

## 内存使用优化

### 优化前
- 启动时加载所有模型：~3GB
- 内存峰值：>512MB
- Render部署失败

### 优化后
- 启动时内存：~50MB
- 按需加载：仅加载需要的模型
- 自动卸载：释放未使用的模型
- 内存峰值：<400MB

## 技术实现

### ModelManager类
```python
class ModelManager:
    def __init__(self):
        self.cache_dir = os.getenv('LOCAL_CACHE_DIR', './model_cache')
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_USAGE_MB', '400'))
        self.enable_unloading = os.getenv('ENABLE_MODEL_UNLOADING', 'true').lower() == 'true'
    
    def download_from_s3(self, s3_path, local_path):
        # S3下载逻辑
    
    def auto_manage_memory(self):
        # 自动内存管理
```

### 按需加载函数
```python
def ensure_sam_model():
    if not models_loaded['sam']:
        # 下载和加载SAM模型
        return load_sam_model_lazy()
    return sam_model, mask_generator

def ensure_dinov2_model():
    if not models_loaded['dinov2']:
        # 下载和加载DINOv2模型
        return load_dinov2_model()
    return dinov2_model, dinov2_transform
```

## 部署建议

### Render部署
1. 使用优化后的配置
2. 设置环境变量
3. 监控内存使用
4. 考虑升级到付费计划以获得更多内存

### 其他云平台
- Heroku: 类似配置
- Railway: 支持更大内存
- AWS/GCP: 可配置实例大小

## 监控和调试

### 内存监控
```python
def get_memory_status():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024
    }
```

### 日志级别
设置`LOG_LEVEL=DEBUG`查看详细日志：
- 模型下载进度
- 内存使用情况
- 自动卸载事件

## 故障排除

### 常见问题

1. **S3下载失败**
   - 检查AWS凭证
   - 验证S3路径
   - 检查网络连接

2. **内存仍然不足**
   - 降低MAX_MEMORY_USAGE_MB
   - 启用ENABLE_MODEL_UNLOADING
   - 考虑升级服务器

3. **模型加载慢**
   - 使用S3存储
   - 选择更近的AWS区域
   - 检查网络带宽

### 性能调优

1. **减少内存使用**
   - 降低points_per_side
   - 减少batch_size
   - 启用自动卸载

2. **提高下载速度**
   - 使用CDN
   - 选择合适的AWS区域
   - 压缩模型文件

## 未来改进

1. **模型压缩**
   - 量化模型
   - 剪枝技术
   - 知识蒸馏

2. **缓存策略**
   - Redis缓存
   - 分布式缓存
   - 预热机制

3. **负载均衡**
   - 多实例部署
   - 模型分片
   - 动态扩缩容