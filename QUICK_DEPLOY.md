# 🚀 侗锦AI系统 - 3分钟快速部署

## 最简洁方案：Render（完全免费）

### 1️⃣ 一键部署步骤

1. **访问 Render**：https://render.com
2. **GitHub登录**：使用您的GitHub账号登录
3. **新建服务**：点击 "New +" → "Web Service"
4. **连接仓库**：选择 `yngyuhn/dongjin-ai-replacement-system`
5. **配置设置**：
   ```
   Name: dongjin-ai-replacement
   Environment: Python 3
   Build Command: 自动检测（使用render.yaml配置）
   Start Command: 自动检测（使用render.yaml配置）
   ```
6. **点击部署**：等待10-15分钟完成

### 2️⃣ 部署完成

✅ 获得公网访问地址：`https://your-app-name.onrender.com`  
✅ 自动HTTPS证书  
✅ 全球CDN加速  
✅ 完全免费使用  

## 🎯 其他快速选项

### Railway（最快）
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/new)

### Heroku（企业级）
```bash
heroku create dongjin-ai-replacement
git push heroku master
```

## 📱 使用说明

部署完成后，任何人都可以通过浏览器访问：
- 🖥️ 电脑端：完整功能
- 📱 手机端：响应式界面
- 🌐 全球访问：无地域限制

## ⚡ 性能说明

- **首次启动**：10-15分钟（自动下载AI模型）
- **后续访问**：秒级响应
- **并发支持**：多用户同时使用
- **存储空间**：10GB免费额度

## 🤖 自动模型下载

✅ **无需手动上传模型文件**  
✅ **首次启动自动下载SAM模型**  
✅ **支持断点续传和备用下载**  
✅ **下载进度实时显示**  

系统会在首次启动时自动从Facebook官方源下载SAM模型文件(2.5GB)，无需您手动处理！

## 🔧 部署验证

### 健康检查
部署完成后，访问 `https://your-app.onrender.com/health` 检查状态：
```json
{
  "status": "healthy",
  "models_loaded": true,
  "replacement_images": 80,
  "device": "cpu"
}
```

### 自动化检查
使用提供的检查脚本：
```bash
python check_deployment.py https://your-app.onrender.com
```

## 🔧 故障排除

### 常见部署问题
1. **Git依赖错误**：已修复，系统会自动安装Git
2. **Gunicorn未找到**：已修复，render.yaml现在使用requirements-cloud.txt安装依赖
   - 错误信息：`bash: line 1: gunicorn: command not found`
   - 解决方案：确保buildCommand使用`pip install -r requirements-cloud.txt`
3. **模型下载失败**：系统会自动重试3次
4. **内存不足**：已优化内存使用和清理
5. **端口绑定问题**：应用已配置绑定到0.0.0.0:$PORT

### 部署状态检查
- ✅ **构建成功**：依赖安装完成
- ⏳ **初始化中**：正在下载模型（10-15分钟）
- ✅ **运行正常**：可以正常使用
- ❌ **部署失败**：查看部署日志

### 性能优化
- **Gunicorn配置**：单进程，300秒超时
- **内存管理**：自动垃圾回收
- **模型缓存**：永久存储，避免重复下载
- **错误重试**：网络问题自动重试

**部署日志关键信息**：
```
步骤 1/4: 下载SAM模型...
步骤 2/4: 加载SAM模型...
步骤 3/4: 创建掩码生成器...
步骤 4/4: 加载DINOv2模型...
✅ 模型初始化完成!
```

---
**💡 提示**：首次部署需要自动下载2.5GB的AI模型，请耐心等待！模型下载完成后会永久保存，后续启动将秒级完成。