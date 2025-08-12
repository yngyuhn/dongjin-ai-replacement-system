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
   Build Command: pip install -r requirements.txt
   Start Command: python app.py
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

## 🔧 故障排除

如果部署失败，检查：
1. GitHub仓库是否公开
2. requirements.txt文件是否存在
3. 等待模型下载完成（查看部署日志）
4. 确保网络连接稳定

**常见问题**：
- 如果模型下载超时，系统会自动重试
- 部署日志中会显示下载进度
- 模型下载完成后会自动开始服务

---
**💡 提示**：首次部署需要自动下载2.5GB的AI模型，请耐心等待！模型下载完成后会永久保存，后续启动将秒级完成。