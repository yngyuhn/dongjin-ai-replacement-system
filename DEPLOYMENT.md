# 侗锦AI智能替换系统 - 远程部署指南

## 🚀 最简洁部署方案

### 方案一：Render（推荐）- 免费部署

1. **注册Render账号**
   - 访问 [render.com](https://render.com)
   - 使用GitHub账号登录

2. **连接GitHub仓库**
   - 在Render控制台点击 "New +"
   - 选择 "Web Service"
   - 连接您的GitHub仓库：`yngyuhn/dongjin-ai-replacement-system`

3. **配置部署设置**
   ```
   Name: dongjin-ai-replacement
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python app.py
   ```

4. **等待部署完成**
   - 首次部署需要10-15分钟（下载模型）
   - 部署成功后获得公网访问地址

### 方案二：Railway - 一键部署

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

1. 点击上方按钮
2. 连接GitHub仓库
3. 自动部署完成

### 方案三：Heroku - 企业级

1. 安装Heroku CLI
2. 执行部署命令：
   ```bash
   heroku create dongjin-ai-replacement
   git push heroku master
   ```

## 📋 部署注意事项

### 模型文件处理
- SAM模型文件较大（2.5GB），首次部署需要时间
- 建议使用云存储或CDN加速模型下载

### 内存要求
- 最低：4GB RAM
- 推荐：8GB RAM
- GPU支持：可选（CPU也可运行）

### 环境变量配置
```
PYTHON_VERSION=3.9.16
FLASK_ENV=production
```

## 🔧 本地测试部署配置

运行以下命令测试Docker部署：
```bash
# 构建镜像
docker build -t dongjin-ai .

# 运行容器
docker run -p 5000:5000 dongjin-ai
```

## 📱 移动端适配

项目已包含响应式设计，支持：
- 手机浏览器访问
- 平板设备使用
- 桌面端完整功能

## 🌐 域名绑定

部署完成后可以：
1. 使用平台提供的免费域名
2. 绑定自定义域名
3. 配置HTTPS证书（自动）

## 💡 性能优化建议

1. **启用缓存**：静态文件CDN加速
2. **压缩图片**：自动图片压缩
3. **负载均衡**：多实例部署
4. **监控告警**：性能监控设置