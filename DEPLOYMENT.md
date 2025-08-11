# 侗锦AI替换系统 - 部署指南

## 🚀 在线部署方案

### 方案1: Render (推荐 - 免费)

1. **准备工作**
   - 确保代码已推送到Gitee仓库
   - 注册 [Render](https://render.com) 账号

2. **部署步骤**
   - 登录Render控制台
   - 点击 "New" → "Web Service"
   - 连接您的Git仓库: `https://gitee.com/yngyuhn/dongjin-ai-replacement-system.git`
   - 配置部署设置:
     - **Name**: `dongjin-ai-replacement`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - 点击 "Create Web Service"

3. **环境变量设置**
   ```
   FLASK_ENV=production
   PYTHON_VERSION=3.9.18
   ```

### 方案2: Railway (免费额度)

1. **部署步骤**
   - 访问 [Railway](https://railway.app)
   - 使用GitHub/GitLab连接您的仓库
   - Railway会自动检测Python项目并部署

### 方案3: Heroku (付费)

1. **部署步骤**
   - 安装Heroku CLI
   - 登录: `heroku login`
   - 创建应用: `heroku create your-app-name`
   - 推送代码: `git push heroku master`

## 📋 部署文件说明

- `requirements.txt`: Python依赖包列表
- `Procfile`: Heroku部署配置
- `render.yaml`: Render部署配置
- `app.py`: 主应用文件（已配置生产环境）

## ⚠️ 注意事项

1. **内存要求**: SAM模型较大，建议选择至少1GB内存的服务
2. **启动时间**: 首次启动需要下载模型，可能需要几分钟
3. **文件上传**: 确保平台支持文件上传功能
4. **超时设置**: 图像处理可能需要较长时间，注意超时配置

## 🔧 本地测试

部署前可以本地测试生产环境配置:

```bash
# 设置环境变量
set FLASK_ENV=production
set PORT=5000

# 使用gunicorn运行
gunicorn --bind 0.0.0.0:5000 app:app
```

## 📞 技术支持

如遇到部署问题，请检查:
1. 依赖包是否正确安装
2. 环境变量是否正确设置
3. 服务器内存是否充足
4. 网络连接是否正常