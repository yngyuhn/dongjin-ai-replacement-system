#!/bin/bash

echo "🚀 侗锦AI智能替换系统 - 一键部署脚本"
echo "=================================="

# 检查Git状态
if [ ! -d ".git" ]; then
    echo "❌ 错误：当前目录不是Git仓库"
    exit 1
fi

# 提交当前更改
echo "📝 提交部署配置文件..."
git add .
git commit -m "添加部署配置文件：支持Render、Railway、Docker部署"

# 推送到远程仓库
echo "📤 推送到远程仓库..."
git push origin master

echo "✅ 部署配置已推送到仓库！"
echo ""
echo "🌐 现在可以选择以下部署方式："
echo ""
echo "1️⃣  Render (推荐 - 免费):"
echo "   访问: https://render.com"
echo "   连接仓库: yngyuhn/dongjin-ai-replacement-system"
echo ""
echo "2️⃣  Railway (快速):"
echo "   访问: https://railway.app"
echo "   一键部署模板"
echo ""
echo "3️⃣  Docker (本地测试):"
echo "   docker build -t dongjin-ai ."
echo "   docker run -p 5000:5000 dongjin-ai"
echo ""
echo "📖 详细说明请查看 DEPLOYMENT.md 文件"