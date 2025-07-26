@echo off
echo 正在启动侗锦AI替换系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查必要文件
if not exist "sam_vit_h_4b8939.pth" (
    echo 错误：未找到SAM模型文件 sam_vit_h_4b8939.pth
    echo 请从以下链接下载：
    echo https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    pause
    exit /b 1
)

if not exist "replacement_dataset1" (
    echo 错误：未找到替换图案文件夹 replacement_dataset1
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查Python依赖包...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误：依赖包安装失败
        pause
        exit /b 1
    )
)

echo.
echo 系统检查完成，正在启动服务器...
echo.
echo 请在浏览器中访问：http://localhost:5000
echo 按 Ctrl+C 停止服务器
echo.

REM 启动Flask应用
python app.py

pause