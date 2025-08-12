#!/usr/bin/env python3
"""
Render部署状态检查脚本
用于验证部署是否成功
"""

import requests
import sys
import time

def check_deployment(url, max_retries=10, delay=30):
    """检查部署状态"""
    print(f"检查部署状态: {url}")
    
    for attempt in range(max_retries):
        try:
            print(f"尝试 {attempt + 1}/{max_retries}...")
            
            # 检查健康端点
            health_response = requests.get(f"{url}/health", timeout=30)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                print("✅ 健康检查通过!")
                print(f"状态: {health_data.get('status')}")
                print(f"模型已加载: {health_data.get('models_loaded')}")
                print(f"替换图像数量: {health_data.get('replacement_images')}")
                print(f"设备: {health_data.get('device')}")
                return True
                
            elif health_response.status_code == 503:
                health_data = health_response.json()
                print(f"⏳ 应用正在初始化... 状态: {health_data.get('status')}")
                
            else:
                print(f"❌ 健康检查失败: HTTP {health_response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 连接失败: {e}")
            
        if attempt < max_retries - 1:
            print(f"等待 {delay} 秒后重试...")
            time.sleep(delay)
    
    print("❌ 部署检查失败")
    return False

def main():
    if len(sys.argv) != 2:
        print("用法: python check_deployment.py <URL>")
        print("示例: python check_deployment.py https://your-app.onrender.com")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    
    success = check_deployment(url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()