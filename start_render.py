#!/usr/bin/env python3
"""
Render 部署启动脚本
优化模型加载和内存管理
"""
import os
import sys
import time
import gc
import signal
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置环境变量"""
    # 设置 PyTorch 相关环境变量
    os.environ['TORCH_HOME'] = '/opt/render/project/src/models'
    os.environ['TRANSFORMERS_CACHE'] = '/opt/render/project/src/models'
    os.environ['HF_HOME'] = '/opt/render/project/src/models'
    
    # 内存优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    logger.info("环境变量设置完成")

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在清理...")
    gc.collect()
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("启动侗锦AI替换系统...")
    
    # 设置环境
    setup_environment()
    
    # 导入并启动应用
    try:
        from app import app, initialize_models
        
        logger.info("初始化模型...")
        initialize_models()
        
        # 获取端口
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        
        logger.info(f"在端口 {port} 启动应用...")
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()