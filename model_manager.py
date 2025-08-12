"""
模型管理器 - 实现外部存储和按需加载
解决Render内存限制问题
"""

import os
import gc
import time
import hashlib
import tempfile
import requests
import torch
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器，支持外部存储和按需加载"""
    
    def __init__(self, cache_dir: str = None, max_memory_mb: int = None):
        # 从环境变量获取配置，如果没有则使用默认值
        self.cache_dir = cache_dir or os.getenv('LOCAL_CACHE_DIR', './model_cache')
        self.max_memory_mb = max_memory_mb or int(os.getenv('MAX_MEMORY_USAGE_MB', '400'))
        self.enable_unloading = os.getenv('ENABLE_MODEL_UNLOADING', 'true').lower() == 'true'
        
        # AWS S3 配置（可选）
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        
        self.models = {}
        self.model_configs = {
            'sam_vit_h': {
                'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                'filename': 'sam_vit_h_4b8939.pth',
                'expected_size': 2564550879,  # 约2.5GB
                'cache_dir': self.cache_dir
            }
        }
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"ModelManager初始化: 缓存目录={self.cache_dir}, 最大内存={self.max_memory_mb}MB, 自动卸载={self.enable_unloading}")
        
        # 如果配置了S3，记录日志
        if self.s3_bucket:
            logger.info(f"S3存储已配置: bucket={self.s3_bucket}, region={self.aws_region}")
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型文件路径"""
        if model_name not in self.model_configs:
            raise ValueError(f"未知模型: {model_name}")
        
        config = self.model_configs[model_name]
        return os.path.join(config['cache_dir'], config['filename'])
    
    def is_model_cached(self, model_name: str) -> bool:
        """检查模型是否已缓存"""
        model_path = self.get_model_path(model_name)
        if not os.path.exists(model_path):
            return False
        
        # 验证文件大小
        config = self.model_configs[model_name]
        file_size = os.path.getsize(model_path)
        expected_size = config['expected_size']
        
        # 允许5%的误差
        return file_size >= expected_size * 0.95
    
    def download_from_s3(self, s3_path: str, local_path: str) -> bool:
        """从S3下载文件"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # 创建S3客户端
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            logger.info(f"从S3下载: s3://{self.s3_bucket}/{s3_path} -> {local_path}")
            
            # 下载文件
            s3_client.download_file(self.s3_bucket, s3_path, local_path)
            
            logger.info(f"S3下载完成: {local_path}")
            return True
            
        except ImportError:
            logger.error("boto3未安装，无法使用S3存储")
            return False
        except ClientError as e:
            logger.error(f"S3下载失败: {e}")
            return False
        except Exception as e:
            logger.error(f"S3下载异常: {e}")
            return False

    def download_model(self, model_name: str, force_download: bool = False) -> str:
        """下载模型文件"""
        if model_name not in self.model_configs:
            raise ValueError(f"未知模型: {model_name}")
        
        config = self.model_configs[model_name]
        model_path = self.get_model_path(model_name)
        
        # 检查是否需要下载
        if not force_download and self.is_model_cached(model_name):
            logger.info(f"模型 {model_name} 已缓存，跳过下载")
            return model_path
        
        # 删除不完整的文件
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # 首先尝试从S3下载（如果配置了）
        if self.s3_bucket:
            s3_path = os.getenv(f'{model_name.upper()}_MODEL_S3_PATH')
            if s3_path and self.download_from_s3(s3_path, model_path):
                # 验证下载的文件
                file_size = os.path.getsize(model_path)
                if file_size >= config['expected_size'] * 0.95:
                    logger.info(f"模型 {model_name} 从S3下载完成! 文件大小: {file_size // (1024*1024)}MB")
                    return model_path
                else:
                    logger.warning(f"S3下载的文件大小异常，尝试从URL下载")
                    os.remove(model_path)
            else:
                logger.warning(f"S3下载失败，尝试从URL下载: {model_name}")
        
        logger.info(f"开始下载模型 {model_name}...")
        logger.info(f"URL: {config['url']}")
        logger.info(f"目标路径: {model_path}")
        
        # 从URL下载文件
        try:
            response = requests.get(config['url'], stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # 每10MB打印一次
                                logger.info(f"下载进度: {percent:.1f}% ({downloaded // (1024*1024)}MB/{total_size // (1024*1024)}MB)")
            
            # 验证下载的文件
            file_size = os.path.getsize(model_path)
            if file_size < config['expected_size'] * 0.95:
                raise Exception(f"下载的文件大小异常: {file_size} bytes, 期望: {config['expected_size']} bytes")
            
            logger.info(f"模型 {model_name} 下载完成! 文件大小: {file_size // (1024*1024)}MB")
            return model_path
            
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception(f"模型下载失败: {e}")
    
    def load_model_lazy(self, model_name: str, loader_func, *args, **kwargs):
        """懒加载模型 - 只在需要时加载到内存"""
        if model_name in self.models:
            logger.info(f"模型 {model_name} 已在内存中")
            return self.models[model_name]
        
        # 确保模型文件存在
        model_path = self.download_model(model_name)
        
        logger.info(f"正在加载模型 {model_name} 到内存...")
        
        # 清理内存
        self.cleanup_memory()
        
        try:
            # 加载模型
            model = loader_func(model_path, *args, **kwargs)
            self.models[model_name] = model
            
            logger.info(f"模型 {model_name} 加载完成")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def unload_model(self, model_name: str) -> bool:
        """卸载模型以释放内存"""
        if model_name in self.models:
            del self.models[model_name]
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"模型已卸载: {model_name}")
            return True
        return False
    
    def auto_manage_memory(self):
        """自动内存管理"""
        if not self.enable_unloading:
            return
        
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_mb:
            logger.warning(f"内存使用过高: {current_memory}MB > {self.max_memory_mb}MB，开始自动卸载模型")
            
            # 按使用时间排序，优先卸载最久未使用的模型
            # 这里简化处理，可以根据需要实现更复杂的策略
            for model_name in list(self.models.keys()):
                if self.unload_model(model_name):
                    current_memory = self.get_memory_usage()
                    logger.info(f"卸载后内存使用: {current_memory}MB")
                    if current_memory <= self.max_memory_mb * 0.8:  # 留20%缓冲
                        break
    
    def unload_all_models(self):
        """卸载所有模型"""
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        logger.info("所有模型已卸载")
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'loaded_models': list(self.models.keys())
        }
        
        if torch.cuda.is_available():
            result['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        
        return result
    
    def clear_cache(self):
        """清理缓存文件"""
        for model_name, config in self.model_configs.items():
            model_path = self.get_model_path(model_name)
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"已删除缓存文件: {model_path}")

# 全局模型管理器实例
model_manager = ModelManager()

def download_sam_model_optimized():
    """优化的SAM模型下载函数"""
    return model_manager.download_model('sam_vit_h')

def load_sam_model_lazy(device="cpu"):
    """懒加载SAM模型"""
    from segment_anything import sam_model_registry
    
    def loader_func(model_path):
        return sam_model_registry["vit_h"](checkpoint=model_path).to(device)
    
    return model_manager.load_model_lazy('sam_vit_h', loader_func)

def get_memory_status():
    """获取内存状态"""
    return model_manager.get_memory_usage()