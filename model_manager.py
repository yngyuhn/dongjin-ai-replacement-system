"""
模型管理器 - 支持AWS S3存储和按需加载
解决Render部署内存限制问题
"""
import os
import boto3
import torch
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict
import gc
import time

# 直接导入f3模块
import f3
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class ModelManager:
    """模型管理器 - 支持S3存储和本地缓存"""
    
    def __init__(self):
        self.sam_model = None
        self.mask_generator = None
        self.dinov2_model = None
        self.dinov2_transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # S3 配置 (可通过环境变量配置)
        self.s3_bucket = os.getenv('S3_BUCKET', 'your-model-bucket')
        self.s3_prefix = os.getenv('S3_PREFIX', 'models/')
        
        # 本地缓存目录
        self.cache_dir = Path(os.getenv('MODEL_CACHE_DIR', './model_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_configs = {
            'sam_vit_h': {
                's3_key': f'{self.s3_prefix}sam_vit_h_4b8939.pth',
                'filename': 'sam_vit_h_4b8939.pth',
                'size_mb': 2500,  # 约2.5GB
                'checksum': 'a7bf3b02f3ebf1267aba913ff637d9a2'  # MD5校验
            }
        }
        
        # 是否启用S3 (如果环境变量配置了AWS凭证)
        self.use_s3 = self._check_s3_available()
        
        print(f"模型管理器初始化 - S3启用: {self.use_s3}, 设备: {self.device}")
    
    def _check_s3_available(self) -> bool:
        """检查S3是否可用"""
        try:
            # 检查AWS凭证是否配置
            aws_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if not aws_key or not aws_secret:
                print("AWS凭证未配置，使用本地模式")
                return False
                
            # 测试S3连接
            s3 = boto3.client('s3')
            s3.head_bucket(Bucket=self.s3_bucket)
            print(f"S3连接成功 - Bucket: {self.s3_bucket}")
            return True
            
        except Exception as e:
            print(f"S3不可用，使用本地模式: {e}")
            return False
    
    def _calculate_file_checksum(self, filepath: str) -> str:
        """计算文件MD5校验和"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _download_from_s3(self, s3_key: str, local_path: str) -> bool:
        """从S3下载模型文件"""
        try:
            s3 = boto3.client('s3')
            
            print(f"正在从S3下载模型: {s3_key}")
            start_time = time.time()
            
            # 下载文件
            s3.download_file(
                self.s3_bucket, 
                s3_key, 
                local_path
            )
            
            download_time = time.time() - start_time
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            
            print(f"S3下载完成: {file_size_mb:.1f}MB, 耗时: {download_time:.1f}秒")
            return True
            
        except Exception as e:
            print(f"S3下载失败: {e}")
            return False
    
    def _download_from_url(self, url: str, local_path: str, max_retries: int = 3) -> bool:
        """从URL下载模型文件（备用方案）- 增强版本"""
        import urllib.request
        import requests
        import socket
        from urllib3.exceptions import ReadTimeoutError
        
        for attempt in range(max_retries):
            try:
                print(f"正在从URL下载模型 (尝试 {attempt + 1}/{max_retries}): {url}")
                start_time = time.time()
                
                # 方法1: 使用requests（推荐）
                if attempt < 2:  # 前两次尝试用requests
                    try:
                        # 增加超时时间和连接配置
                        session = requests.Session()
                        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
                        
                        response = session.get(
                            url, 
                            stream=True, 
                            timeout=(30, 300),  # (连接超时, 读取超时)
                            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        )
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        chunk_size = 64 * 1024  # 增大chunk size到64KB
                        last_progress_time = time.time()
                        
                        with open(local_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    
                                    # 控制进度显示频率
                                    current_time = time.time()
                                    if current_time - last_progress_time > 2:  # 每2秒显示一次
                                        if total_size > 0:
                                            percent = (downloaded / total_size) * 100
                                            speed_mb = downloaded / (1024 * 1024) / (current_time - start_time)
                                            print(f"\r下载进度: {percent:.1f}% ({speed_mb:.1f}MB/s)", end='', flush=True)
                                        last_progress_time = current_time
                        
                        print()  # 换行
                        break  # 成功下载，退出重试循环
                        
                    except (requests.exceptions.Timeout, ReadTimeoutError, socket.timeout) as e:
                        print(f"\nrequests下载超时 (尝试 {attempt + 1}): {e}")
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        if attempt < max_retries - 1:
                            print(f"等待 {(attempt + 1) * 5} 秒后重试...")
                            time.sleep((attempt + 1) * 5)
                        continue
                        
                    except Exception as e:
                        print(f"\nrequests下载失败 (尝试 {attempt + 1}): {e}")
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        continue
                
                # 方法2: 使用urllib作为最后备选
                else:
                    print("使用urllib进行最后尝试...")
                    try:
                        # 自定义进度回调
                        def progress_hook(block_num, block_size, total_size):
                            if total_size > 0:
                                percent = min(100, (block_num * block_size / total_size) * 100)
                                print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
                        
                        urllib.request.urlretrieve(url, local_path, progress_hook)
                        print()  # 换行
                        break
                        
                    except Exception as e:
                        print(f"\nurllib下载失败: {e}")
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        continue
                
            except Exception as e:
                print(f"\n下载尝试 {attempt + 1} 失败: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        # 检查下载是否成功
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:  # 至少1MB
            download_time = time.time() - start_time
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✅ URL下载完成: {file_size_mb:.1f}MB, 耗时: {download_time:.1f}秒")
            return True
        else:
            print(f"❌ {max_retries} 次尝试后下载失败")
            return False
    
    def _get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型文件路径，必要时下载"""
        if model_name not in self.model_configs:
            raise ValueError(f"未知模型: {model_name}")
        
        config = self.model_configs[model_name]
        local_path = self.cache_dir / config['filename']
        
        # 检查本地缓存
        if local_path.exists():
            # 验证文件完整性
            if config.get('checksum'):
                actual_checksum = self._calculate_file_checksum(str(local_path))
                if actual_checksum == config['checksum']:
                    print(f"使用缓存模型: {local_path}")
                    return str(local_path)
                else:
                    print(f"缓存文件校验失败，重新下载: {local_path}")
                    local_path.unlink()
            else:
                # 简单大小检查
                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                if file_size_mb >= config['size_mb'] * 0.95:  # 允许5%误差
                    print(f"使用缓存模型: {local_path}")
                    return str(local_path)
                else:
                    print(f"缓存文件大小异常，重新下载: {local_path}")
                    local_path.unlink()
        
        # 下载模型
        download_success = False
        
        if self.use_s3:
            # 优先从S3下载
            download_success = self._download_from_s3(config['s3_key'], str(local_path))
        
        if not download_success:
            # 备用方案：从官方URL下载
            if model_name == 'sam_vit_h':
                sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                download_success = self._download_from_url(sam_url, str(local_path))
        
        if download_success and local_path.exists():
            return str(local_path)
        else:
            raise RuntimeError(f"无法下载模型: {model_name}")
    
    def load_sam_model(self, force_reload: bool = False) -> Tuple[object, object]:
        """按需加载SAM模型"""
        if self.sam_model is not None and self.mask_generator is not None and not force_reload:
            print("SAM模型已加载，直接返回")
            return self.sam_model, self.mask_generator
        
        # 清理之前的模型
        if force_reload:
            self.unload_sam_model()
        
        print("正在加载SAM模型...")
        
        try:
            # 获取模型文件
            model_path = self._get_model_path('sam_vit_h')
            
            # 加载模型
            model_type = "vit_h"
            print(f"从文件加载SAM模型: {model_path}")
            
            self.sam_model = sam_model_registry[model_type](checkpoint=model_path).to(self.device)
            
            # 创建掩码生成器
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=64,
                points_per_batch=32,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.8,
                min_mask_region_area=64,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
            )
            
            print("✅ SAM模型加载完成")
            return self.sam_model, self.mask_generator
            
        except Exception as e:
            print(f"❌ SAM模型加载失败: {e}")
            raise
    
    def load_dinov2_model(self, force_reload: bool = False) -> Tuple[object, object]:
        """按需加载DINOv2模型"""
        if self.dinov2_model is not None and self.dinov2_transform is not None and not force_reload:
            print("DINOv2模型已加载，直接返回")
            return self.dinov2_model, self.dinov2_transform
        
        # 清理之前的模型
        if force_reload:
            self.unload_dinov2_model()
        
        print("正在加载DINOv2模型...")
        
        try:
            self.dinov2_model, self.dinov2_transform = f3.load_dinov2_model(self.device, "dinov2_vitb14")
            print("✅ DINOv2模型加载完成")
            return self.dinov2_model, self.dinov2_transform
            
        except Exception as e:
            print(f"❌ DINOv2模型加载失败: {e}")
            raise
    
    def unload_sam_model(self):
        """卸载SAM模型以释放内存"""
        if self.sam_model is not None:
            print("卸载SAM模型...")
            del self.sam_model
            del self.mask_generator
            self.sam_model = None
            self.mask_generator = None
        
        self._cleanup_memory()
    
    def unload_dinov2_model(self):
        """卸载DINOv2模型以释放内存"""
        if self.dinov2_model is not None:
            print("卸载DINOv2模型...")
            del self.dinov2_model
            del self.dinov2_transform
            self.dinov2_model = None
            self.dinov2_transform = None
        
        self._cleanup_memory()
    
    def unload_all_models(self):
        """卸载所有模型"""
        print("卸载所有模型...")
        self.unload_sam_model()
        self.unload_dinov2_model()
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, str]:
        """获取内存使用情况"""
        import psutil
        
        # 系统内存
        memory = psutil.virtual_memory()
        
        result = {
            'system_total': f"{memory.total / (1024**3):.1f} GB",
            'system_used': f"{memory.used / (1024**3):.1f} GB",
            'system_percent': f"{memory.percent:.1f}%"
        }
        
        # GPU内存
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_cached = torch.cuda.memory_reserved(0)
            
            result.update({
                'gpu_total': f"{gpu_memory / (1024**3):.1f} GB",
                'gpu_allocated': f"{gpu_allocated / (1024**3):.1f} GB", 
                'gpu_cached': f"{gpu_cached / (1024**3):.1f} GB"
            })
        
        return result

# 全局模型管理器实例
model_manager = ModelManager()