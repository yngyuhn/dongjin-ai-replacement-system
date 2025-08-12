import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import time
import torchvision.transforms as T
from sklearn.cluster import KMeans
from scipy import ndimage
import base64
import io
from flask import Flask, render_template, request, jsonify, send_file
import json
import tempfile
import shutil
import gc
import logging

# 导入模块
import f3
from model_manager import ModelManager, load_sam_model_lazy, get_memory_status

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化模型管理器
model_manager = ModelManager()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量存储模型和数据 - 改为按需加载
sam_model = None
mask_generator = None
dinov2_model = None
dinov2_transform = None
replacement_embeddings = []
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型加载状态
models_loaded = {
    'sam': False,
    'dinov2': False,
    'replacement_images': False
}

# 存储当前处理的数据
current_session = {
    'image': None,
    'masks': None,
    'mask_features': None,
    'root_nodes': None,
    'canvas': None,
    'original_gray': None,
    'history': [],  # 操作历史记录
    'max_history': 10  # 最大历史记录数
}

def ensure_sam_model():
    """确保SAM模型已加载（按需加载）"""
    global sam_model, mask_generator, models_loaded
    
    if models_loaded['sam'] and sam_model is not None:
        return sam_model, mask_generator
    
    try:
        logger.info("按需加载SAM模型...")
        
        # 清理内存
        model_manager.cleanup_memory()
        
        # 加载SAM模型
        sam_model = load_sam_model_lazy(device)
        
        # 创建掩码生成器
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,  # 减少参数以节省内存
            points_per_batch=16,  # 减少批次大小
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            min_mask_region_area=64,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        
        models_loaded['sam'] = True
        logger.info("✅ SAM模型加载完成!")
        
        return sam_model, mask_generator
        
    except Exception as e:
        logger.error(f"❌ SAM模型加载失败: {e}")
        raise Exception(f"SAM模型加载失败: {e}")

def ensure_dinov2_model():
    """确保DINOv2模型已加载（按需加载）"""
    global dinov2_model, dinov2_transform, models_loaded
    
    if models_loaded['dinov2'] and dinov2_model is not None:
        return dinov2_model, dinov2_transform
    
    try:
        logger.info("按需加载DINOv2模型...")
        
        # 清理内存
        model_manager.cleanup_memory()
        
        # 加载DINOv2模型
        dinov2_model, dinov2_transform = f3.load_dinov2_model(device, "dinov2_vitb14")
        
        models_loaded['dinov2'] = True
        logger.info("✅ DINOv2模型加载完成!")
        
        return dinov2_model, dinov2_transform
        
    except Exception as e:
        logger.error(f"❌ DINOv2模型加载失败: {e}")
        raise Exception(f"DINOv2模型加载失败: {e}")

def initialize_models():
    """轻量级初始化 - 不预加载模型"""
    logger.info("🚀 应用启动 - 使用按需加载模式")
    logger.info("模型将在首次使用时自动下载和加载")
    
    # 检查内存状态
    memory_status = get_memory_status()
    logger.info(f"当前内存使用: {memory_status['rss_mb']:.1f}MB")
    
    # 预加载替换图像特征（较小，可以预加载）
    try:
        load_replacement_images()
    except Exception as e:
        logger.warning(f"替换图像加载失败: {e}")
    
    logger.info("✅ 应用初始化完成!")

def load_replacement_images():
    """加载替换图像特征（延迟加载）"""
    global replacement_embeddings, models_loaded
    
    if models_loaded['replacement_images'] and replacement_embeddings:
        return replacement_embeddings
    
    replacement_folder = "replacement_dataset1"
    if not os.path.exists(replacement_folder):
        logger.warning(f"替换图像文件夹不存在: {replacement_folder}")
        return []
    
    replacement_paths = []
    for root, _, files in os.walk(replacement_folder):
        for f in files:
            full_path = os.path.join(root, f)
            if f3.is_valid_image(full_path):
                replacement_paths.append(full_path)
    
    logger.info(f"找到 {len(replacement_paths)} 张替换图像")
    
    if replacement_paths:
        # 确保DINOv2模型已加载
        dinov2_model, dinov2_transform = ensure_dinov2_model()
        
        replacement_embeddings = f3.get_replacement_image_embeddings(
            replacement_paths, dinov2_transform, dinov2_model, device)
        
        models_loaded['replacement_images'] = True
    
    logger.info(f"成功加载 {len(replacement_embeddings)} 张替换图像特征")
    return replacement_embeddings

def save_history_state(operation_name):
    """保存当前状态到历史记录"""
    if current_session['canvas'] is not None:
        # 创建历史记录条目
        history_entry = {
            'operation': operation_name,
            'canvas': current_session['canvas'].copy(),
            'timestamp': time.time()
        }
        
        # 添加到历史记录
        current_session['history'].append(history_entry)
        
        # 限制历史记录数量
        if len(current_session['history']) > current_session['max_history']:
            current_session['history'].pop(0)

def image_to_base64(image):
    """将图像转换为base64编码"""
    if image is None:
        return None
    
    # 确保图像是uint8格式
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 如果是灰度图，转换为RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(image)
    
    # 转换为base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        # 检查模型是否已加载
        models_loaded = (sam_model is not None and 
                        mask_generator is not None and 
                        dinov2_model is not None and 
                        dinov2_transform is not None)
        
        status = {
            'status': 'healthy' if models_loaded else 'initializing',
            'models_loaded': models_loaded,
            'replacement_images': len(replacement_embeddings) if replacement_embeddings else 0,
            'device': str(device)
        }
        
        return jsonify(status), 200 if models_loaded else 503
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    """上传图像并进行初始处理"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图像'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 读取图像
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({'error': '无法解码图像'}), 400
        
        # 转换为灰度图
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 如果图像太大，适当缩小
        max_dim = 2048
        h, w = image_gray.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_gray = cv2.resize(image_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 存储到当前会话
        current_session['image'] = image_bgr
        current_session['original_gray'] = image_gray.copy()
        current_session['canvas'] = np.ones_like(image_gray) * 255
        current_session['history'] = []  # 重置历史记录
        
        # 初始化会话状态（不生成掩码）
        current_session['masks'] = None
        current_session['mask_features'] = None
        current_session['root_nodes'] = None
        current_session['history'] = []
        
        # 保存初始状态
        save_history_state("初始上传")
        
        # 返回原始图像（不生成掩码）
        original_b64 = image_to_base64(image_gray)
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'mask_overlay': None,  # 暂时没有掩码
            'mask_count': 0,
            'image_size': f"{image_gray.shape[1]}x{image_gray.shape[0]}",
            'message': '图像上传成功！请调整参数后点击"开始处理"生成掩码。'
        })
        
    except Exception as e:
        logger.error(f"上传处理错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    """处理图像替换"""
    try:
        data = request.get_json()
        
        # 获取参数
        confidence_threshold = float(data.get('confidence', 0.8))
        stability_threshold = float(data.get('stability', 0.8))
        similarity_threshold = float(data.get('similarity', 0.8))
        density_threshold = float(data.get('density', 0.5))
        
        if current_session['image'] is None:
            return jsonify({'error': '请先上传图像'}), 400
        
        # 自动内存管理
        model_manager.auto_manage_memory()
        
        # 确保SAM模型已加载
        sam_model, mask_generator = ensure_sam_model()
        
        # 重新创建掩码生成器（使用用户参数）
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,  # 减少内存使用
            points_per_batch=16,  # 减少内存使用
            pred_iou_thresh=confidence_threshold,
            stability_score_thresh=stability_threshold,
            min_mask_region_area=64,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        
        # 生成掩码（使用用户设置的参数）
        logger.info(f"使用用户参数生成掩码: 置信值={confidence_threshold}, 稳定值={stability_threshold}")
        image_rgb = cv2.cvtColor(current_session['image'], cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)
        
        if not masks:
            return jsonify({'error': '未能生成掩码，请尝试调整参数'}), 400
        
        current_session['masks'] = masks
        
        # 确保DINOv2模型已加载
        dinov2_model, dinov2_transform = ensure_dinov2_model()
        
        # 重新计算特征和树结构
        image_gray = current_session['original_gray']
        mask_features = f3.calculate_mask_features(image_gray, masks, dinov2_transform, dinov2_model, device)
        current_session['mask_features'] = mask_features
        
        root_nodes = f3.build_mask_tree(masks)
        current_session['root_nodes'] = root_nodes
        
        # 创建新画布
        canvas_gray = np.ones_like(image_gray) * 255
        
        # 保存处理前状态
        save_history_state("处理前状态")
        
        # 确保替换图像已加载
        replacement_embeddings = load_replacement_images()
        
        # 执行替换
        for root in root_nodes:
            f3.recursive_replace_strategy(
                root, canvas_gray, image_gray, replacement_embeddings,
                dinov2_transform, dinov2_model, device, mask_features,
                similarity_threshold, density_threshold
            )
        
        current_session['canvas'] = canvas_gray
        
        # 保存处理后状态
        save_history_state(f"自动处理(相似度:{similarity_threshold:.2f}, 密度:{density_threshold:.2f})")
        
        # 返回结果
        result_b64 = image_to_base64(canvas_gray)
        
        return jsonify({
            'success': True,
            'result_image': result_b64,
            'mask_count': len(masks)
        })
        
    except Exception as e:
        logger.error(f"处理错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

@app.route('/get_masks', methods=['GET'])
def get_masks():
    """获取所有掩码信息（过滤空掩码）"""
    try:
        if current_session['masks'] is None:
            return jsonify({'error': '没有可用的掩码'}), 400
        
        masks_info = []
        min_area_threshold = 100  # 最小面积阈值，过滤掉过小的掩码
        
        for i, mask_dict in enumerate(current_session['masks']):
            mask = mask_dict["segmentation"]
            area = np.sum(mask)
            
            # 过滤掉面积过小的掩码（空掩码或噪声掩码）
            if area < min_area_threshold:
                continue
            
            # 检查掩码是否有实际内容（不是全黑）
            mask_img = (mask.astype(np.uint8) * 255)
            if np.max(mask_img) == 0:  # 全黑掩码
                continue
            
            # 获取掩码边界框
            ys, xs = np.where(mask)
            if len(ys) == 0 or len(xs) == 0:
                continue
                
            bbox = {
                'x': int(xs.min()),
                'y': int(ys.min()),
                'width': int(xs.max() - xs.min() + 1),
                'height': int(ys.max() - ys.min() + 1)
            }
            
            # 获取掩码轮廓点（用于精确点击检测）
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_points = []
            if contours:
                # 选择最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                # 简化轮廓点
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                contour_points = [[int(point[0][0]), int(point[0][1])] for point in approx]
            
            mask_b64 = image_to_base64(mask_img)
            
            masks_info.append({
                'id': i,
                'area': int(area),
                'image': mask_b64,
                'bbox': bbox,
                'contour': contour_points
            })
        
        return jsonify({
            'success': True,
            'masks': masks_info
        })
        
    except Exception as e:
        logger.error(f"获取掩码错误: {e}")
        return jsonify({'error': f'获取掩码失败: {str(e)}'}), 500

@app.route('/delete_mask', methods=['POST'])
def delete_mask():
    """删除指定掩码区域"""
    try:
        data = request.get_json()
        mask_id = int(data.get('mask_id'))
        
        if current_session['masks'] is None:
            return jsonify({'error': '没有可用的掩码'}), 400
        
        if mask_id >= len(current_session['masks']):
            return jsonify({'error': '掩码ID无效'}), 400
        
        if current_session['canvas'] is None:
            return jsonify({'error': '没有可操作的画布'}), 400
        
        # 保存操作前状态
        save_history_state("删除掩码前")
        
        # 获取掩码
        mask = current_session['masks'][mask_id]["segmentation"]
        canvas_gray = current_session['canvas'].copy()
        
        # 将掩码区域设置为白色（删除效果）
        canvas_gray[mask] = 255
        
        current_session['canvas'] = canvas_gray
        
        # 保存操作后状态
        save_history_state(f"删除掩码{mask_id}")
        
        result_b64 = image_to_base64(canvas_gray)
        
        return jsonify({
            'success': True,
            'result_image': result_b64
        })
        
    except Exception as e:
        logger.error(f"删除掩码错误: {e}")
        return jsonify({'error': f'删除掩码失败: {str(e)}'}), 500

@app.route('/replace_manual', methods=['POST'])
def replace_manual():
    """手动替换指定掩码"""
    try:
        data = request.get_json()
        mask_id = int(data.get('mask_id'))
        replacement_name = data.get('replacement_name')
        
        if current_session['masks'] is None:
            return jsonify({'error': '没有可用的掩码'}), 400
        
        if mask_id >= len(current_session['masks']):
            return jsonify({'error': '掩码ID无效'}), 400
        
        # 确保替换图像已加载
        replacement_embeddings = load_replacement_images()
        
        # 找到指定的替换图像
        replacement_path = None
        for path, _ in replacement_embeddings:
            if os.path.basename(path) == replacement_name:
                replacement_path = path
                break
        
        if replacement_path is None:
            return jsonify({'error': '找不到指定的替换图像'}), 400
        
        # 保存操作前状态
        save_history_state("手动替换前")
        
        # 执行替换
        mask = current_session['masks'][mask_id]["segmentation"]
        image_gray = current_session['original_gray']
        canvas_gray = current_session['canvas'].copy()
        
        # 手动替换单个掩码
        success = manual_replace_mask(canvas_gray, image_gray, mask, replacement_path, mask_id)
        
        if success:
            current_session['canvas'] = canvas_gray
            # 保存操作后状态
            save_history_state(f"手动替换掩码{mask_id}")
            result_b64 = image_to_base64(canvas_gray)
            
            return jsonify({
                'success': True,
                'result_image': result_b64
            })
        else:
            return jsonify({'error': '替换失败'}), 500
        
    except Exception as e:
        logger.error(f"手动替换错误: {e}")
        return jsonify({'error': f'手动替换失败: {str(e)}'}), 500

def manual_replace_mask(canvas_gray, image_gray, mask, replacement_path, mask_index):
    """手动替换单个掩码"""
    try:
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return False
        
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        
        # 读取替换图像
        rep_img = f3.imread_unicode(replacement_path)
        if rep_img is None:
            return False
        
        # 转换为灰度图
        if len(rep_img.shape) == 3:
            rep_gray = cv2.cvtColor(rep_img, cv2.COLOR_BGR2GRAY)
        else:
            rep_gray = rep_img
        
        # 调整大小
        dst_h, dst_w = y1 - y0, x1 - x0
        rep = f3.resize_with_padding(rep_gray, (dst_h, dst_w))
        if rep is None:
            return False
        
        # 亮度调整
        roi_gray = image_gray[y0:y1, x0:x1]
        mask_region = mask[y0:y1, x0:x1]
        orig_region = roi_gray[mask_region]
        rep_adjusted = f3.adjust_brightness(rep, orig_region)
        
        # 应用掩码
        smoothed_mask = f3.smooth_mask(mask_region.astype(np.uint8))
        canvas_roi = canvas_gray[y0:y1, x0:x1]
        canvas_roi[smoothed_mask] = rep_adjusted[smoothed_mask]
        
        return True
        
    except Exception as e:
        logger.error(f"手动替换掩码失败: {e}")
        return False

@app.route('/get_replacements', methods=['GET'])
def get_replacements():
    """获取所有可用的替换图像"""
    try:
        # 确保替换图像已加载
        replacement_embeddings = load_replacement_images()
        
        replacements = []
        for path, _ in replacement_embeddings:
            name = os.path.basename(path)
            
            # 读取图像并转换为base64
            img = f3.imread_unicode(path)
            if img is not None:
                # 缩小图像用于预览
                h, w = img.shape[:2]
                if max(h, w) > 100:
                    scale = 100 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h))
                
                img_b64 = image_to_base64(img)
                replacements.append({
                    'name': name,
                    'image': img_b64
                })
        
        return jsonify({
            'success': True,
            'replacements': replacements
        })
        
    except Exception as e:
        logger.error(f"获取替换图像错误: {e}")
        return jsonify({'error': f'获取替换图像失败: {str(e)}'}), 500

@app.route('/download', methods=['GET'])
def download_result():
    """下载处理结果"""
    try:
        if current_session['canvas'] is None:
            return jsonify({'error': '没有可下载的结果'}), 400
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_file.name, current_session['canvas'])
        temp_file.close()
        
        return send_file(temp_file.name, as_attachment=True, download_name='dongji_result.png')
        
    except Exception as e:
        logger.error(f"下载错误: {e}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/undo', methods=['POST'])
def undo_operation():
    """撤销上一步操作"""
    try:
        if not current_session['history']:
            return jsonify({'error': '没有可撤销的操作'}), 400
        
        # 移除最后一个操作（当前状态）
        if len(current_session['history']) > 1:
            current_session['history'].pop()
            
            # 恢复到上一个状态
            last_state = current_session['history'][-1]
            current_session['canvas'] = last_state['canvas'].copy()
            
            result_b64 = image_to_base64(current_session['canvas'])
            
            return jsonify({
                'success': True,
                'result_image': result_b64,
                'operation': last_state['operation'],
                'can_undo': len(current_session['history']) > 1
            })
        else:
            return jsonify({'error': '已经是最初状态，无法继续撤销'}), 400
            
    except Exception as e:
        return jsonify({'error': f'撤销操作失败: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """获取操作历史"""
    try:
        history_list = []
        for i, entry in enumerate(current_session['history']):
            history_list.append({
                'index': i,
                'operation': entry['operation'],
                'timestamp': entry['timestamp']
            })
        
        return jsonify({
            'history': history_list,
            'can_undo': len(current_session['history']) > 1
        })
        
    except Exception as e:
        return jsonify({'error': f'获取历史失败: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("正在初始化侗锦AI替换系统...")
    initialize_models()
    logger.info("系统启动完成!")
    
    # 获取端口号（云平台会设置PORT环境变量）
    port = int(os.environ.get('PORT', 5000))
    
    # 生产环境关闭debug模式
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)