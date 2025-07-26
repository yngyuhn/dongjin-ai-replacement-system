import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from tqdm import tqdm
import time
import torchvision.transforms as T
from sklearn.cluster import KMeans
from scipy import ndimage

# ========== DINOv2模型加载 ==========
def load_dinov2_model(device="cuda", model_type="dinov2_vitb14"):
    print(f"加载DINOv2模型: {model_type}...")
    model = torch.hub.load('facebookresearch/dinov2', model_type)
    model = model.to(device)
    model.eval()
    
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return model, transform

# ========== 图像处理工具函数 ==========
def imread_unicode(path, bg_color=(255, 255, 255)):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ 无法解码图像: {path}")
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            alpha = a.astype(np.float32) / 255.0
            bg = np.full_like(img[:, :, :3], bg_color, dtype=np.uint8).astype(np.float32)
            fg = img[:, :, :3].astype(np.float32)
            img = (fg * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
        elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            print(f"⚠️ 图像尺寸无效: {path} (形状: {img.shape})")
            return None
        return img
    except Exception as e:
        print(f"读取图片失败: {path}, 错误: {e}")
        return None

def is_valid_image(path):
    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)
    return ext in {'.jpg', '.jpeg', '.png'} and not basename.startswith("._") and '__MACOSX' not in path

def resize_with_padding(img, target_size):
    try:
        h, w = img.shape[:2]
        th, tw = target_size
        if th <= 0 or tw <= 0:
            print(f"⚠️ 目标尺寸无效: {target_size}")
            return None
        if h <= 0 or w <= 0:
            print(f"⚠️ 原始图像尺寸无效: {img.shape}")
            return None
            
        scale_w = tw / w
        scale_h = th / h
        scale = min(scale_w, scale_h)
        
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            print(f"⚠️ 调整后尺寸无效: ({nw}, {nh})")
            return None
            
        interpolation = cv2.INTER_LANCZOS4 if scale < 1 else cv2.INTER_CUBIC
        img_resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
        
        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left
        
        if top < 0 or bottom < 0 or left < 0 or right < 0:
            print(f"⚠️ 边框尺寸无效: top={top}, bottom={bottom}, left={left}, right={right}")
            return None
            
        return cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_REFLECT)
    except Exception as e:
        print(f"调整图像大小失败: {e}")
        return None

# ========== 掩码处理函数 ==========
def clean_mask(mask, min_area_ratio=0.0000000000001):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, -1]
    if len(areas) == 0:
        return mask.astype(bool)
    max_area = np.max(areas)
    min_keep_area = np.mean(areas) * min_area_ratio
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, -1]
        if area >= min_keep_area:
            cleaned_mask[labels == i] = 1
    
    cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.medianBlur(closed, 3)
    
    return smoothed.astype(bool)

def smooth_mask(mask, kernel_size=3):
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.medianBlur(closed, 3)
    
    return smoothed.astype(bool)

# ========== 掩码树节点类 ==========
class MaskTreeNode:
    def __init__(self, idx, mask):
        self.idx = idx
        self.mask = mask
        self.area = np.sum(mask)
        self.parent = None
        self.children = []
        self.is_root = True
        self.brightness = None  # 存储亮度信息
    
    def add_child(self, child_node):
        child_node.parent = self
        child_node.is_root = False
        self.children.append(child_node)
    
    def __repr__(self):
        return f"MaskNode(idx={self.idx}, area={self.area}, children={len(self.children)}, is_root={self.is_root})"

# ========== 构建掩码树结构 ==========
def build_mask_tree(masks, min_overlap_ratio=0.95):
    sorted_indices = sorted(range(len(masks)), key=lambda i: np.sum(masks[i]["segmentation"]))
    
    nodes = [MaskTreeNode(i, masks[i]["segmentation"]) for i in range(len(masks))]
    
    assigned = [False] * len(masks)
    
    for i in sorted_indices:
        current_mask = masks[i]["segmentation"]
        current_area = np.sum(current_mask)
        
        # if current_area < 100:
        #     continue
            
        best_parent = None
        best_parent_area = float('inf')
        
        for j in range(len(masks)):
            if i == j or assigned[j]:
                continue
                
            candidate_mask = masks[j]["segmentation"]
            candidate_area = np.sum(candidate_mask)
            
            overlap = np.logical_and(current_mask, candidate_mask)
            overlap_area = np.sum(overlap)
            overlap_ratio = overlap_area / current_area
            
            if overlap_ratio >= min_overlap_ratio and candidate_area > current_area:
                if candidate_area < best_parent_area:
                    best_parent = nodes[j]
                    best_parent_area = candidate_area
        
        if best_parent is not None:
            best_parent.add_child(nodes[i])
            assigned[i] = True
    
    root_nodes = [node for node in nodes if node.is_root and np.sum(node.mask) > 100]
    
    print(f"构建掩码树完成: 总掩码数={len(masks)}, 根节点数={len(root_nodes)}")
    return root_nodes

# ========== 特征计算 ==========
def calculate_mask_features(image_gray, masks, transform, dinov2_model, device):
    print("计算掩码特征中...")
    mask_features = []
    
    for idx, mask_dict in enumerate(tqdm(masks, desc="计算掩码特征")):
        mask = mask_dict["segmentation"]
        ys, xs = np.where(mask)
        if len(ys) == 0:
            mask_features.append(None)
            continue
            
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        
        if y1 - y0 < 10 or x1 - x0 < 10:
            mask_features.append(None)
            continue
            
        roi_gray = image_gray[y0:y1, x0:x1]
        
        if roi_gray.size == 0:
            mask_features.append(None)
            continue
            
        try:
            masked_roi = np.zeros_like(roi_gray)
            mask_region = mask[y0:y1, x0:x1]
            masked_roi[mask_region] = roi_gray[mask_region]
            
            # 转换为三通道灰度图像
            masked_roi_rgb = np.stack([masked_roi]*3, axis=-1)
            roi_pil = Image.fromarray(masked_roi_rgb)
            
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features_dict = dinov2_model.forward_features(roi_tensor)
                patch_tokens = features_dict['x_norm_patchtokens']
                features = patch_tokens.mean(dim=1).cpu().numpy()
                
            mask_features.append(features)
        except Exception as e:
            print(f"计算掩码 {idx} 特征失败: {e}")
            mask_features.append(None)
            
    return mask_features

def calculate_cluster_similarity(features, indices):
    if len(indices) < 2:
        return 1.0
        
    valid_features = []
    for idx in indices:
        if features[idx] is not None:
            valid_features.append(features[idx])
            
    if len(valid_features) < 2:
        return 0.0
        
    total_sim = 0
    count = 0
    for i in range(len(valid_features)):
        for j in range(i+1, len(valid_features)):
            feat_i = torch.tensor(valid_features[i]).squeeze()
            feat_j = torch.tensor(valid_features[j]).squeeze()
            if feat_i.dim() == 0 or feat_j.dim() == 0:
                continue
                
            feat_i = feat_i.unsqueeze(0)
            feat_j = feat_j.unsqueeze(0)
            
            sim = torch.cosine_similarity(feat_i, feat_j, dim=1).item()
            total_sim += sim
            count += 1
            
    return total_sim / count if count > 0 else 0.0

# ========== 替换图像特征计算 ==========
def get_replacement_image_embeddings(image_paths, transform, dinov2_model, device):
    print("预计算替换图像特征中...")
    image_embeddings = []
    
    for path in tqdm(image_paths, desc="处理替换图像"):
        try:
            img = Image.open(path).convert('L')  # 转换为灰度
            # 转换为三通道灰度图像
            img_rgb = Image.merge('RGB', (img, img, img))
            img_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features_dict = dinov2_model.forward_features(img_tensor)
                patch_tokens = features_dict['x_norm_patchtokens']
                features = patch_tokens.mean(dim=1).cpu()
                
            image_embeddings.append((path, features))
        except Exception as e:
            print(f"处理替换图像失败: {path}, 错误: {e}")
    
    print(f"成功处理 {len(image_embeddings)}/{len(image_paths)} 张替换图像")
    return image_embeddings

# ========== 亮度调整函数 ==========
def adjust_brightness(source, target):
    """
    调整源图像亮度以匹配目标图像亮度
    """
    if source.size == 0 or target.size == 0:
        return source
        
    # 计算源图像和目标图像的平均亮度
    source_mean = np.mean(source)
    target_mean = np.mean(target)
    
    if source_mean < 1e-5 or target_mean < 1e-5:
        return source
        
    # 计算缩放因子
    ratio = target_mean / source_mean
    
    # 限制调整幅度以避免失真
    ratio = np.clip(ratio, 0.7, 1.3)
    
    # 应用亮度调整
    adjusted = source * ratio
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

# ========== 改进的替换逻辑 ==========
def replace_one_mask(canvas_gray, image_gray, mask, replacement_embeddings, 
                   transform, dinov2_model, device, mask_index=None):
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        if mask_index is not None:
            print(f"⚠️ 掩码 {mask_index} 为空，跳过")
        return False
    
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    if y1 <= y0 or x1 <= x0:
        if mask_index is not None:
            print(f"⚠️ 掩码 {mask_index} 尺寸无效 ({x1-x0}x{y1-y0})，跳过")
        return False
    
    roi_gray = image_gray[y0:y1, x0:x1]
    if roi_gray.size == 0:
        if mask_index is not None:
            print(f"⚠️ 掩码 {mask_index} ROI为空，跳过")
        return False
    
    if mask_index is not None:
        print(f"\n=== 处理掩码 {mask_index} ===")
        print(f"位置: [y:{y0}-{y1}, x:{x0}-{x1}], 尺寸: {x1-x0}x{y1-y0}")

    try:
        masked_roi = np.zeros_like(roi_gray)
        mask_region = mask[y0:y1, x0:x1]
        masked_roi[mask_region] = roi_gray[mask_region]
        
        # 转换为三通道灰度图像
        masked_roi_rgb = np.stack([masked_roi]*3, axis=-1)
        roi_pil = Image.fromarray(masked_roi_rgb)
        
        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features_dict = dinov2_model.forward_features(roi_tensor)
            patch_tokens = features_dict['x_norm_patchtokens']
            roi_emb = patch_tokens.mean(dim=1)
    except Exception as e:
        if mask_index is not None:
            print(f"❌ 处理ROI失败 (掩码 {mask_index}): {e}")
        return False
    
    try:
        similarities = []
        for path, emb in replacement_embeddings:
            emb = emb.squeeze()
            roi_emb_flat = roi_emb.squeeze()
            
            emb = emb.unsqueeze(0) if emb.dim() == 1 else emb
            roi_emb_flat = roi_emb_flat.unsqueeze(0) if roi_emb_flat.dim() == 1 else roi_emb_flat
            
            sim = torch.cosine_similarity(roi_emb_flat.cpu(), emb, dim=1).item()
            similarities.append((path, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if mask_index is not None:
            print(f"Top 3匹配图像:")
            for i, (path, sim) in enumerate(similarities[:3]):
                print(f"{i+1}. {os.path.basename(path)} (相似度: {sim:.3f})")
                
        best_path = similarities[0][0]
        if mask_index is not None:
            print(f"✅ 选择替换对象: {os.path.basename(best_path)}")
    except Exception as e:
        if mask_index is not None:
            print(f"❌ 匹配图像失败 (掩码 {mask_index}): {e}")
        return False
    
    rep_img = imread_unicode(best_path)
    if rep_img is None or rep_img.size == 0 or rep_img.shape[0] == 0 or rep_img.shape[1] == 0:
        if mask_index is not None:
            print(f"❌ 无法读取替换图像: {best_path}")
        return False
    
    # 转换为灰度图
    if len(rep_img.shape) == 3 and rep_img.shape[2] == 3:
        rep_gray = cv2.cvtColor(rep_img, cv2.COLOR_BGR2GRAY)
    else:
        rep_gray = rep_img
    
    dst_h, dst_w = y1 - y0, x1 - x0
    if mask_index is not None:
        print(f"调整图像尺寸: {dst_w}x{dst_h}")
    
    rep = resize_with_padding(rep_gray, (dst_h, dst_w))
    if rep is None:
        if mask_index is not None:
            print("❌ 调整图像大小失败")
        return False
    
    # 亮度调整 - 匹配原图区域
    orig_region = roi_gray[mask_region]
    rep_region = rep[mask_region]
    rep_adjusted = adjust_brightness(rep, orig_region)
    
    # 应用掩码
    mask_region = mask[y0:y1, x0:x1].astype(np.uint8)
    smoothed_mask = smooth_mask(mask_region)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(smoothed_mask.astype(np.uint8), kernel, iterations=1)
    border = dilated_mask - smoothed_mask
    
    canvas_roi = canvas_gray[y0:y1, x0:x1]
    
    # 主掩码区域替换
    canvas_roi[smoothed_mask] = rep_adjusted[smoothed_mask]
    
    # 边界区域混合处理
    if np.any(border > 0):
        alpha = 0.6
        canvas_roi[border > 0] = (
            alpha * rep_adjusted[border > 0] + 
            (1 - alpha) * canvas_roi[border > 0]
        ).astype(np.uint8)
    
    if mask_index is not None:
        print(f"✔ 成功替换掩码 {mask_index} (使用 {os.path.basename(best_path)})\n")
    return True

# ========== 密度评分函数 ==========
def calculate_density_score(node, max_child_ratio=0.8):
    if not node.children:
        return 0.0
    
    root_area = node.area
    total_child_area = 0
    
    queue = [node]
    while queue:
        current = queue.pop(0)
        for child in current.children:
            if child.area <= root_area * max_child_ratio:
                total_child_area += child.area
            # queue.append(child)
    
    density_score = total_child_area / root_area if root_area > 0 else 0.0
    return density_score

# ========== 视觉结构评分 ==========
def calculate_visual_structure_score(node, image_gray):
    """
    计算节点的视觉结构评分，考虑纹理复杂度、边缘密度和亮度变化
    """
    if node.area < 100:
        return 0.0
    
    # 提取掩码区域
    region = image_gray[node.mask]
    if region.size == 0:
        return 0.0
    
    # 计算纹理复杂度 (标准差)
    texture_score = np.std(region) / 50.0  # 标准化
    
    # 计算边缘密度
    edges = cv2.Canny(image_gray, 50, 150)
    edge_density = np.sum(edges[node.mask]) / (node.area * 255)  # 边缘像素占比
    
    # 计算亮度变化 (直方图范围)
    hist = cv2.calcHist([region], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)
    low = np.argmax(cdf > 0.05)
    high = np.argmax(cdf >= 0.95)
    brightness_range = high - low

    
    # 综合评分
    structure_score = 0.4 * texture_score + 0.4 * edge_density + 0.2 * (brightness_range / 100.0)
    return min(structure_score, 1.0)

# ========== 递归替换策略 ==========
def recursive_replace_strategy(node, canvas_gray, image_gray, replacement_embeddings, 
                              transform, dinov2_model, device, mask_features,
                              sim_threshold=0.8, density_threshold=0.5):
    # 计算相似度
    child_indices = [child.idx for child in node.children]
    similarity_score = 0.0
    
    if child_indices:
        similarity_score = calculate_cluster_similarity(mask_features, [node.idx] + child_indices)
        print(f"节点 {node.idx}: 相似度评分 = {similarity_score:.2f}")
    
    # 计算密度分数
    density_score = calculate_density_score(node)
    print(f"节点 {node.idx}: 密度评分 = {density_score:.2f}")
    
    # 计算视觉结构评分
    structure_score = calculate_visual_structure_score(node, image_gray)
    print(f"节点 {node.idx}: 结构评分 = {structure_score:.2f}")
    
    # 决策逻辑 - 三重阈值决策
    if similarity_score > sim_threshold and density_score > density_threshold and structure_score > 0.45:
        print(f"节点 {node.idx}: 高相似度、高密度、高结构 -> 整体替换")
        replace_one_mask(
            canvas_gray, image_gray, node.mask, 
            replacement_embeddings, transform, dinov2_model, device,
            mask_index=node.idx
        )
    else:
        if node.children:
            print(f"节点 {node.idx}: 未达到阈值 -> 单独替换子节点")
            for child in node.children:
                recursive_replace_strategy(child, canvas_gray, image_gray, replacement_embeddings,
                                          transform, dinov2_model, device, mask_features,
                                          sim_threshold, density_threshold)
        else:
            print(f"节点 {node.idx}: 叶子节点 -> 直接替换")
            replace_one_mask(
                canvas_gray, image_gray, node.mask, 
                replacement_embeddings, transform, dinov2_model, device,
                mask_index=node.idx
            )

# ========== 亮度一致性处理 ==========
def adjust_brightness_hierarchy(root_nodes, image_gray):
    """
    调整整个树结构的亮度，保持视觉一致性
    """
    print("调整亮度层次结构...")
    
    # 计算根节点亮度
    for root in root_nodes:
        root.brightness = np.mean(image_gray[root.mask])
    
    # BFS遍历树结构
    for root in root_nodes:
        queue = [root]
        while queue:
            current = queue.pop(0)
            if current.parent:
                # 根据父节点亮度调整当前节点
                current.brightness = current.parent.brightness * 0.9 + np.mean(image_gray[current.mask]) * 0.1
            
            for child in current.children:
                queue.append(child)
    
    return root_nodes

# ========== 解决重叠区域归属 ==========
def resolve_overlapping_ownership(root_nodes, mask_features):
    print("解决重叠区域归属问题...")
    
    all_masks = []
    for root in root_nodes:
        queue = [root]
        while queue:
            node = queue.pop(0)
            all_masks.append((node.idx, node.mask))
            queue.extend(node.children)
    
    resolved_nodes = []
    while all_masks:
        idx_i, mask_i = all_masks.pop(0)
        overlapping_nodes = []
        
        for j, (idx_j, mask_j) in enumerate(all_masks):
            overlap = np.logical_and(mask_i, mask_j)
            if np.sum(overlap) > 0:
                overlapping_nodes.append((idx_j, mask_j, j))
        
        if not overlapping_nodes:
            resolved_nodes.append((idx_i, mask_i))
            continue
        
        best_similarity = -1
        best_idx = idx_i
        best_mask = mask_i
        
        for idx_j, mask_j, j in overlapping_nodes:
            if (mask_features[idx_i] is not None and 
                mask_features[idx_j] is not None and
                len(mask_features[idx_i]) > 0 and 
                len(mask_features[idx_j]) > 0):
                
                feat_i = torch.tensor(mask_features[idx_i]).squeeze()
                feat_j = torch.tensor(mask_features[idx_j]).squeeze()
                
                if feat_i.dim() == 0 or feat_j.dim() == 0:
                    continue
                    
                feat_i = feat_i.unsqueeze(0) if feat_i.dim() == 1 else feat_i
                feat_j = feat_j.unsqueeze(0) if feat_j.dim() == 1 else feat_j
                
                similarity = torch.cosine_similarity(feat_i, feat_j, dim=1).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = idx_i
                    best_mask = mask_i
        
        for _, mask_j, j in overlapping_nodes:
            mask_j_clean = np.logical_and(mask_j, np.logical_not(mask_i))
            all_masks[j] = (all_masks[j][0], mask_j_clean)
        
        resolved_nodes.append((best_idx, best_mask))
    
    print(f"重叠区域处理完成: 原始掩码数={len(all_masks) + len(resolved_nodes)}, 处理后掩码数={len(resolved_nodes)}")
    return resolved_nodes

# ========== 主入口 ==========
if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    image_path = "example1/example13.jpg"
    replacement_folder = "replacement_dataset1"
    output_dir = "final_replaced_output2"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载SAM模型...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    
    # 优化掩码生成参数
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        min_mask_region_area=64,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    
    # 加载DINOv2模型
    dinov2_model, dinov2_transform = load_dinov2_model(device, "dinov2_vitb14")
    
    print(f"读取图像: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        image_bgr = imread_unicode(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"图像未找到或无法读取: {image_path}")
    
    # 转换为灰度图
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 如果图像太大，适当缩小
    max_dim = 4096
    h, w = image_gray.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_gray = cv2.resize(image_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"图像已缩小至: {new_w}x{new_h}")
    
    print(f"图像尺寸: {image_gray.shape}")
    
    # 创建灰度画布
    canvas_gray = np.ones_like(image_gray) * 255
    
    print("生成掩码中...")
    # 使用RGB图像生成掩码（更好的语义分割）
    image_rgb = cv2.cvtColor(cv2.resize(image_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    if not masks:
        print("未生成掩码")
        exit(0)
    print(f"生成掩码数量: {len(masks)}")
    
    # 计算掩码特征
    mask_features = calculate_mask_features(image_gray, masks, dinov2_transform, dinov2_model, device)
    
    # 构建掩码树结构
    root_nodes = build_mask_tree(masks)
    
    # 调整亮度层次
    root_nodes = adjust_brightness_hierarchy(root_nodes, image_gray)
    
    # 解决重叠区域归属问题
    resolved_masks = resolve_overlapping_ownership(root_nodes, mask_features)
    
    # 获取替换图像列表
    print(f"扫描替换图像文件夹: {replacement_folder}")
    replacement_paths = []
    for root, _, files in os.walk(replacement_folder):
        for f in files:
            full_path = os.path.join(root, f)
            if is_valid_image(full_path):
                replacement_paths.append(full_path)
    
    print(f"找到 {len(replacement_paths)} 张候选替换图像")
    
    # 预计算替换图像特征
    if replacement_paths:
        replacement_embeddings = get_replacement_image_embeddings(
            replacement_paths, dinov2_transform, dinov2_model, device)
    else:
        print("⚠️ 未找到任何有效替换图像")
        replacement_embeddings = []
    
    # 设置阈值
    sim_threshold = 2
    density_threshold = 2
    
    # 对每个根节点应用递归替换策略
    for root in root_nodes:
        print(f"\n处理根节点: {root.idx}")
        recursive_replace_strategy(root, canvas_gray, image_gray, replacement_embeddings,
                                 dinov2_transform, dinov2_model, device, mask_features,
                                 sim_threshold, density_threshold)
    
    # 保存结果
    output_path = os.path.join(output_dir, "merged_replacement13.png")
    cv2.imwrite(output_path, canvas_gray)
    print(f"已处理完成，保存结果到: {output_path}")
    
    # 保存原图灰度版作为参考
    cv2.imwrite(os.path.join(output_dir, "original_gray.png"), image_gray)
    
    # 输出所有掩码和可视化图像
    mask_vis_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_vis_dir, exist_ok=True)
    
    for i, mask_dict in enumerate(masks):
        mask_img = (mask_dict["segmentation"].astype(np.uint8) * 255)
        mask_path = os.path.join(mask_vis_dir, f"mask_{i:03d}.png")
        cv2.imwrite(mask_path, mask_img)
        
        mask_overlay = np.zeros_like(image_gray)
        mask_overlay[mask_dict["segmentation"]] = 255
        vis = cv2.addWeighted(image_gray, 0.7, mask_overlay, 0.3, 0)
        overlay_path = os.path.join(mask_vis_dir, f"overlay_{i:03d}.png")
        cv2.imwrite(overlay_path, vis)
    
    # 保存树结构信息
    tree_info_path = os.path.join(output_dir, "mask_tree.txt")
    with open(tree_info_path, "w") as f:
        f.write("掩码树结构:\n")
        for i, root in enumerate(root_nodes):
            f.write(f"\n树 #{i} (根节点: {root.idx})\n")
            queue = [(root, 0)]
            while queue:
                node, depth = queue.pop(0)
                f.write("  " * depth + f"|- 节点 {node.idx} (面积: {np.sum(node.mask)})\n")
                for child in node.children:
                    queue.append((child, depth + 1))
    
    print("处理完成！")