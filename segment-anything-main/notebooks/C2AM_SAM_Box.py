import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import os
import argparse
from segment_anything import sam_model_registry, SamPredictor
import random
import torch

def main(dir_a, dir_b, save_path):
    # 设置随机种子以保证可复现性
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)

    os.makedirs(save_path, exist_ok=True)
    # 加载 SAM 模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 遍历目录中的文件
    for filename in os.listdir(dir_a):
        if filename.endswith('.png'):
            # 构建完整路径
            a = os.path.join(dir_a, filename)
            b = os.path.join(dir_b, filename)
            # b = os.path.join(dir_b, filename.replace('.png', '.jpg'))

            # 读取前景二值图（路径a）和原图（路径b）
            foreground_mask = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
            
            original_image = cv2.imread(b)
            if original_image is None:
                print(f"无法读取图像: {b}")
                continue
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            original_image_alpha = cv2.imread(b, cv2.IMREAD_UNCHANGED)  # 读取图像时保留 alpha 通道
            # 检查图像是否包含 alpha 通道
            if original_image_alpha.shape[2] == 4:
                alpha_channel = original_image_alpha[:, :, 3]
                non_transparent_mask = (alpha_channel > 0).astype(np.uint8) * 255
            else:
                non_transparent_mask = np.ones_like(foreground_mask, dtype=np.uint8) * 255

            # 提取前景区域的轮廓
            contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 设置面积阈值
            area_threshold = 200  # 根据需要调整这个阈值
            # 筛选出面积大于阈值的轮廓
            contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]

            # 找到每个轮廓的边界框
            boxes = []
            img_height, img_width = foreground_mask.shape[:2]
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h != 0 else float('inf')
                if 0.5 <= aspect_ratio <= 5:
                    boxes.append((x, y, x+w, y+h))
            if len(boxes) == 0:
                print(f"没有框: {filename}")
                continue
            # 将边界框转换为SAM模型需要的格式
            input_boxes = torch.tensor(boxes, device="cuda")
            # 应用变换（归一化）到边界框
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, original_image.shape[:2])

            # 找到每个轮廓的中心点
            centers = []
            # 找到背景区域
            background_mask = cv2.bitwise_not(foreground_mask)
            background_mask = cv2.bitwise_or(background_mask, non_transparent_mask)
            # 提取背景中的坐标
            background_points = np.column_stack(np.where(background_mask > 0))
            if len(background_points) >= transformed_boxes.shape[0]:
                background_points = background_points[np.random.choice(len(background_points), transformed_boxes.shape[0], replace=False)]
                # 设置背景点标签为0
                background_labels = np.zeros(background_points.shape[0], dtype=int)
                # 合并前景点和背景点
                if not centers:
                    all_points = background_points
                    all_labels = background_labels
                else:
                    all_points = np.vstack((np.array(centers), background_points))
                    all_labels = np.hstack((np.ones(len(centers), dtype=int), background_labels))
                all_points = torch.tensor(all_points, device="cuda").unsqueeze(0)
                all_labels = torch.tensor(all_labels, device="cuda").unsqueeze(0)
                all_points = predictor.transform.apply_coords_torch(all_points, original_image.shape[:2])
            else:
                all_points = None
                all_labels = None

            # 设置当前要处理的图像
            predictor.set_image(original_image)
            # 使用批量预测2
            masks, _, _ = predictor.predict_torch(
                point_coords=all_points,
                point_labels=all_labels,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.cpu().numpy()

            # 合并掩码
            combined_mask = np.any(masks, axis=0)
            segmentation_result = (combined_mask.squeeze(0) * 255).astype(np.uint8)

            # 构建保存文件名
            save_filename = os.path.join(save_path, filename)
            # 保存分割结果
            cv2.imwrite(save_filename, segmentation_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images with SAM.")
    parser.add_argument("--dir_a", type=str, required=True, help="Directory containing foreground masks")
    parser.add_argument("--dir_b", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the segmentation results")

    args = parser.parse_args()
    main(args.dir_a, args.dir_b, args.save_path)