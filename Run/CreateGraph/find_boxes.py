
import os

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import openslide
from matplotlib import pyplot as plt
from tqdm import tqdm


# def find_boxes(wsi_path, data_path, save_dir, patient_label):
#     patient_label = patient_label.astype(str)
#     for patient_id in tqdm(os.listdir(data_path)):
#         wsi_data = os.path.join(wsi_path, patient_id + '.ndpi')
#         patch_size = 256 // 2
#         slide = openslide.open_slide(wsi_data)
#         print("Finished Reading WSI file: ", wsi_data)
#         img = slide.read_region((0, 0), 3, slide.level_dimensions[3]).convert('RGB')
#         print("Finished region extraction.")
#         Ratio = float(slide.properties['openslide.mpp-x'])
#         patches = os.listdir(os.path.join(data_path, patient_id))
#         boxes = [(float(patch.split('_')[0]), float(patch.split('_')[1])) for patch in patches]
#         print("Finished Reading patches.")
#         label = patient_label[patient_label['标本号'] == patient_id]['risk_group'].values[0]
#         print("label: ", label)
#         for box in tqdm(boxes):
#             x, y = box
#             x = x / Ratio / 4 / 2
#             y = y / Ratio / 4 / 2
#             x = int(x)
#             y = int(y)
#             img = np.array(img)
#             print("Finished Reading Image.")
#             # 以 xy为左上角顶点绘制正方形滑倒img上面
#             img = cv2.rectangle(img, (y, x), (y + patch_size, x + patch_size), (255, 0, 0), 5)
#             print("Finished Drawing Box.")
#
#         # 对img进行可视化
#         img = Image.fromarray(img)
#         save_path = os.path.join(save_dir, patient_id + '_' + str(len(boxes)) + '_' + label + '.png')
#         # img.show()
#         img.save(save_path)

def find_boxes(wsi_path, data_path, save_dir, patient_label):
    patient_label = patient_label.astype(str)
    patients = os.listdir(data_path)
    patients = ['201714953']
    for patient_id in tqdm(patients):
        wsi_data = os.path.join(wsi_path, patient_id + '.ndpi')
        patch_size = 256 // 2
        slide = openslide.open_slide(wsi_data)
        print("Finished Reading WSI file: ", wsi_data)
        img = slide.read_region((0, 0), 3, slide.level_dimensions[3]).convert('RGB')
        print("Finished region extraction.")
        Ratio = float(slide.properties['openslide.mpp-x'])
        patches = os.listdir(os.path.join(data_path, patient_id))
        boxes = [(float(patch.split('_')[0]), float(patch.split('_')[1])) for patch in patches]
        print("Finished Reading patches.")
        label = patient_label[patient_label['标本号'] == patient_id]['risk_group'].values[0]
        print("label: ", label)
        img = np.array(img)
        for box in tqdm(boxes):
            x, y = box
            x = x / Ratio / 4 / 2
            y = y / Ratio / 4 / 2
            x = int(x)
            y = int(y)
            print("Finished Reading Image.")

            # 计算矩形块的中心
            center_x = y + patch_size // 2
            center_y = x + patch_size // 2

            # 填充矩形块为橘红色
            img[x:x + patch_size, y:y + patch_size] = [255, 165, 0]  # 橘红色填充
            print("Finished Filling Rectangle with Orange-Red.")

            # 画一个圆，以矩形块的中心为圆心
            radius = patch_size // 2  # 半径等于块大小的一半
            cv2.circle(img, (center_x, center_y), radius, (255, 165, 0), -1)  # 填充橘红色的圆
            print("Finished Drawing Orange-Red Circle.")

            # 创建从橘红色到浅蓝色的热图效果
            start_color = np.array([255, 165, 0])  # 橘红色
            end_color = np.array([173, 216, 230])  # 浅蓝色
            max_radius = radius + 50  # 渐变效果的最大半径

            for r in range(radius, max_radius):
                alpha = (r - radius) / (max_radius - radius)  # 计算插值比例
                color = (1 - alpha) * start_color + alpha * end_color  # 插值颜色
                color = tuple(map(int, color))  # 转换为整数
                cv2.circle(img, (center_x, center_y), r, color, 2)  # 绘制圆形
            print("Finished Creating Gradient Effect.")


        # 为img增加圆

        # 保存结果图像
        img = Image.fromarray(img)
        save_path = os.path.join(save_dir, patient_id + '_' + str(len(boxes)) + '_' + label + '.png')
        img.save(save_path)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    wsi_path = '/data0/pathology/all_patients'
    data_path = '/data0/yuanyz/NewGraph/datasets/patientminimum_spanning_tree256412/test'
    save_path = '/data0/yuanyz/NewGraph/tools/result/img_new1'
    high_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/high_risk_group.csv')
    low_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/low_risk_group.csv')
    patient_label = pd.concat([high_patient, low_patient], axis=0)
    find_boxes(wsi_path, data_path, save_path, patient_label)
