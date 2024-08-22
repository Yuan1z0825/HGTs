import os
import warnings

import cv2
import dgl
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from dgl.data.utils import save_graphs
from fuzzywuzzy import process
from histocartography.visualization import OverlayGraphVisualization
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from options import parse_args

warnings.filterwarnings('ignore')
np.random.seed(123)

CellTypes = ['Tumor cells',
             'Vascular endothelial cells',
             'Lymphocytes',
             'Fibroblasts',
             'Biliary epithelial cells',
             'Hepatocytes',
             'Others']


def img_energy(GrayImage):
    tmp = np.abs(cv2.Sobel(GrayImage, cv2.CV_32F, 1, 1))
    energy = tmp.sum() / (2048 * 2048)
    return energy

from collections import defaultdict

def replace_with_best_match_vectorized(cell_names):
    unique_names = cell_names.unique()
    best_matches = {}

    for name in unique_names:
        best_match, score = process.extractOne(name, CellTypes)
        if score > 0:  # Adjust this threshold based on your needs
            best_matches[name] = best_match
        else:
            best_matches[name] = name

    return cell_names.map(best_matches)

def load_txt(LabelDataPath, Patient):
    inTXTDataPath = os.path.join(LabelDataPath, Patient, Patient + '.txt')
    inTXTData = pd.read_csv(inTXTDataPath, sep='\t', engine='python', encoding='utf-8')

    inTXTData['Class'] = replace_with_best_match_vectorized(inTXTData['Class'])
    print('Finished load_txt')
    return inTXTData


def Distance(x, y):
    return pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2)


def polymerization(inTXTData, tr):
    final_data = pd.DataFrame(columns=inTXTData.columns)
    for type in CellTypes:
        data = inTXTData.loc[inTXTData['Class'] == type]
        if data.shape[0] == 0:
            continue
        data = data.reset_index(drop=True)
        # print("之前的数量",len(data))
        i = 0
        length = data.shape[0]
        while i < length:
            j = i + 1
            while j < length:
                # print(i,j,data.index)
                x = pd.concat([data.iloc[i, 5: 13], data.iloc[i, 25: 31]], axis=0)
                y = pd.concat([data.iloc[j, 5: 13], data.iloc[j, 25: 31]], axis=0)
                x = np.array(x)
                y = np.array(y)
                dis = Distance(x, y)
                if dis <= tr:
                    # print(cos_sim)
                    new_feature = (x + y) / 2
                    data.drop(axis=0, index=j, inplace=True)
                    length -= 1
                    data.iloc[i][5:13] = new_feature[:8]
                    data.iloc[i][25:31] = new_feature[8:]
                    data = data.reset_index(drop=True)
                    j -= 1
                j += 1
            i += 1
        # print("之后的数量",len(data))
        final_data = pd.concat([final_data, data], axis=0)
    return final_data


def get_range_for_every_feature(LabelDataPath):
    """
    确定所有患者txt表格中每个指标的上下限, 用于后面归一化操作
    :param LabelDataPath: 保存标记工程的总文件
    :return: (2 * 41)数据, 第一维为min 第二维为max, 每一列为一个指标
    """
    Patients = os.listdir(LabelDataPath)
    MinAndMax = np.empty(shape=(2, 41))
    MinAndMax[0, :] = 10000
    MinAndMax[1, :] = -10000
    for Patient in Patients:
        TXTData = load_txt(LabelDataPath, Patient)
        Features = TXTData.iloc[:, 7:]
        for i in range(41):
            if Features.min()[i] < MinAndMax[0, i]:
                MinAndMax[0, i] = Features.min()[i]
            if Features.max()[i] > MinAndMax[1, i]:
                MinAndMax[1, i] = Features.max()[i]

    print('Finished get_range_for_every_feature')
    return MinAndMax


def load_image(LabelDataPath, Patient):
    """
    加载当前patient对应的细胞标注结果图和细胞核结果图
    :param LabelDataPath: 保存所有标记结果工程的总文件夹
    :param Patient: 当前患者编号
    :return: Numpy数组格式细胞标注图和细胞核标注图
    """
    Image.MAX_IMAGE_PIXELS = None
    LabeledCellImagePath = os.path.join(LabelDataPath, Patient, Patient + '-CellLabels.png')
    LabeledCellImage = Image.open(LabeledCellImagePath)
    LabeledCellImage = np.array(LabeledCellImage)
    LabeledNucleiImagePath = os.path.join(LabelDataPath, Patient, Patient + '-NucleiLabels.png')
    LabeledNucleiImage = Image.open(LabeledNucleiImagePath)
    LabeledNucleiImage = np.array(LabeledNucleiImage)

    print('Finished load_image')
    return LabeledCellImage, LabeledNucleiImage,


def load_wsi(WSIDataPath, Patient):
    """
    加载当前patient对应WSI数据
    :param WSIDataPath: 保存所有患者WSI数据的总文件夹
    :param Patient: 当前患者编号
    :return: openslide格式的slide数据，可使用np.array(slide.read_region函数读取图片转化为numpy数组)
    """
    path = os.path.join(WSIDataPath, Patient, Patient + '.ndpi')
    slide = openslide.OpenSlide(path)

    print('Finished load_wsi')
    return slide


def find_patch_boxes_new(LabeledCellImage, patch_size=256, ratio=0.1):
    """
    由细胞标记图像提取细胞种类最多的Patch，
    :param LabeledCellImage: 细胞标记图像
    :param patch_size: patch边长
    :param ratio: patch个数/总面积
    :return:
    """
    # 分割出前景
    CellImage = ((LabeledCellImage > 0) * 255).astype('uint8')
    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(CellImage), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((50, 50), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    # 计算总面积
    image_binary = np.array(image_open > 0, dtype='int')
    S = image_binary.sum()

    # 获得前景的Bounding Box
    contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = np.array([cv2.boundingRect(c) for c in contours])

    xmin = np.min(boundingBoxes[:, 0])
    xmax = np.max(boundingBoxes[:, 0] + boundingBoxes[:, 2])
    ymin = np.min(boundingBoxes[:, 1])
    ymax = np.max(boundingBoxes[:, 1] + boundingBoxes[:, 3])
    boundingBox = [xmin, ymin, xmax, ymax]

    boundingBox = [ymin, xmin, ymax, xmax]

    # 滑窗获得每个patch的BoundingBox坐标
    box_list = []
    for i in np.arange(xmin, xmax - patch_size, patch_size):
        for j in np.arange(ymin, ymax - patch_size, patch_size):
            # 注意到cv2 x和y坐标颠倒问题
            box = (j, i, j + patch_size, i + patch_size)
            box_list.append(box)

    # 统计每个patch内细胞种类数
    CellTypeNum = []
    for box in box_list:
        # 这里的x和y就是图像对应的竖直方向和水平方向
        patch_xmin, patch_ymin, patch_xmax, patch_ymax = box
        patch = LabeledCellImage[patch_xmin: patch_xmax, patch_ymin: patch_ymax]
        CellTypeNum.append(len(np.unique(patch)) - 1)
        #CellTypeNum.append(np.sum(patch > 0))
    coords_list = pd.DataFrame(box_list)
    CellTypeNum = pd.DataFrame(CellTypeNum)
    coordinates = pd.concat([coords_list, CellTypeNum], axis=1)
    coordinates.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'num']
    coordinates = coordinates.sort_values(by='num', ascending=False)

    coordinates = coordinates.iloc[: 5000, :4]

    #全部选择
    SelectedCords = coordinates.iloc[:32, :4]
    if SelectedCords.shape[0] < 32:
        print('Warning: The number of patches is less than 32')
        raise ValueError('The number of patches is less than 32')

    # coordinates = np.array(coordinates)
    # energy = []
    # for coordinate in tqdm(coordinates, desc='Windows loop'):
    #     try:
    #         img = np.array(WSIData.read_region((int(coordinate[1]) * 4, int(coordinate[0]) * 4), 2, (512, 512)))[:, :,
    #               :3]
    #         GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         _, b = cv2.threshold(GrayImage, 170, 255, cv2.THRESH_BINARY)
    #         if b.mean() > 150 or b.mean() < 20:
    #             eg = 0.0
    #             energy.append(eg)
    #             continue
    #         eg = img_energy(GrayImage)
    #
    #     except openslide.lowlevel.OpenSlideError as e:
    #         eg = 0.0
    #     energy.append(eg)
    #
    # coords_list = pd.DataFrame(coordinates)
    # energy_list = pd.DataFrame(energy)
    # coordinates = pd.concat([coords_list, energy_list], axis=1)
    # coordinates.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'energy']
    # coordinates = coordinates.sort_values(by='energy', ascending=False)
    # SelectedCords = coordinates.iloc[: 128, :4]

    SelectedCords = np.array(SelectedCords)
    imgs = []
    # for coordinate in tqdm(SelectedCords, desc='Image'):
    #     # img = np.array(WSIData.read_region((int(coordinate[1] * 4), int(coordinate[0] * 4)), 2, (patch_size, patch_size)))[:, :, :3]
    #     img = np.array(WSIData.read_region((int(coordinate[1] * 4), int(coordinate[0] * 4)), 0, (patch_size * 4, patch_size * 4)))[:, :, :3]
    #     imgs.append(img)
    #     # plt.imshow(img)
    #     # plt.show()
    return SelectedCords, imgs



def get_origin_image(WSIData, level_dimensions):
    """
    从slide数据读出降采样level为level_dimensions的RGB图片，输出为numpy数组格式
    :param WSIData: slide数据
    :param level_dimensions: 降采样层级，取值[0-8]
    :return: numpy数组RGB图片
    """
    (m, n) = WSIData.level_dimensions[level_dimensions]
    OriginImage = np.array(WSIData.read_region((0, 0), level_dimensions, (m, n)))[:, :, :3]

    print('Finished get_origin_image')
    return OriginImage


def get_cell_coordinate_pixel(CellData, box):
    """
    将CellData表格中坐标信息转化为当前patch中对应的像素坐标
    :param CellData: 当前patch中的细胞表格
    :param box: 当前patch对应的box, 单位为微米
    :return: 列表格式各细胞对应像素坐标
    """
    CoordinateUm = [list(CellData['Centroid X µm']), list(CellData['Centroid Y µm'])]
    # CoordinatePixelX = ((CoordinateUm[0] - box[1]) / (Ratio * 4)).astype('int')
    # CoordinatePixelY = ((CoordinateUm[1] - box[0]) / (Ratio * 4)).astype('int')
    CoordinatePixelX = (CoordinateUm[0])
    CoordinatePixelY = (CoordinateUm[1])
    CoordinatePixel = []
    for i in range(len(CellData)):
        CoordinatePixel.append((CoordinatePixelX[i], CoordinatePixelY[i]))
    CoordinatePixel = list(CoordinatePixel)
    return CoordinatePixel


# def generate_graph(CellData, Centroids, Features, PatchSize=1024, k=5, thresh=100):
#     """
#     以KNN方式构建图结构
#     :param Centroids: 图节点的坐标, list
#     :param Features: 所有节点的特征向量, tensor
#     :param PatchSize: 当前Patch大小, int
#     :param k: KNN中K
#     :param thresh: 超过threshold不再建立边
#     :return: dgl.graph
#     """
#     graph = dgl.DGLGraph()
#     graph.add_nodes(len(Centroids))
#
#     image_size = (PatchSize, PatchSize)
#     # 设置图节点中心坐标
#     graph.ndata['centroid'] = torch.FloatTensor(Centroids) * 4
#     # 设置图节点特征（特征还包括归一化的坐标信息）
#     centroids = graph.ndata['centroid']
#     normalized_centroids = torch.empty_like(centroids)
#     normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
#     normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]
#
#     if Features.ndim == 3:
#         normalized_centroids = normalized_centroids \
#             .unsqueeze(dim=1) \
#             .repeat(1, Features.shape[1], 1)
#         concat_dim = 2
#     elif Features.ndim == 2:
#         concat_dim = 1
#
#     concat_features = torch.cat(
#         (
#             Features,
#             normalized_centroids
#         ),
#         dim=concat_dim,
#     )
#
#     graph.ndata['feat'] = concat_features[:, 2:]
#     name_tensor = []
#     for item in CellData['Class']:
#         Flag = 1
#         for i in range(len(CellTypes)):
#             if item == CellTypes[i]:
#                 name_tensor.append([i + 1])
#                 Flag = 0
#         if Flag:
#             name_tensor.append([7])  # 过滤掉其他可能标注但是未在CellType的细胞，我们归为Other
#     name_tensor = torch.tensor(name_tensor)
#     graph.ndata['name'] = name_tensor
#     # 利用KNN方法构建图
#     if Features.shape[0] != 1:
#         k = min(Features.shape[0] - 1, k)
#         adj = kneighbors_graph(
#             centroids,
#             k,
#             mode="distance",
#             include_self=False,
#             metric="euclidean").toarray()
#
#         if thresh is not None:
#             adj[adj > thresh] = 0
#
#         edge_list = np.nonzero(adj)
#         graph.add_edges(list(edge_list[0]), list(edge_list[1]))
#
#     return graph

# import dgl
# import torch
# import numpy as np
# from scipy.spatial import Delaunay
#
# def generate_graph(CellData, Centroids, Features, PatchSize=1024, thresh=100):
#     """
#     使用Delaunay三角剖分构建图结构
#     :param Centroids: 图节点的坐标, list
#     :param Features: 所有节点的特征向量, tensor
#     :param PatchSize: 当前Patch大小, int
#     :param thresh: 超过threshold不再建立边
#     :return: dgl.graph
#     """
#     graph = dgl.DGLGraph()
#     graph.add_nodes(len(Centroids))
#
#     image_size = (PatchSize, PatchSize)
#     # 设置图节点中心坐标
#     graph.ndata['centroid'] = torch.FloatTensor(Centroids) * 4
#     # 设置图节点特征（特征还包括归一化的坐标信息）
#     centroids = graph.ndata['centroid']
#     normalized_centroids = torch.empty_like(centroids)
#     normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
#     normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]
#
#     if Features.ndim == 3:
#         normalized_centroids = normalized_centroids \
#             .unsqueeze(dim=1) \
#             .repeat(1, Features.shape[1], 1)
#         concat_dim = 2
#     elif Features.ndim == 2:
#         concat_dim = 1
#
#     concat_features = torch.cat(
#         (
#             Features,
#             normalized_centroids
#         ),
#         dim=concat_dim,
#     )
#
#     graph.ndata['feat'] = concat_features[:, 2:]
#     name_tensor = []
#     for item in CellData['Class']:
#         Flag = 1
#         for i in range(len(CellTypes)):
#             if item == CellTypes[i]:
#                 name_tensor.append([i + 1])
#                 Flag = 0
#         if Flag:
#             name_tensor.append([7])  # 过滤掉其他可能标注但是未在CellType的细胞，我们归为Other
#     name_tensor = torch.tensor(name_tensor)
#     graph.ndata['name'] = name_tensor
#
#     # 使用Delaunay三角剖分构建图
#     tri = Delaunay(centroids.numpy())
#     for simplex in tri.simplices:
#         for i in range(3):
#             for j in range(i+1, 3):
#                 dist = np.linalg.norm(centroids[simplex[i]].numpy() - centroids[simplex[j]].numpy())
#                 if thresh is None or dist <= thresh:
#                     graph.add_edge(simplex[i], simplex[j])
#                     graph.add_edge(simplex[j], simplex[i])
#
#     return graph

import dgl
import torch
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

def generate_graph(CellData, Centroids, Features, PatchSize=1024):
    """
    使用最小生成树 (MST) 构建图结构
    :param Centroids: 图节点的坐标, list
    :param Features: 所有节点的特征向量, tensor
    :param PatchSize: 当前Patch大小, int
    :return: dgl.graph
    """
    graph = dgl.DGLGraph()
    graph.add_nodes(len(Centroids))

    image_size = (PatchSize, PatchSize)
    # 设置图节点中心坐标
    graph.ndata['centroid'] = torch.FloatTensor(Centroids)
    # 设置图节点特征（特征还包括归一化的坐标信息）
    # centroids = graph.ndata['centroid']

    normalized_centroids = torch.empty_like(graph.ndata['centroid'])
    centroids = normalized_centroids
    centroids[:, 0] = Features[:, 0]
    centroids[:, 1] = Features[:, 1]
    normalized_centroids[:, 0] = Features[:, 0] / 256
    normalized_centroids[:, 1] = Features[:, 1] / 256

    if Features.ndim == 3:
        normalized_centroids = normalized_centroids \
            .unsqueeze(dim=1) \
            .repeat(1, Features.shape[1], 1)
        concat_dim = 2
    elif Features.ndim == 2:
        concat_dim = 1

    concat_features = torch.cat(
        (
            Features,
            normalized_centroids
        ),
        dim=concat_dim,
    )

    graph.ndata['feat'] = concat_features[:, 2:]
    name_tensor = []
    for item in CellData['Class']:
        Flag = 1
        for i in range(len(CellTypes)):
            if item == CellTypes[i]:
                name_tensor.append([i + 1])
                Flag = 0
        if Flag:
            name_tensor.append([7])  # 过滤掉其他可能标注但是未在CellType的细胞，我们归为Other
    name_tensor = torch.tensor(name_tensor)
    graph.ndata['name'] = name_tensor

    # 使用最小生成树 (MST) 构建图
    distances = cdist(centroids.numpy(), centroids.numpy(), metric='euclidean')
    mst = minimum_spanning_tree(distances).tocoo()
    graph.add_edges(mst.row, mst.col)

    return graph


# 增加ont-hot编码特征
def concat_one_hot(Features, CellData):
    temp = []
    # print(Features.shape)
    # print(CellData.shape)
    for item in CellData['Class']:
        tmp = []
        for i in range(len(CellTypes)):
            if item == CellTypes[i]:
                tmp.append(1)
            else:
                tmp.append(0)
        temp.append(tmp)
    temp = torch.tensor(temp)
    Features = torch.cat((Features, temp), dim=1)
    return Features


def generate_and_save_cell_graphs(box, TXTData, Label, OutPath):
    """
    对应于每个box, 对于其中的每类细胞都构建一张图, 并保存在Graph文件中
    :param box: patch坐标单位为um
    :param TXTData: 医生标记结果导出的TXT表格
    :param Patient: 当前病患, 在Graph文件夹中建立相应文件夹
    """
    # 将box内的所有细胞筛选出来
    CellINPatch = TXTData[(box[1] < TXTData['Centroid X µm']) & (TXTData['Centroid X µm'] < box[3]) &
                          (box[0] < TXTData['Centroid Y µm']) & (TXTData['Centroid Y µm'] < box[2])]

    # 在这里添加剔除一些不相关细胞的东西

    # CellINPatch=polymerization(CellINPatch,5000)
    # 对于方框内的所有细胞，按照细胞分类分别提取
    print(CellINPatch.shape)
    CellData = CellINPatch
    # 一个图中的节点个数要大于5
    if len(CellData) > 0:
        CoordinatePixel = get_cell_coordinate_pixel(CellData, box)
        # 提取txt表格中的形态学数据（剔除颜色相关数据）
        Features = pd.concat([CellData.iloc[:, 5: 13], CellData.iloc[:, 13:]], axis=1)
        Features['Centroid X µm'] = ((Features['Centroid X µm'] - box[1]) / Ratio / 4).astype('int')
        Features['Centroid Y µm'] = ((Features['Centroid Y µm'] - box[0]) / Ratio / 4).astype('int')

        # Features = Features.drop_duplicates(subset=['Centroid X µm', 'Centroid Y µm'], keep='first')
        Features = np.array(Features, dtype='float64')
        # NormalizedFeatures =
        Features = torch.from_numpy(Features)  # Features要归一化！！！！！
        # print("one-hot之前",Features.shape,CellData.shape)
        # 进行one-hot编码
        Features = concat_one_hot(Features, CellData)
        # print("one-hot之后",Features.shape,CellData.shape,len(CoordinatePixel))
        Graph = generate_graph(CellData, CoordinatePixel, Features)
        img = Image.open(OutPath + "/wsi.png")
        img = np.array(img)
        visualizer = OverlayGraphVisualization(
            node_style='fill',
            node_radius=3,
            edge_thickness=1,
            colormap='coolwarm',
            show_colormap=True,
            min_max_color_normalize=False
        )
        canvas = visualizer.process(img, Graph)
        canvas.save(
            OutPath + "/node_in_img.png",
            quality=100
        )
        print("hello")
        # 一个图中边的个数要大于5
        if Graph.num_edges() > -1:
            GraphName = 'AllCell.bin'
            GraphPath = os.path.join(OutPath, GraphName)
            save_graphs(GraphPath, Graph, Label)

    # print('Finished generate_graph')


def graph_visualize(box, WSIImage, graph):
    """
    图结构的可视化，用于检查和超参数的选择
    :param box: patch边界box，微米
    :param WSIImage: 读入的numpy数组格式的原始图像，降采样率：4
    :param graph: 要可视化的graph，dgl.graph
    """
    boxpixel = (box / Ratio / 4).astype('int')
    image = WSIImage[boxpixel[1]: boxpixel[3], boxpixel[0]: boxpixel[2]]
    visualizer = OverlayGraphVisualization(node_radius=1,
                                           edge_thickness=1,
                                           )
    canvas = visualizer.process(image, graph)
    canvas.show()


def show_big_array(array):
    """
    展示非常大数组图片（避免plt导致python无响应）
    :param array: 大数组
    """
    p = Image.fromarray(array)
    p.show()


def robust_read_csv(file_path, sep='\t'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    encoding_options = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16']
    for encoding in encoding_options:
        try:
            data = pd.read_csv(file_path, sep=sep, engine='python', encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            return data
        except Exception as e:
            print(f"Failed to read the file with encoding: {encoding}. Error: {e}")

    raise ValueError("Failed to read the file with any of the tried encodings.")


if __name__ == '__main__':
    opt = parse_args()
    print('\n')
    # 载入数据
    LabelDataPath = opt.label_data_path  # 保存所有标记结果工程的总文件夹
    WSIDataPath = opt.WSI_data_path  # 保存所有WSI数据的总文件夹
    # FollowUpData = pd.read_csv(opt.follow_up_data, sep='\t', engine='python', encoding='utf-8')  # 加载随访数据
    FollowUpData = robust_read_csv(opt.follow_up_data, sep='\t')
    FollowUpData.dropna(axis=0, how='all')
    # Patients = os.listdir(LabelDataPath)

    # 带区域标记的病人
    Patients = os.listdir(LabelDataPath)
    Patients.sort()
    # FeaturesMinAndMax = np.loadtxt('MinAndMax.csv')
    result_dataset = '/data/yuanyz/datasets/32_patches_graphs'
    Patients = os.listdir('/data0/yuanyz/NewGraph/datasets/patientminimum_spanning_tree256412/test')
    for Patient in tqdm(Patients):
        if os.path.exists(os.path.join(result_dataset, Patient)) is False:
            os.makedirs(os.path.join(result_dataset, Patient))
            # TXTData = load_txt('./txtdata/PCA3', Patient)  # 加载TXT表格
            TXTData = load_txt(LabelDataPath, Patient)  # 加载TXT表格
            LabeledCellImage, LabeledNucleiImage = load_image(LabelDataPath, Patient)  # 加载标注图片
            WSIData = load_wsi(WSIDataPath, Patient)  # 加载WSI数据
            # WSIImage = get_origin_image(WSIData, level_dimensions=2)

            Ratio = float(WSIData.properties['openslide.mpp-x'])
            # 选择包含细胞种类最多的Patches
            SelectedPatchBoxes, imgs = find_patch_boxes_new(LabeledCellImage, ratio=Ratio)
            SelectedPatchBoxes = np.array(SelectedPatchBoxes)

            # 构建图结构
            Ratio = float(WSIData.properties['openslide.mpp-x'])

            SelectedPatchBoxesUm = SelectedPatchBoxes * Ratio * 4  # 将像素框转换为实际距离
            for idx, box in enumerate(SelectedPatchBoxesUm):
                BoxName = str(box[0]) + '_' + str(box[1])
                OutPath = os.path.join(result_dataset, Patient, BoxName)
                if not os.path.exists(OutPath):
                    os.makedirs(OutPath)
                # plt.imsave(OutPath + "/wsi.png", imgs[idx])
                PatientFollowUp = FollowUpData[FollowUpData['标本号'] == int(Patient)]
                if not PatientFollowUp.empty:
                    SurvLabel = {
                        'CoxLabel': torch.tensor([(float(PatientFollowUp['无瘤/月']), float(PatientFollowUp['复发']))]),
                    'SurvLabel': torch.tensor([(float(PatientFollowUp['生存/月']), float(PatientFollowUp['死亡']))])}
                    ImgSave = True
                    generate_and_save_cell_graphs(box, TXTData, SurvLabel, OutPath)
                else:
                    PatientFollowUp = FollowUpData[FollowUpData['标本号'] == str(Patient)]
                    if not PatientFollowUp.empty:
                        SurvLabel = {
                            'CoxLabel': torch.tensor(
                                [(float(PatientFollowUp['无瘤/月']), float(PatientFollowUp['复发']))]),
                        'SurvLabel': torch.tensor([(float(PatientFollowUp['生存/月']), float(PatientFollowUp['死亡']))])}
                        generate_and_save_cell_graphs(box, TXTData, SurvLabel, OutPath)
            print('Finished generate graph for ' + Patient)

        else:
            continue

    # 可视化
    # Graph = dgl.load_graphs(os.path.join(OutPath, 'Tumor cells.bin'))[0][0]
    # graph_visualize(box, WSIImage, Graph)
