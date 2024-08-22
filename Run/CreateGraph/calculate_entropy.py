import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})
def calculate_entropy(coords):
    """
    计算二维坐标的熵
    :param coords: N x 2 的二维坐标数组
    :return: 二维熵值
    """
    hist_2d, x_edges, y_edges = np.histogram2d(coords[:, 0], coords[:, 1], bins=4)
    hist_2d = hist_2d / np.sum(hist_2d)  # 归一化
    entropy = -np.sum(hist_2d * np.log(hist_2d + 1e-10))  # 计算熵
    return entropy

def find_entropy_differences(wsi_path, data_path, patient_label):
    patient_label = patient_label.astype(str)
    high_risk_entropies = []
    low_risk_entropies = []

    for patient_id in tqdm(os.listdir(data_path)):
        patches = os.listdir(os.path.join(data_path, patient_id))
        boxes = [(float(patch.split('_')[0]), float(patch.split('_')[1])) for patch in patches]

        coords = np.array(boxes)
        entropy = calculate_entropy(coords)

        label = patient_label[patient_label['标本号'] == patient_id]['risk_group'].values[0]
        if label == 'high':
            high_risk_entropies.append(entropy)
        else:
            low_risk_entropies.append(entropy)

    return high_risk_entropies, low_risk_entropies


def compare_entropy_mannwhitney(high_risk_entropies, low_risk_entropies):
    """
    Use Mann-Whitney U test to compare the entropy values of high and low-risk groups
    """
    stat, p_value = stats.mannwhitneyu(high_risk_entropies, low_risk_entropies, alternative='two-sided')
    print(f"Mann-Whitney U Test Result: U-statistic = {stat:.4f}, p-value = {p_value:.4f}")

    if p_value < 0.05:
        print("There is a significant difference in entropy values between high and low-risk groups.")
    else:
        print("There is no significant difference in entropy values between high and low-risk groups.")

    # Combine data into a list for plotting
    data = [high_risk_entropies, low_risk_entropies]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the violin plot
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False,
                          positions=np.arange(1, len(data) + 1) + 0.45)

    # Customize the colors of the violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#DDA0DD')  # Lavender color
        pc.set_edgecolor('#ADD8E6')  # Light blue color
        pc.set_alpha(0.5)  # Set transparency

    # Overlay the boxplot on top of the violin plot
    ax.boxplot(data, positions=np.arange(1, len(data) + 1), widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue'),
               whiskerprops=dict(color='darkblue'),
               capprops=dict(color='darkblue'),
               medianprops=dict(color='darkblue'),
               flierprops=dict(markerfacecolor='darkblue', markeredgecolor='darkblue', markersize=6))

    # Customize the plot
    ax.set_title('Violin Plot with Overlayed Boxplot')
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(['High', 'Low'])  # Customize labels
    ax.set_xlabel('Group')
    ax.set_ylabel('Entropy Value')

    plt.show()




# def compare_entropy_mannwhitney(high_risk_entropies, low_risk_entropies):
#     """
#     Use Mann-Whitney U test to compare the entropy values of high and low-risk groups
#     """
#     stat, p_value = stats.mannwhitneyu(high_risk_entropies, low_risk_entropies, alternative='two-sided')
#     print(f"Mann-Whitney U Test Result: U-statistic = {stat:.4f}, p-value = {p_value:.4f}")
#
#     if p_value < 0.05:
#         print("There is a significant difference in entropy values between high and low-risk groups.")
#     else:
#         print("There is no significant difference in entropy values between high and low-risk groups.")
#
#     # Visualization
#     plt.figure(figsize=(12, 6))
#
#     # Create a combined DataFrame for easier plotting
#     combined_data = pd.DataFrame({
#         'Entropy': high_risk_entropies + low_risk_entropies,
#         'Group': ['High Risk'] * len(high_risk_entropies) + ['Low Risk'] * len(low_risk_entropies)
#     })
#
#     # Prepare data for violin and box plots
#     high_risk_data = combined_data[combined_data['Group'] == 'High Risk']['Entropy'].values
#     low_risk_data = combined_data[combined_data['Group'] == 'Low Risk']['Entropy'].values
#
#     # Create subplots
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # Draw the violin plot
#     positions = np.arange(1, 3)  # Positions for the violin plots
#     ax.violinplot([high_risk_data, low_risk_data], showmeans=False, showmedians=False, positions=positions - 0.3, widths=0.4)
#     #     sns.violinplot(x='Group', y='Entropy', data=combined_data, inner=None, color='lightgray', alpha=0.5)
#
#     # Draw the boxplot on top of the violin plot
#     # ax.boxplot([high_risk_data, low_risk_data], positions=positions + 0.1, widths=0.2, patch_artist=True,
#     #            boxprops=dict(facecolor='lightblue', color='darkblue'),
#     #            whiskerprops=dict(color='darkblue'),
#     #            capprops=dict(color='darkblue'),
#     #            showfliers=True, flierprops=dict(marker='o', markerfacecolor='red', markersize=5)
#     #            )
#
#     # Set Y-axis limits to ensure the violin plot shows full range
#     # ax.set_ylim(min(np.min(high_risk_data), np.min(low_risk_data)) - 0.1, max(np.max(high_risk_data), np.max(low_risk_data)) + 0.1)
#
#     # Customize the plot
#     ax.set_xticks(positions)
#     ax.set_xticklabels(['High Risk', 'Low Risk'])  # Customize labels as needed
#     ax.set_ylabel('Entropy Value')
#     ax.grid(axis='y')
#
#     plt.tight_layout()
#     plt.savefig('entropy_comparison_side_by_side.png')  # Save the image
#     plt.show()  # Show the image
#
#     # 设置总体图形大小
#     plt.figure(figsize=(7, 7))
#     plt.rcParams.update({'font.size': 20})
#
#     # KDE图的子图
#     ax_kde = plt.subplot(1, 1, 1)
#
#     # 分别获取高风险组和低风险组的数据
#     high_risk_data = combined_data[combined_data['Group'] == 'High Risk']['Entropy'].values
#     low_risk_data = combined_data[combined_data['Group'] == 'Low Risk']['Entropy'].values
#
#     # 对高风险组和低风险组分别计算KDE并归一化
#     for risk_data, color, label in zip([high_risk_data, low_risk_data], ["#DDA0DD", "#ADD8E6"], ['High', 'Low']):
#         # 计算KDE
#         density = stats.gaussian_kde(risk_data)
#
#         # 获取密度值
#         x = np.linspace(0, 1, 1000)  # 因为数据已经归一化，x轴范围设为0-1
#         density_values = density(x)
#
#         # 归一化
#         normalized_density_values = density_values / np.max(density_values)
#
#         # 绘制KDE曲线
#         ax_kde.plot(x, normalized_density_values, color=color, alpha=0.5, label=f'{label} Risk')
#         ax_kde.fill_between(x, normalized_density_values, color=color, alpha=0.5)
#
#
#     ax_kde.set_xlim(0, 1)  # 设置x轴的范围为0到1
#     ax_kde.set_xticks([0, 0.5, 1])
#         # 设置刻度格式
#     ax_kde.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#     ax_kde.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#     ax_kde.set_ylim(0, 1)  # 设置y轴的范围为0到1
#
#     plt.tight_layout()
#     # plt.savefig("kdeplot_importance_value.png", dpi=600)
#     plt.show()
if __name__ == '__main__':
    wsi_path = '/data0/pathology/all_patients'
    data_path = '/data0/yuanyz/NewGraph/datasets/patientminimum_spanning_tree256412/test'
    save_path = '/data0/yuanyz/NewGraph/tools/result/img_new'
    high_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/high_risk_group.csv')
    low_patient = pd.read_csv('/data0/yuanyz/NewGraph/tools/data/low_risk_group.csv')
    patient_label = pd.concat([high_patient, low_patient], axis=0)
    high_risk_entropies, low_risk_entropies = find_entropy_differences(wsi_path, data_path, patient_label)
    compare_entropy_mannwhitney(high_risk_entropies, low_risk_entropies)
