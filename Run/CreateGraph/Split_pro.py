"""Split data into 3 sets: train, validation and test"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def copy_file(old_file_path, new_path):
    """
    copy old_file_path to new_path dirs
    :param old_file_path:
    :param new_path:
    """
    file_name = old_file_path.split('/')[-1]
    src = old_file_path
    dst = os.path.join(new_path, file_name)
    shutil.copyfile(src, dst)


def copy_dir(src_path, target_path):
    if os.path.isdir(src_path) and os.path.isdir(target_path):
        filelist_src = os.listdir(src_path)
        for file in filelist_src:
            path = os.path.join(os.path.abspath(src_path), file)
            if os.path.isdir(path):
                path1 = os.path.join(os.path.abspath(target_path), file)
                if not os.path.exists(path1):
                    os.mkdir(path1)
                copy_dir(path, path1)
            else:
                with open(path, 'rb') as read_stream:
                    contents = read_stream.read()
                    path1 = os.path.join(target_path, file)
                    with open(path1, 'wb') as write_stream:
                        write_stream.write(contents)
        return True

    else:
        return False


def move_file(old_file_path, new_path):
    file_name = old_file_path.split('/')[-1]
    src = old_file_path
    dst = os.path.join(new_path, file_name)
    shutil.move(src, dst)

RANDOM_SEED = 42
if __name__ == "__main__":

    # 数据集的路径自己根据目标修改
    cg_path = '/data0/yuanyz/NewGraph/datasets/test'
    censor_file = '/data0/yuanyz/NewGraph/censor_with_normalized_risk.csv'
    censor_df = pd.read_csv(censor_file, usecols=['标本号', '复发'])
    
    patient_fnames = []
    patients = os.listdir(cg_path)
    # 构建标本号到censor标记的映射
    censor_dict = {}
    for i, row in censor_df.iterrows():
        print(row['标本号'])
        if row['复发'] == 1:
            censor_dict[str(row['标本号'])] = 1 
        else:
            censor_dict[str(row['标本号'])] = 0
    # 划分 censor 和 uncensor
    censor_list = []
    uncensor_list = []
    for patient in patients:
        patient_fname = os.path.join(cg_path, patient)
        #patient_fnames.append(patient_fname)
        censor_list.append(patient_fname) if censor_dict[patient]==0 else uncensor_list.append(patient_fname)
    
    sava_path = '/data0/yuanyz/NewGraph/datasets'
    keys = ['TCGA']
    for key in keys:
        data = patient_fnames
        if len(data) != 1:
        
            # 按比例分割数据集       
            censor_train, censor_test = train_test_split(censor_list, test_size=0.36, random_state=RANDOM_SEED)
            censor_val, censor_test = train_test_split(censor_test, test_size=5/9, random_state=RANDOM_SEED)
            uncensor_train, uncensor_test = train_test_split(uncensor_list, test_size=0.36, random_state=RANDOM_SEED)
            uncensor_val, uncensor_test = train_test_split(uncensor_test, test_size=5/9, random_state=RANDOM_SEED)
            #x_train, x_test = train_test_split(data, test_size=0.2)
            x_train = censor_train + uncensor_train
            x_val = censor_val + uncensor_val
            x_test = censor_test + uncensor_test
            train_target_path = os.path.join(sava_path, key, 'train')
            if not os.path.exists(train_target_path):
                os.makedirs(train_target_path)
            for file in x_train:
                os.makedirs(os.path.join(train_target_path, file.split('/')[-1]))
                copy_dir(file, os.path.join(train_target_path, file.split('/')[-1]))
            #x_val, x_test = train_test_split(x_test, test_size=0.5)
            test_target_path = os.path.join(sava_path, key, 'test')
            if not os.path.exists(test_target_path):
                os.makedirs(test_target_path)
            for file in x_test:
                os.makedirs(os.path.join(test_target_path, file.split('/')[-1]))
                copy_dir(file, os.path.join(test_target_path, file.split('/')[-1]))

            new_validation_path = os.path.join(sava_path, key, 'val')
            if not os.path.exists(new_validation_path):
                os.makedirs(new_validation_path)
            for file in x_val:
                os.makedirs(os.path.join(new_validation_path, file.split('/')[-1]))
                copy_dir(file, os.path.join(new_validation_path, file.split('/')[-1]))
