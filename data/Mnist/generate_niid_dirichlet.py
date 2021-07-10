from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

random.seed(42)
np.random.seed(42)

# 狄利克雷函数是一个定义在实数范围上、值域不连续的函数。
# 狄利克雷函数的图像以Y轴为对称轴，是一个偶函数，它处处不连续，处处极限不存在，不可黎曼积分。这是一个处处不连续的可测函数。

# 按照类别分开数据
def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def get_dataset(mode='train'):
    '''
    分类数据
    :param mode: 数据类型
    :return: data_by_class: 按照标签分好的数据
             n_sample: 数据总数目
             SRC_N_CLASS: 每类标签数量
    '''
    transform = transforms.Compose(
       [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # 加载并转成torch，进行标准化
    dataset = MNIST(root='/home/liuyuan/st_codes/fed_pytorch_mod/dataset/mnist/data', train=True if mode == 'train' else False, download=True, transform=transform)
    # dataset = MNIST(root='./data', train=True if mode=='train' else False, download=True, transform=transform)

    n_sample = len(dataset.data) # 数据长度
    SRC_N_CLASS = len(dataset.classes)# 类别数
    # full batch
    trainloader = DataLoader(dataset, batch_size=n_sample, shuffle=False)

    print("Loading data from storage ...")
    for _, xy in enumerate(trainloader, 0):
        dataset.data, dataset.targets = xy

    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(
        dataset.data.cpu().detach().numpy(),
        dataset.targets.cpu().detach().numpy(),
        SRC_N_CLASS # 标签类别数
    )
    print(f"{mode.upper()} SET:\n  Total #samples: {n_sample}. sample shape: {dataset.data[0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])

    return data_by_class, n_sample, SRC_N_CLASS

def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]

def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sampling_ratio=0.5):
    '''
    划分训练数据
    :param data: 分好类别的数据
    :param n_sample: 数据总量
    :param SRC_CLASSES: 标签数目
    :param NUM_USERS: 客户端数量
    :param min_sample: 最小采样
    :param alpha: 分布参数
    :param sampling_ratio: 采样比率
    :return:
    '''
    min_sample = 10 #len(SRC_CLASSES) * min_sample
    min_size = 0 # track minimal samples per user
    # 采样
    ###### Determine Sampling #######
    while min_size < min_sample: # 当最小数量达到要求时结束采样
        print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        # 每个客户端最大采样数量
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int( min(max_samples_per_user, int(sampling_ratio * len(data[l]))) ) # 第l个标签采样数目：最大和从该标签中选一个最小的
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))
            # dirichlet sampling from this label
            # np.repeat(alpha, NUM_USERS): 将alpha重复NUM_USERS（客户端数）次
            proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            # re-balance proportions
            # 重新平衡比例
            proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
            proportions=proportions / proportions.sum() # 归一化
            proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1] # 为什么去掉最后
            # participate data of that label
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                # add new idex to the user
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size=min(samples_per_user)

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    # 按照samples_per_user中的划分的数据取值
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test: # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels =  list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS )
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y

def main():
    parser = argparse.ArgumentParser()
    # 存储方式
    parser.add_argument("--format", "-f", type=str, default="pt", help="Format of saving: pt (torch.save), json", choices=["pt", "json"])
    # 分类标签数
    parser.add_argument("--n_class", type=int, default=10, help="number of classification labels")
    # 每个用户的最小样本数
    parser.add_argument("--min_sample", type=int, default=10, help="Min number of samples per user.")
    # 训练样本抽样比例
    parser.add_argument("--sampling_ratio", type=float, default=0.05, help="Ratio for sampling training samples.")
    # 是否允许每个用户看不到测试标签
    parser.add_argument("--unknown_test", type=int, default=0, help="Whether allow test label unseen for each user.")
    # Dirichelt分布中的α（较小意味着较大的异质性）
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    # 本地客户端的数量，应为10的倍数
    parser.add_argument("--n_user", type=int, default=10,
                        help="number of local clients, should be muitiple of 10.")
    args = parser.parse_args()
    print()
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Min # of samples per uesr: {}".format(args.min_sample))
    print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.n_user

    # Setup directory for train/test data
    path_prefix = f'u{args.n_user}c{args.n_class}-alpha{args.alpha}-ratio{args.sampling_ratio}'

    def process_user_data(mode, data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0):
        '''
        客户端数据处理
        :param mode: 数据类型
        :param data: 分好类别的数据
        :param n_sample: 数据总量
        :param SRC_CLASSES: 类别数目
        :param Labels: None
        :param unknown_test: 0
        :return:
        '''
        if mode == 'train':
            X, y, Labels, idx_batch, samples_per_user  = devide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha, args.sampling_ratio)
        if mode == 'test':
            assert Labels != None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data, Labels, unknown_test)
        dataset={'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname='f_{0:05d}'.format(i)
            dataset['users'].append(uname)
            dataset['user_data'][uname]={
                'x': torch.tensor(X[i], dtype=torch.float32),
                'y': torch.tensor(y[i], dtype=torch.int64)}
            dataset['num_samples'].append(len(X[i]))

        print("{} #sample by user:".format(mode.upper()), dataset['num_samples'])

        data_path=f'./{path_prefix}/{mode}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path=os.path.join(data_path, "{}.".format(mode) + args.format)
        if args.format == "json":
            raise NotImplementedError(
                "json is not supported because the train_data/test_data uses the tensor instead of list and tensor cannot be saved into json.")
            with open(data_path, 'w') as outfile:
                print(f"Dumping train data => {data_path}")
                json.dump(dataset, outfile)
        elif args.format == "pt":
            with open(data_path, 'wb') as outfile:
                print(f"Dumping train data => {data_path}")
                torch.save(dataset, outfile)
        if mode == 'train':
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                # train_idx_batch, train_samples_per_user
                n_samples_for_u = 0
                for l in sorted(list(Labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(len(Labels[u]), n_samples_for_u, u))
            return Labels, idx_batch, samples_per_user


    print(f"Reading source dataset.")
    train_data, n_train_sample, SRC_N_CLASS = get_dataset(mode='train')
    test_data, n_test_sample, SRC_N_CLASS = get_dataset(mode='test')
    SRC_CLASSES=[l for l in range(SRC_N_CLASS)]
    random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES))) # 类别数目
    Labels, idx_batch, samples_per_user = process_user_data('train', train_data, n_train_sample, SRC_CLASSES)
    process_user_data('test', test_data, n_test_sample, SRC_CLASSES, Labels=Labels, unknown_test=args.unknown_test)
    print("Finish Generating User samples")

if __name__ == "__main__":
    main()