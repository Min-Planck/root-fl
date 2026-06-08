from typing import List
import random
import torch
from collections import Counter
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, FashionMNIST
import torchvision.transforms as transforms
from tqdm import tqdm 
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from datasets import load_dataset 
import re   
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, targets, transform=None):
        self.data = input_data
        self.targets = targets
        self.classes = torch.unique(torch.tensor(targets)).tolist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def clean_text(tweet):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = tweet.lower()
    tweet = re.sub(urlPattern, '', tweet)
    tweet = re.sub(userPattern, '', tweet)
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
    tweet = tweet.replace('\r', '').replace('\n', ' ').lower()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7f]', r'', tweet)

    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    tweet = tweet.translate(table)

    tweet = " ".join(word.strip() for word in re.split('#|_', tweet))
    tweet = ' '.join([word if ('$' not in word) and ('&' not in word) else '' for word in tweet.split(' ')])
    tweet = re.sub("\s\s+", " ", tweet)
    return tweet.strip()

def preprocess_text_agnews(root):
    keep = ['text', 'label']
    data = {key: root[key] for key in keep}

    data['text'] = [clean_text(text) for text in data['text']]
    data['label'] = [int(label) for label in data['label']] 

    return data

def get_transform(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name in ['emnist', 'fmnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    else:
        return None
    
def load_agnews():
    dataset = load_dataset("SetFit/ag_news")

    train_raw = dataset['train']
    test_raw = dataset['test']

    train_data = preprocess_text_agnews(train_raw)
    test_data = preprocess_text_agnews(test_raw)

    max_words = 2000
    max_len = 500
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_data['text'])
    
    train_seq = tokenizer.texts_to_sequences(train_data['text'])
    test_seq = tokenizer.texts_to_sequences(test_data['text'])

    train_pad = torch.tensor(pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post'), dtype=torch.long)
    test_pad = torch.tensor(pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post'), dtype=torch.long)

    train_labels = torch.tensor(train_data['label'], dtype=torch.long)
    test_labels = torch.tensor(test_data['label'], dtype=torch.long)

    trainset = CustomDataset(train_pad, train_labels)
    testset = CustomDataset(test_pad, test_labels)

    return trainset, testset

def load_data(dataset: str): 
    datasets = {
        'cifar10': (CIFAR10, 'image'),
        'emnist': (EMNIST, 'image'),
        'fmnist': (FashionMNIST, 'image'),
        'cifar100': (CIFAR100, 'image'),
        'agnews': ('text', 'text')  
    }

    if dataset in datasets:
        if dataset == 'agnews':
            return load_agnews()

        dataset_class, datatype = datasets[dataset]
        transform = get_transform(dataset)

        if dataset in ['cifar10', 'cifar100']:
            trainset = dataset_class("data", train=True, download=True, transform=transform)
            testset = dataset_class("data", train=False, download=True, transform=transform)
        else:
            trainset = dataset_class("data", train=True, download=True, transform=transform)
            testset = dataset_class("data", train=False, download=True, transform=transform)

        return trainset, testset
    
def partition_data_sharding(dataset, num_clients, num_shards_per_client, classes_name):
    num_classes = len(classes_name)
    
    total_shards = num_clients * num_shards_per_client
    
    indices_class = []
    for j in range(num_classes):
        idx = np.array([i for i, lab in enumerate(dataset.targets) if lab == j])
        np.random.shuffle(idx)
        indices_class.append(idx)
        
    sorted_indices = np.concatenate(indices_class)
 
    shards_list = np.array_split(sorted_indices, total_shards)
    
    shards_list_indices = list(range(total_shards))
    np.random.shuffle(shards_list_indices)
    
    ids = []
    label_dist = []
    
    for i in range(num_clients):
        client_indices = []
        start_idx = i * num_shards_per_client
        
        for j in range(num_shards_per_client):
            shard_idx = shards_list_indices[start_idx + j]
            client_indices.extend(shards_list[shard_idx].tolist())
            
        
        ids.append(client_indices)
        
        if isinstance(dataset, CustomDataset):
            counter = Counter(list(map(lambda x: int(dataset.targets[x]), ids[i])))
        else: 
            counter = Counter(list(map(lambda x: dataset[x][1], ids[i])))
            
        label_dist.append({classes_name[j]: counter.get(j, 0) for j in range(num_classes)})

    return ids, label_dist

def partition_data(dataset,
                   num_clients,
                   alpha,
                   classes_name):

    num_classes = len(classes_name)

    client_size = len(dataset) // num_clients
    indices_class = [[] for _ in range(num_classes)]

    for i, lab in enumerate(dataset.targets):
        indices_class[lab].append(i)

    labels = list(range(num_classes))
    ids = []
    label_dist = []

    for i in tqdm(range(num_clients)):
        concentration = torch.ones(len(labels)) * alpha
        dist = Dirichlet(concentration).sample()

        client_indices = []
        for _ in range(client_size):
            if not labels:
                break

            label = random.choices(labels, dist)[0]
            if indices_class[label]:
                id_sample = random.choice(indices_class[label])
                client_indices.append(id_sample)
                indices_class[label].remove(id_sample)

                if not indices_class[label]:
                    dist = renormalize(dist, labels, label)
                    labels.remove(label)

        ids.append(client_indices)
        if isinstance(dataset, CustomDataset):
            counter = Counter(list(map(lambda x: int(dataset.targets[x]), ids[i])))
        else: 
            counter = Counter(list(map(lambda x: dataset[x][1], ids[i])))
        label_dist.append({classes_name[j]: counter.get(j, 0) for j in range(num_classes)})

    return ids, label_dist

def get_train_data(dataset_name,
                   num_clients,
                   batch_size,
                   alphas: list = [0.5, 0.7, 0.9, 1],
                   fractions: list = [0.25, 0.25, 0.25, 0.25],
                   mode: str = 'dirichlet',
                   shards: list = None):

    assert abs(sum(fractions) - 1.0) < 1e-6, "Tổng 'fractions' phải bằng 1"
    assert num_clients > 0, "num_clients phải > 0"

    trainset, testset = load_data(dataset_name)
    classes = trainset.classes

    clients_per_fold = [int(frac * num_clients) for frac in fractions]
    while sum(clients_per_fold) < num_clients:
        for i in range(len(clients_per_fold)):
            clients_per_fold[i] += 1
            if sum(clients_per_fold) == num_clients:
                break

    total_data = len(trainset)
    data_per_fold = [int((num / num_clients) * total_data) for num in clients_per_fold]
    while sum(data_per_fold) < total_data:
        for i in range(len(data_per_fold)):
            data_per_fold[i] += 1
            if sum(data_per_fold) == total_data:
                break

    partition_fold = random_split(trainset, data_per_fold)
    ids, labels_dist = [], []

    for i in range(len(fractions)):
        sub_set = partition_fold[i]
        original_indices = sub_set.indices

        if dataset_name in ['cifar10', 'cifar100', 'agnews']:
            data = [trainset.data[idx] for idx in original_indices]
            targets = [trainset.targets[idx] for idx in original_indices]
        else:
            data = trainset.data[original_indices]
            targets = trainset.targets[original_indices].tolist()

        sub_dataset = CustomDataset(data, targets)

        if mode == 'sharding' and shards is not None and shards[i] > 0:
            id, dist = partition_data_sharding(sub_dataset, clients_per_fold[i], shards[i], classes)
        elif mode == "dirichlet":
            id, dist = partition_data(sub_dataset, clients_per_fold[i], alphas[i], classes)

        id = [[original_indices[idx] for idx in client_ids] for client_ids in id]

        ids.extend(id)
        labels_dist.extend(dist)

    trainloaders = [
        DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(ids[i]))
        for i in range(num_clients)
    ]
    testloader = DataLoader(testset, batch_size=batch_size)

    client_dataset_ratio: float = int(len(trainset) / num_clients) / len(trainset)

    return ids, labels_dist, trainloaders, testloader, client_dataset_ratio