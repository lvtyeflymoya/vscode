import torch
import gzip
import csv
from torch.utils.data import DataLoader
import time
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 网络参数
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = False

class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'D:\\vscode\python\Pytorch_Tutorial\RNN_classifier\\names_train.csv.gz' if is_train_set else 'D:\\vscode\python\Pytorch_Tutorial\RNN_classifier\\names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
    
    def __len__(self):
        return self.len
    
    def getCountryDict(self):
        country_dict = {}
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict
    
    def idx2country(self, index):
        return self.country_list[index]
    
    def getCountriesNum(self):
        return self.country_num



# 数据准备
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()



class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)
    
    def forward(self, input, seq_lengths):
        # input shape: BxS->SxB
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size=batch_size)
        embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
    
def name2list(name):
    """将名字转换成列表"""
    arr = [ord(c) for c in name]
    return arr, len(arr)

def make_tensors(names, countries):
    """将姓名和国家转换成tensor"""
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # make tensors of name, BatchSize x Seqlen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor),\
           create_tensor(seq_lengths),\
           create_tensor(countries)


def create_tensor(tensor):
    """
    将输入的张量转换为对应设备上的张量
    """
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def trainModel():
    toltal_loss = 0
    for i, (names, countries) in enumerate(trainloader, 0):
        inputs, seq_lengths, target = make_tensors(names=names, countries=countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toltal_loss += loss.item()
        if i % 10 == 9:
            print(f"[{time_since(start)}] Epoch {epoch}", end='')
            print(f"[{i * len(inputs)}/{len(trainset)}]", end='')
            print(f"loss={toltal_loss/(i * len(inputs))}")
        return toltal_loss
    


def testModel():
    correct = 0
    toltal = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names=names, countries=countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct +=pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / toltal)
        print (f'Test set: Accuracy: {correct}/{toltal} ({percent}%)\n')
        return correct / toltal

  
if __name__ == "__main__":
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER) 
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("trianing for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
