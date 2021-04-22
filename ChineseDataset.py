import torch
import json
from ltp import LTP
from torch_geometric.data import InMemoryDataset, download_url,Data
import re
import sys
from tqdm import tqdm



class ChineseDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(ChineseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    ltp = LTP()
    rel_tag = ['SBV', 'VOB', 'IOB', 'FOB', 'DBL', 'ATT', 'ADV', 'CMP', 'COO', 'POB', 'LAD', 'RAD', 'IS', 'HED', 'WP']
    pos_tag = ['ROOT', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns',
               'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x','z']

    @property
    def raw_file_names(self):
        return ['CED.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def cut_sent(self,para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def pos_one_hot(self,pos_str):
        index = ChineseDataset.pos_tag.index(pos_str)
        t = torch.tensor(index)
        return torch.nn.functional.one_hot(t, num_classes=31).numpy().tolist()

    def rel_one_hot(self,rel_str):
        index = ChineseDataset.rel_tag.index(rel_str)
        t = torch.tensor(index)
        return torch.nn.functional.one_hot(t, num_classes=15).numpy().tolist()

    def process_data(self,text,label):
        sents = self.cut_sent(text)
        seg, hidden = self.ltp.seg(sents)
        pos = self.ltp.pos(hidden)
        dep = self.ltp.dep(hidden)
        index = 0
        x_arr = []
        edge_index_arr = [[], []]
        edge_attr_arr = []
        for j, each_pos in enumerate(pos):

            if j != 0:
                index = index + len(pos[j - 1]) + 1
            x_arr.append(self.pos_one_hot('ROOT'))

            for i, node in enumerate(each_pos):
                x_arr.append(self.pos_one_hot(node))

            for edge_info in dep[j]:
                edge_index_arr[0].append(edge_info[0] + index)
                edge_index_arr[1].append(edge_info[1] + index)
                edge_attr_arr.append(self.rel_one_hot(edge_info[2]))

        # 节点特征向量
        # 生成onehot词性即可
        x = torch.tensor(x_arr, dtype=torch.float)

        # 第一个列表是所有边上起始节点的 index，第二个列表是对应边上目标节点的 index
        edge_index = torch.tensor(edge_index_arr, dtype=torch.long)

        # 图分类标签，0真1假
        y = torch.tensor([label], dtype=torch.int64)
        edge_attr = torch.tensor(edge_attr_arr, dtype=torch.float)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def process(self):
        train_set = []
        with open(self.raw_file_names[0],'r') as f:
            data_set=f.read()
            train_set = json.loads(data_set)
        # edge_attr设置边属性，借鉴one-hot对节点词性编码，或干脆引入词嵌入
        # Read data into huge `Data` list.
        data_list = []
        total=len(train_set)
        i=0
        for each_data in tqdm(train_set):
            data_list.append(self.process_data(each_data['text'], each_data['label']))
        # 每个图中可以有多个独立子图（多句话），注意编号就可以了
        # data = Data(x=x,y=y,edge_index=edge_index)
        # 把每个data放进list
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset=ChineseDataset(root='THUDataset')
