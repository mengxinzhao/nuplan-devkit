import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.data import Data

# TODO: 2. According to the paper complete the polyline subgraphs computation
class GraphLayerProp(MessagePassing):
    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        """初始化 GNN 结构中涉及到的 Aggregator 和 Node Encoder 以及调试信息输出的标志位
        Args:
            in_channels (int)          : 输入的尺寸
            hidden_unit (int, optional): 隐藏层的尺寸. 缺省为 64.
            verbose (bool, optional)   : 输出调试信息的标志位. 缺省为 False.
        """
        # 初始化 GNN 的聚合操作为 Max Pooling
        super(GraphLayerProp, self).__init__(aggr='max')  

        pass

    def forward(self, x, edge_index):
        """GNN 层的推理流程，对应论文中的 Figure 3
        Args:
            x         : Polyline Features
            edge_index: Adjacency Array
        Returns: Output Nodes Features
        """

        # 调用 propagate 方法执行 message update 和 max pooling aggregate
        pass

    def message(self, x_j):
        """Message Passing 的消息生成阶段
        Args:
            x_j: Node feature

        Returns: Exactly the input
        """    
        pass

    def update(self, aggr_out, x):
        """Concat 阶段
        Args:
            aggr_out: Aggregator output
            x       : Node Encoder output
        Returns: Output Node Features
        """
        pass


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        """初始化 Polyline Subgraph 的参数

        Args:
            in_channels (int)                  : 输入的维度
            num_subgraph_layers (int, optional): GNN 的层数. 缺省为 3.
            hidden_unit (int, optional)        : 隐层的维度. 缺省为 64.
        """
        # 初始化 nn.Module
        super(SubGraph, self).__init__()
        
        # 初始化 subgraph 的层数
        self.num_subgraph_layres = num_subgraph_layres
        
        # 初始化 GNN 的多个图层
        self.layer_seq = nn.Sequential()

        # 循环创建多个图层，并将它们塞入 layer_seq 中
        for i in range(num_subgraph_layres):
            # 添加 GraphLayerProp 图层并命名为 glp_i
            self.layer_seq.add_module(
                f'glp_{i}', 
                GraphLayerProp(in_channels, hidden_unit)
            )

            # GraphLayerProp 每次输出的节点数量都会 double 
            # 所以这里的 in_channels 会一直乘以2            
            in_channels *= 2

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        # 从 sub_data 中取出向量化的特征 x 和邻接矩阵 edge_index 信息
        x = sub_data.x.to(torch.float32)
        edge_index = sub_data.edge_index.to(torch.long)

        # 遍历 layer_seq 中的所有层进行推理
        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)

        # update features
        sub_data.x = x

        # max pooling to extract subgraph features
        out_data = max_pool(sub_data.cluster, sub_data)

        # normalize output
        out_data.x = out_data.x / (out_data.x.norm(dim=0) + 1e-6)

        return out_data
