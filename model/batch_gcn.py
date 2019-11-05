import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class GCN_simple(nn.Module):
    """GCN分支学习semantic relation，CNN分支生成query hash code、database hash code
    Parameters
        cnn_pretrained: bool
        True: 加载预训练AlexNet模型; False: 加载未训练模型
        cnn_feature_dim: int
        Alexnet fc7特征维度
        cnn_hidden_dim: int
        由fc7特征到cnn hash prob的隐藏层维度(cnn feature —> cnn hidden —> cnn hash prob)
        gcn_hidden_dim: int
        两层GCN,由fc7特征到gcn hash prob的隐藏层维度(cnn feature —> gcn hidden —> gcn hash prob)
        dropout: float
        随机丢弃的神经元比例
        semantic_dim: int
        word embedding维度
        hash_bits: int
        hash code的长度
    Inputs
        op: str
        'train': 训练阶段; 'eval': 检索测试阶段
        input_feat: pytorch tensor, with shape [batch_size, image size]
        dataloader加载的原始图像
        semantic_adj: pytorch tensor, with shape [batch_size, batch_size]
        semantic邻接矩阵
    Outputs 
        if op == 'train':
        cnn_hash_prob: pytorch tensor, with shape [batch_size, hash bits]
        经过sigmoid激活层的cnn hash prob
        gcn_feature: pytorch tensor, with shape [batch_size, hash bits]
        经过两层GCN的gcn feature
        gcn_hash_code: pytorch tensor, {0, 1} with shape [batch_size, hash bits]
        量化后的gcn hash code (gcn feature -> sigmoid() -> sign())
        recon_semantic: pytorch tensor, with shape [batch_size, word embedding size]
        重建class semantic vector
        elif op == 'eval':
        cnn_hash_prob: pytorch tensor, with shape [batch_size, hash bits]
        量化后可以生成hash code
    """
    def __init__(self, cnn_pretrained, cnn_feature_dim, cnn_hidden_dim, gcn_hidden_dim,
                 dropout, batch_normalize, semantic_dim, hash_bits):
        super(GCN_simple, self).__init__()

        self.dropout = dropout
        self.batch_normalize = batch_normalize

        self.cnn_model = torchvision.models.alexnet(pretrained=cnn_pretrained)
        self.cnn_classifier = nn.Sequential(*list(self.cnn_model.classifier.children())[:-1]) #fc7-ReLU feature
        #self.cnn_classifier = nn.Sequential(*list(self.cnn_model.classifier.children())[:-2]) #fc7 feature

        self.cnn_hash_layer = nn.Sequential(
            #nn.Dropout(self.dropout),
            nn.Linear(cnn_feature_dim, cnn_hidden_dim),
            nn.ReLU(inplace=True),
            #nn.Dropout(self.dropout),
            nn.Linear(cnn_hidden_dim, hash_bits),
            )

        self.gconv1 = nn.Linear(cnn_feature_dim, gcn_hidden_dim)
        self.bn1 = nn.BatchNorm1d(gcn_hidden_dim)
        self.activate1 = nn.LeakyReLU(0.2, inplace=True)

        self.gconv2 = nn.Linear(gcn_hidden_dim, hash_bits)
        self.bn2 = nn.BatchNorm1d(hash_bits)
        self.activate2 = nn.LeakyReLU(0.2, inplace=True)

        self.recon_semantic_layer = nn.Linear(hash_bits,semantic_dim)
        #self.recon_label_layer = nn.Linear(hash_bits,seen_class_num)

    def get_cnn_feature(self, input_feat):
        maxpool5_feature = self.cnn_model.features(input_feat).view(input_feat.size(0), 256 * 6 * 6)
        fc7_feature = self.cnn_classifier(maxpool5_feature)
        return fc7_feature

    def get_cnn_hash(self, cnn_feature):
        cnn_hash = self.cnn_hash_layer(cnn_feature)
        cnn_hash_prob = torch.sigmoid(cnn_hash)
        return cnn_hash_prob


    def get_gcn_feature(self, cnn_feature, semantic_adj):
        # layer 1
        out = self.gconv1(cnn_feature)
        out = torch.mm(semantic_adj, out)
        out = self.bn1(out)
        out = self.activate1(out)

        # layer 2
        out = self.gconv2(out)
        out = torch.mm(semantic_adj, out)
        out = self.bn2(out)
        out = self.activate2(out)

        return out

    def get_gcn_hash(self, gcn_feature):
        gcn_hash_prob = torch.sigmoid(gcn_feature)
        gcn_hash_code = (torch.sign(gcn_hash_prob - 0.5) + 1.0) / 2.0
        return gcn_hash_code

    def get_recon_semantic(self, gcn_hash):
        recon_semantic = self.recon_semantic_layer(gcn_hash)
        return recon_semantic

    def get_recon_label(self, gcn_hash):
        recon_label = self.recon_label_layer(gcn_hash)
        return recon_label

    def forward(self, op, input_feat, semantic_adj=None):
        if op == 'train':
            cnn_feature = self.get_cnn_feature(input_feat)
            cnn_hash_prob = self.get_cnn_hash(cnn_feature)
            gcn_feature = self.get_gcn_feature(cnn_feature, semantic_adj)
            gcn_hash_code = self.get_gcn_hash(gcn_feature)
            recon_semantic = self.get_recon_semantic(gcn_hash_code)
            #recon_label = self.get_recon_label(gcn_hash_code)
            return cnn_hash_prob, gcn_feature, gcn_hash_code, recon_semantic
        elif op == 'eval':
            cnn_feature = self.get_cnn_feature(input_feat)
            cnn_hash_prob = self.get_cnn_hash(cnn_feature)
            return cnn_hash_prob
