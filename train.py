import os
import time
import argparse
import numpy as np

import torch
from model.batch_gcn import GCN_simple
from util.conver_to_graph import build_adjacency,normalize_adj

from data.dataloader import load_data
from data.transform import encode_onehot
from eval.calc_map import calc_map


def compute_loss(cnn_hash_prob, 
                 gcn_feature, 
                 gcn_hash_code, 
                 recon_semantic, 
                 input_semantic
                 ):
    bceloss = torch.nn.BCEWithLogitsLoss()
    q_loss = bceloss(gcn_feature, gcn_hash_code)
    mseloss = torch.nn.MSELoss()
    p_loss = mseloss(recon_semantic, input_semantic)
    hash_loss = mseloss(cnn_hash_prob, gcn_hash_code)
    #loss = q_loss + alpha*p_loss + beta*hash_loss
    return q_loss, p_loss, hash_loss

def train(opt, 
          query_loader, 
          train_loader, 
          database_loader, 
          query_targets, 
          train_targets, 
          database_targets
          ):
    gcn_model = GCN_simple(opt.cnn_pretrained, 
                           opt.cnn_feature_dim, 
                           opt.cnn_hidden_dim, 
                           opt.gcn_hidden_dim,
                           opt.dropout, 
                           opt.batch_normalize, 
                           opt.semantic_dim, 
                           opt.hash_bits
                           )
    gcn_model.to(opt.device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=opt.lr, weight_decay=1e-5)
    
    best_map = 0.0
    last_model = None
    for epoch in range(opt.epochs):
        total_loss = 0.0
        start = time.time()
        gcn_model.train()
        for step, (input_feat, input_label ,input_semantic, index) in enumerate(train_loader):
            
            input_feat = input_feat.to(opt.device)
            input_label = input_label.to(opt.device)
            input_semantic = input_semantic.to(opt.device)

            semantic_adj = build_adjacency(input_semantic, input_semantic, opt.adj_scaler)
            semantic_adj = normalize_adj(semantic_adj, opt.overflow_margin, opt.device)
            
            cnn_hash_prob, gcn_feature, gcn_hash_code, recon_semantic = gcn_model('train', 
                                                                                  input_feat, 
                                                                                  semantic_adj
                                                                                  )
            optimizer.zero_grad()
            q_loss, p_loss, hash_loss = compute_loss(cnn_hash_prob, 
                                                     gcn_feature, 
                                                     gcn_hash_code, 
                                                     recon_semantic, 
                                                     input_semantic
                                                     )
            train_loss = q_loss + opt.alpha*p_loss + opt.beta*hash_loss
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            '''
            print("[Epoch {}/{}] [Batch {}/{}] [Total loss: {:.4f}] "\
                  "[q loss: {:.4f}] [p loss: {:.4f}] [hash loss: {:.4f}]".format(epoch+1, 
                                                                                 opt.epochs, 
                                                                                 step+1, 
                                                                                 len(train_loader), 
                                                                                 train_loss.item(), 
                                                                                 q_loss.item(), 
                                                                                 p_loss.item(), 
                                                                                 hash_loss.item()))
            '''
        if (epoch+1) % opt.evaluate_freq == 0:
            train_meanAP, query_meanAP = evaluate(gcn_model, 
                                                  query_loader, 
                                                  train_loader, 
                                                  database_loader, 
            								      query_targets, 
                                                  train_targets, 
                                                  database_targets, 
                                                  opt.hash_bits, 
                                                  opt.device, 
                                                  opt.topk
                                                  )
            # Save best result
            if best_map < query_meanAP:
                if last_model:
                    os.remove(os.path.join('result', last_model))
                best_map = query_meanAP
                last_model = 'model_{:.4f}.t'.format(best_map)
                torch.save(gcn_model, os.path.join('result', last_model))

            print('[Epoch: {}] [time:{:.4f}] [loss: {:.4f}] '\
                  '[train_map: {:.4f}] [query_map: {:.4f}]'.format(epoch+1,
                                                                   time.time()-start,
                                                                   total_loss/len(train_loader),
                                                                   train_meanAP,
                                                                   query_meanAP))

def evaluate(model, 
             query_loader, 
             train_loader, 
             database_loader, 
			 query_targets, 
             train_targets, 
             database_targets, 
             hash_bits, 
             device, 
             topk):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_loader, hash_bits, device).to(device)
    train_code = generate_code(model, train_loader, hash_bits, device).to(device)
    database_code = generate_code(model, database_loader, hash_bits, device).to(device)

    # Compute map
    train_meanAP = calc_map(train_code, database_code, train_targets, database_targets, device,topk)
    query_meanAP = calc_map(query_code, database_code, query_targets, database_targets, device,topk)

    model.train()

    return train_meanAP, query_meanAP

def generate_code(model, dataloader, code_length, device):
    """generate hash code
    Parameters
        model: Model
        CNN model
        dataloader: DataLoader
        Dataloader
        code_length: int
        Hash code length
        device: str
        GPU or CPU
    Returns
        code: Tensor
        Hash code
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for feature, label, semantic, index in dataloader:
            feature = feature.to(device)
            outputs = model('eval', feature)
            outputs_code = torch.sign(outputs - 0.5)
            code[index, :] = outputs_code.cpu()
    return code

def load_parse():
    """Load configuration
    Parameters
        None
    Returns
        opt: parser
        Configuration
    """
    parser = argparse.ArgumentParser(description='ZSH_GCN')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset used to train (default: cifar10)')
    parser.add_argument('--data-path', default='/data3/huangmeixue/Dataset/cifar-10/pytorch-cifar', type=str,
                        help='Path of cifar10 dataset')
    parser.add_argument('--semantic-path', default='/data3/huangmeixue/Dataset/GoogleNews-vectors-negative300.bin', type=str,
                        help='Path of word2vec model')
    parser.add_argument('--unseen-class', default='truck', type=str,
                        help='Name of unseen class')
    parser.add_argument('--seen-class-num', default=9, type=int,
                        help='Number of seen classes(default: 9)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query(default: 1000)')
    parser.add_argument('--num-train', default=10000, type=int,
                        help='Number of train(default: 10000)')

    parser.add_argument('--gpu', default=1, type=int,
                        help='Use gpu(default: 0. -1: use cpu)')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Maximum iteration (default: 30)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size(default: 100)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers(default: 0)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate(default: 1e-4)')

    parser.add_argument('--cnn-pretrained', default=True, type=bool,
                        help='Whether to load pretrain-cnn model (default: True)')
    parser.add_argument('--cnn-feature-dim', default=4096, type=int,
                        help='Dimension of pretrain-cnn feature (default: 4096)')
    parser.add_argument('--cnn-hidden-dim', default=1024, type=int,
                        help='Dimension of cnn hidden layer (default: 1024)')
    parser.add_argument('--semantic-dim', default=300, type=int,
                        help='Dimension of word embedding (default: 300)')
    parser.add_argument('--gcn-hidden-dim', default=1024, type=int,
                        help='Dimension of gcn hidden layer (default: 1024)')
    parser.add_argument('--batch-normalize', default=True, type=bool,
                        help='Whether to use batch normalize (default: True)')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout of full connect (default: 0.5)')
    parser.add_argument('--adj-scaler', default=0.01, type=float,
                        help='Scaler of construct adjacency matrix (default: 0.01)')
    parser.add_argument('--overflow-margin', default=1e-8, type=float,
                        help='overflow margin to normalize adjacency matrix (default: 1e-8)')  
    parser.add_argument('--hash-bits', default=16, type=int,
                        help='Binary hash code length (default: 16)')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Scaler of reconstruct semantic loss (default: 0.5)')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='Scaler of hash loss (default: 0.1)')

    parser.add_argument('--topk', default=5000, type=int,
                        help='Compute map of top k (default: 5000)')
    parser.add_argument('--evaluate-freq', default=1, type=int,
                        help='Frequency of evaluate (default: 1)')

    return parser.parse_args()

if __name__ == '__main__':
    opt = load_parse()
    if opt.gpu == -1:
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:%d" % opt.gpu)
    # Load data
    query_dataloader, train_dataloader, database_dataloader = load_data(opt)

    # onehot targets
    if opt.dataset == 'cifar10':
        query_targets = torch.FloatTensor(encode_onehot(query_dataloader.dataset.targets)).to(opt.device)
        train_targets = torch.FloatTensor(encode_onehot(train_dataloader.dataset.targets)).to(opt.device)
        database_targets = torch.FloatTensor(encode_onehot(database_dataloader.dataset.targets)).to(opt.device)

    train(opt, query_dataloader, train_dataloader, database_dataloader, query_targets, train_targets, database_targets)