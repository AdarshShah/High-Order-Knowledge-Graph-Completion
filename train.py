import argparse
import logging
import warnings
import os

import torch
from tqdm import tqdm

from preprocess.datasets import DataLoader, MyDataset
from utils.initialization_utils import (initialize_experiment, initialize_models, save_models)

warnings.filterwarnings('ignore')

def train(params):
    dataset = MyDataset(dataset=params.dataset, num_classes=params.num_classes, max_dim=params.max_dim, dim=params.dim, iterations=params.iter)

    scn, san, gat, writer = initialize_models(params, mode='train')

    gs = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    ep = 0

    def custom_collate(X):
        return X[0]

    optim1 = torch.optim.Adam(scn.parameters())
    optim2 = torch.optim.Adam(san.parameters())
    optim3 = torch.optim.Adam(gat.parameters())
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=custom_collate)
    with tqdm(dataloader) as tepoch:
        for pos_embeddings, neg_embeddings, laplacians, boundaries, order, idx, label, subgraph in tepoch:
            label, subgraph = label.to(params.device), subgraph.to(params.device)
            pos_embeddings = [x.to(params.device) if x is not None else None for x in pos_embeddings]
            neg_embeddings = [x.to(params.device) if x is not None else None for x in neg_embeddings]
            laplacians = [x.to(params.device) if x is not None else None for x in laplacians]
            boundaries = [x.to(params.device) if x is not None else None for x in boundaries]

            try:
                pos_pred = scn(pos_embeddings, laplacians, boundaries, order, idx, torch.ones_like(pos_embeddings[0][0])).squeeze()
                neg_pred = scn(neg_embeddings, laplacians, boundaries, order, idx, torch.ones_like(pos_embeddings[0][0])).squeeze()
                loss1 += torch.nn.functional.binary_cross_entropy_with_logits(pos_pred, label) + torch.nn.functional.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))

                pos_pred = san(pos_embeddings, laplacians, boundaries, order, idx, torch.ones_like(pos_embeddings[0][0])).squeeze()
                neg_pred = san(neg_embeddings, laplacians, boundaries, order, idx, torch.ones_like(pos_embeddings[0][0])).squeeze()
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(pos_pred, label) + torch.nn.functional.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))

                subgraph = subgraph.to(params.device)
                pos_pred = gat(subgraph, pos_embeddings[0], order, torch.ones_like(pos_embeddings[0][0])).squeeze()
                neg_pred = gat(subgraph, neg_embeddings[0], order, torch.ones_like(pos_embeddings[0][0])).squeeze()
                loss3 += torch.nn.functional.binary_cross_entropy_with_logits(pos_pred, label) + torch.nn.functional.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
                ep += 1
            except:
                pass            

            if ep % params.batch_size == 0:
                loss1 = loss1 / params.batch_size
                loss2 = loss2 / params.batch_size
                loss3 = loss3 / params.batch_size
                optim1.zero_grad()
                loss1.backward()
                optim1.step()
                optim2.zero_grad()
                loss2.backward()
                optim2.step()
                optim3.zero_grad()
                loss3.backward()
                optim3.step()
                writer.add_scalars('Train Loss', {'SimplicialConvolutionModel': loss1.item(), 'SimplicialAttentionModel': loss2.item(), 'GraphAttentionNetwork': loss3.item()}, gs)

                loss1 = 0
                loss2 = 0
                loss3 = 0
                gs += 1
            torch.cuda.empty_cache()

    save_models(params, scn, san, gat)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Inductive H-KGC training')

    parser.add_argument("--experiment_name", "-e", type=str, help="experiment name", required=True)
    parser.add_argument("--dataset", "-d", type=str,
                        help="dataset folder name", required=True)
    parser.add_argument("--num_classes", type=int,
                        help="number of relation types", required=True)
    parser.add_argument("--max_dim", type=int, default=4,
                        help="maximum dimension of simplex to consider")
    parser.add_argument("--dim", type=int, default=None,
                        help="particular dimension to infere on")
    parser.add_argument("--iter", type=int, default=10000,
                        help="number of iterations")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size")
    parser.add_argument("--reset_model", action='store_true',
                        help='reset model')
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument(
        '--disable_cuda', action='store_true', help='Disable CUDA')
    params = parser.parse_args()
    
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    initialize_experiment(params, mode='train')

    train(params)
