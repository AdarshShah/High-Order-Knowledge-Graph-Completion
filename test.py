import warnings

import logging
import argparse
import torch
from utils.initialization_utils import (initialize_experiment,
                                        initialize_models)
from preprocess.datasets import DataLoader, MyDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

def test(params):
    dataset = MyDataset(dataset=params.dataset, num_classes=params.num_classes,
                        max_dim=params.max_dim, dim=params.dim, iterations=params.iter)

    cm, baseGnn, writer = initialize_models(params, mode='test')
    def custom_collate(X):
            return X[0]
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=16, collate_fn=custom_collate)
    with torch.no_grad():
        H0 = []
        H1 = []
        H2 = []
        H3 = []
        labels0 = []
        labels1 = []
        with tqdm(dataloader) as tepoch:
            for pos_embeddings, neg_embeddings, laplacians, boundaries, order, idx, label, subgraph in tepoch:
                label, subgraph = label.to(params.device), subgraph.to(params.device)
                pos_embeddings = [ x.to(params.device) if x is not None else None for x in pos_embeddings]
                neg_embeddings = [ x.to(params.device) if x is not None else None for x in neg_embeddings]
                laplacians = [ x.to(params.device) if x is not None else None for x in laplacians]
                boundaries = [ x.to(params.device) if x is not None else None for x in boundaries]
                if order > 0: # makes no sense to perform node classification since node embedding will be 0.
                    pred0 = (torch.sum(pos_embeddings[0][:order+1], dim=0)!=0).long().squeeze()
                    pred1 = (torch.prod(pos_embeddings[0][:order+1], dim=0)!=0).long().squeeze()
                    H0.append((pred0==label).long())
                    H1.append((pred1==label).long())
                    labels0.append(label)
                pred2_pos = baseGnn(subgraph, pos_embeddings[0], order, label)
                pred2_neg = baseGnn(subgraph, neg_embeddings[0], order, label)
                pred3_pos = cm(pos_embeddings, laplacians, boundaries, order, idx, label)
                pred3_neg = cm(neg_embeddings, laplacians, boundaries, order, idx, label)
                # H1.append((torch.round(torch.sigmoid(pred1))==label).long())
                # H2.append((torch.round(torch.sigmoid(pred2))==label).long())
                H2.append(pred2_pos)
                H2.append(pred2_neg)
                H3.append(pred3_pos)
                H3.append(pred3_neg)
                labels1.append(torch.ones_like(pred2_pos))
                labels1.append(torch.zeros_like(pred2_neg))
                torch.cuda.empty_cache()
    H0, H1, H2, H3, labels0, labels1 = torch.stack(H0), torch.stack(H1), torch.cat(H2), torch.cat(H3), torch.stack(labels0), torch.cat(labels1)

    A1 = roc_auc_score(labels0.cpu().numpy(), H0.cpu().numpy(), average='weighted')
    B1 = roc_auc_score(labels0.cpu().numpy(), H1.cpu().numpy(),  average='weighted')
    C1 = roc_auc_score(labels1.cpu().numpy(), H2.cpu().numpy(),  average='weighted')
    D1 = roc_auc_score(labels1.cpu().numpy(), H3.cpu().numpy(),  average='weighted')

    A2 = average_precision_score(labels0.cpu().numpy(), H0.cpu().numpy(), average='weighted')
    B2 = average_precision_score(labels0.cpu().numpy(), H1.cpu().numpy(),  average='weighted')
    C2 = average_precision_score(labels1.cpu().numpy(), H2.cpu().numpy(),  average='weighted')
    D2 = average_precision_score(labels1.cpu().numpy(), H3.cpu().numpy(),  average='weighted')

    result = {
        'auc' : {
            'Union' : A1, 'Intersection' : B1, 'GNN with attention' : C1, 'Simplicial Attention Convolution' : D1
        },
        'auc_pr': {
            'Union' : A2, 'Intersection' : B2, 'GNN with attention' : C2, 'Simplicial Attention Convolution' : D2
        }
    }
    logging.info(f'AUC : ')
    [ logging.info(f'{key} : {value}') for key, value in result['auc'].items() ]
    logging.info(f'AUC PR : ')
    [ logging.info(f'{key} : {value}') for key, value in result['auc_pr'].items() ]

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Inductive H-KGC training')

    parser.add_argument("--dataset", "-d", type=str, help="dataset folder name", required=True)
    parser.add_argument("--num_classes", type=int, help="number of relation types", required=True)
    parser.add_argument("--max_dim", type=int, default=4, help="maximum dimension of simplex to consider")
    parser.add_argument("--dim", type=int, default=None, help="particular dimension to infere on")
    parser.add_argument("--iter", type=int, default=10000, help="number of iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    params = parser.parse_args()

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    
    initialize_experiment(params, mode='test')

    test(params)