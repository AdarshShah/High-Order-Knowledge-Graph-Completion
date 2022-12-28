import os
import logging
from models.model import SimplicialModel1, BaseGNN
from torch.utils.tensorboard import SummaryWriter
import torch

def initialize_experiment(params, mode='train'):
    
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    
    if mode=='train':
        file_handler = logging.FileHandler(os.path.join(exps_dir, f"log_train.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(exps_dir, f"log_test.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

def initialize_models(params, mode='train'):

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    model_dir = os.path.join(exps_dir, 'models')

    writer = SummaryWriter(f'/home/adarsh/H-KGC/datasets/{params.dataset}/logs')
    cm = SimplicialModel1(classes=params.num_classes, dim=params.max_dim, device=params.device).to(params.device)
    baseGnn = BaseGNN(classes=params.num_classes, dim=params.max_dim, device=params.device).to(params.device)

    if mode=='train':
        logging.info('Initializing new models')
        return cm, baseGnn, writer
    else:
        logging.info(f'loading models from {model_dir}')
        cm.load_state_dict(torch.load(os.path.join(model_dir,'simplicial_model.pth')))
        baseGnn.load_state_dict(torch.load(os.path.join(model_dir,'base_gnn_model.pth')))
        return cm, baseGnn, writer

def save_models(params, cm:torch.nn.Module, baseGNN:torch.nn.Module):

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    model_dir = os.path.join(exps_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.info(f'saving models in {model_dir}')
    torch.save(cm.state_dict(), os.path.join(model_dir, 'simplicial_model.pth'))
    torch.save(baseGNN.state_dict(), os.path.join(model_dir, 'base_gnn_model.pth'))
