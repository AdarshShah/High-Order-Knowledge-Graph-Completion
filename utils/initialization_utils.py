import os
import logging
from models.model import SimplicialAttentionModel, SimplicialConvolutionModel, GATModel
from torch.utils.tensorboard import SummaryWriter
import torch

def initialize_experiment(params, mode='train'):
    
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    cache_dir = os.path.join(exps_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    logs_dir = os.path.join(exps_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if mode=='train':
        file_handler = logging.FileHandler(os.path.join(exps_dir, f"log_train.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(exps_dir, f"log_test.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    params.test_csv = os.path.join(exps_dir, f'{params.experiment_name}_scores.csv')
    if not os.path.exists(path=params.test_csv):
        with open(params.test_csv, 'w') as f:
            f.write(','.join(['Dims','Iter','AUC_union','AUC_intersect','AUC_GAN','AUC_SCN','AUC_SAN','AUC_pr_union','AUC_pr_intersect','AUC_pr_GAN','AUC_pr_SCN','AUC_pr_SAN'])+'\n')

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

def initialize_models(params, mode='train'):

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    model_dir = os.path.join(exps_dir, 'models')

    writer = SummaryWriter(os.path.join(params.main_dir, exps_dir, 'logs'))
    cm1 = SimplicialConvolutionModel(classes=params.num_classes, dim=params.max_dim, device=params.device).to(params.device)
    cm2 = SimplicialAttentionModel(classes=params.num_classes, dim=params.max_dim, device=params.device).to(params.device)
    baseGnn = GATModel(classes=params.num_classes, dim=params.max_dim, device=params.device).to(params.device)

    if params.reset_model and mode=='train':
        logging.info('Initializing new models')
        return cm1, cm2, baseGnn, writer
    else:
        logging.info(f'loading models from {model_dir}')
        cm1.load_state_dict(torch.load(os.path.join(model_dir,f'{params.experiment_name}_simplicial_model_cnn.pth')))
        cm2.load_state_dict(torch.load(os.path.join(model_dir,f'{params.experiment_name}_simplicial_model_attn.pth')))
        baseGnn.load_state_dict(torch.load(os.path.join(model_dir,f'{params.experiment_name}_base_gnn_model.pth')))
        return cm1, cm2, baseGnn, writer

def save_models(params, cm1:torch.nn.Module, cm2:torch.nn.Module, baseGNN:torch.nn.Module):

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'datasets', f'{params.dataset}')
    model_dir = os.path.join(exps_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.info(f'saving models in {model_dir}')
    torch.save(cm1.state_dict(), os.path.join(model_dir, f'{params.experiment_name}_simplicial_model_cnn.pth'))
    torch.save(cm2.state_dict(), os.path.join(model_dir, f'{params.experiment_name}_simplicial_model_attn.pth'))
    torch.save(baseGNN.state_dict(), os.path.join(model_dir, f'{params.experiment_name}_base_gnn_model.pth'))
