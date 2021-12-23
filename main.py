# Main script for CoMix.
import params
from core import train_comix
from models import *
from utils import *
from dataset import *
import argparse
import os

parser = argparse.ArgumentParser(description='All arguments for the program.')

parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--dataset_name', type=str, default='UCF-HMDB', help='Name of the dataset from \'UCF-HMDB\', \'Jester\', and \'Epic-Kitchens\'.')
parser.add_argument('--src_dataset', type=str, default='UCF', help='Name of the SOURCE DOMAIN e.g. UCF, Epic-Kitchens-D1, etc.')
parser.add_argument('--tgt_dataset', type=str, default='HMDB', help='Name of the TARGET DOMAIN e.g. HMDB, Epic-Kitchens-D2, etc.')
parser.add_argument('--model_root', type=str, default='./checkpoints', help='Directory to save the models.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the framework.')
parser.add_argument('--learning_rate_ws', type=float, default=0.01, help='Learning rate for the Warmstart.')
parser.add_argument('--save_in_steps', type=int, default=500, help='Save models with this frequency.')
parser.add_argument('--log_in_steps', type=int, default=50, help='Log with this frequency.')
parser.add_argument('--eval_in_steps', type=int, default=50, help='Validate with this frequency.')
parser.add_argument('--momentum', type=float, default=0.9, help='For SGD optimizer.')
parser.add_argument('--num_iter_warmstart', type=int, default=4000, help='Number of iterations to warmstart.')
parser.add_argument('--num_iter_adapt', type=int, default=10000, help='Number of iterations to adapt (train CoMix).')
parser.add_argument('--warmstart_models', type=str, default='True', help='Whether to warmstart the models or not.')
parser.add_argument('--pseudo_threshold', type=float, default=0.7, help='Threshold value for TPL.')
parser.add_argument('--manual_seed', type=int, default=1, help='Seed for Random Initialization.')
parser.add_argument('--warmstart_graph', type=str, default='None', help='Load checkpoints from.')
parser.add_argument('--lambda_tpl', type=float, default=0.01, help='Coefficient to multiply the TPL loss.')
parser.add_argument('--Temperature', type=float, default=0.5, help='Temperature for the SimCLR loss.')
parser.add_argument('--warmstart_i3d', type=str, default='None', help='Warmstart i3d from.')
parser.add_argument('--num_segments', type=int, default=16, help='Number of segments (clips) per batch.')
parser.add_argument('--auto_resume', type=str, default='False', help='Auto resume.')
parser.add_argument('--random_aux', type=str, default='True', help='Random Aux.')
parser.add_argument('--lambda_bgm', type=float, default=0.1, help='Coefficient to multiply the BGM loss.')
parser.add_argument('--max_gamma', type=float, default=0.5, help='Max value of Gamma for MixUp.')
parser.add_argument('--base_dir', type=str, default='./data', help='Base directory for data.')

args = parser.parse_args()

if __name__ == '__main__':

    init_random_seed(args.manual_seed)

    params.batch_size = args.batch_size
    params.dataset_name = args.dataset_name
    params.src_dataset = args.src_dataset
    params.tgt_dataset = args.tgt_dataset
    params.model_root = args.model_root
    params.learning_rate = args.learning_rate
    params.learning_rate_ws = args.learning_rate_ws
    params.save_in_steps = args.save_in_steps
    params.log_in_steps = args.log_in_steps
    params.eval_in_steps = args.eval_in_steps
    params.momentum = args.momentum
    params.num_iter_warmstart = args.num_iter_warmstart
    params.num_iter_adapt = args.num_iter_adapt
    params.warmstart_models = args.warmstart_models
    params.pseudo_threshold = args.pseudo_threshold
    params.manual_seed = args.manual_seed
    params.warmstart_graph = args.warmstart_graph
    params.lambda_tpl = args.lambda_tpl
    params.Temperature = args.Temperature
    params.warmstart_i3d = args.warmstart_i3d
    params.num_segments = args.num_segments
    params.auto_resume = args.auto_resume
    params.random_aux = args.random_aux
    params.lambda_bgm = args.lambda_bgm
    params.max_gamma = args.max_gamma
    params.base_dir = args.base_dir

    print(args)

    if params.dataset_name=="UCF-HMDB":
        if params.src_dataset=="UCF" and params.tgt_dataset=="HMDB":
            source_dataset = VideoDataset_UCFHMDB(csv_file='./video_splits/ucf101_train_hmdb_ucf.csv', dataset_name='ucf', transform=None, base_dir = params.base_dir)
            target_dataset = VideoDataset_UCFHMDB(csv_file='./video_splits/hmdb51_train_hmdb_ucf.csv', dataset_name='hmdb', transform=None, base_dir = params.base_dir)
            target_dataset_eval = VideoDataset_UCFHMDB(csv_file='./video_splits/hmdb51_val_hmdb_ucf.csv', dataset_name='hmdb', transform=None, base_dir = params.base_dir, is_test=True)
        elif params.src_dataset=="HMDB" and params.tgt_dataset=="UCF":
            source_dataset = VideoDataset_UCFHMDB(csv_file='./video_splits/hmdb51_train_hmdb_ucf.csv', dataset_name='hmdb', transform=None, base_dir = params.base_dir)
            target_dataset = VideoDataset_UCFHMDB(csv_file='./video_splits/ucf101_train_hmdb_ucf.csv', dataset_name='ucf', transform=None, base_dir = params.base_dir)
            target_dataset_eval = VideoDataset_UCFHMDB(csv_file='./video_splits/ucf101_val_hmdb_ucf.csv', dataset_name='ucf', transform=None, base_dir = params.base_dir, is_test=True)

    elif params.dataset_name=="Jester":
        source_dataset = VideoDataset_Jester(csv_file='./video_splits/jester_source_train.csv', base_dir = params.base_dir, transform=None)
        target_dataset = VideoDataset_Jester(csv_file='./video_splits/jester_target_train.csv', base_dir = params.base_dir, transform=None)
        target_dataset_eval = VideoDataset_Jester(csv_file='./video_splits/jester_source_val.csv', base_dir = params.base_dir, transform=None, is_test=True)

    elif params.dataset_name=="Epic-Kitchens":
        if params.src_dataset=="D1":
            source_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D1_train.pkl', transform=None, base_dir=params.base_dir)
        elif params.src_dataset=="D2":
            source_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D2_train.pkl', transform=None, base_dir=params.base_dir)
        elif params.src_dataset=="D3":
            source_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D3_train.pkl', transform=None, base_dir=params.base_dir)
        if params.tgt_dataset=="D1":
            target_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D1_train.pkl', transform=None, base_dir=params.base_dir)
            target_dataset_eval = VideoDataset_EpicKitchens(csv_file='./video_splits/D1_test.pkl', transform=None, base_dir=params.base_dir, is_test=True)
        elif params.tgt_dataset=="D2":
            target_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D2_train.pkl', transform=None, base_dir=params.base_dir)
            target_dataset_eval = VideoDataset_EpicKitchens(csv_file='./video_splits/D2_test.pkl', transform=None, base_dir=params.base_dir, is_test=True)
        elif params.tgt_dataset=="D3":
            target_dataset = VideoDataset_EpicKitchens(csv_file='./video_splits/D3_train.pkl', transform=None, base_dir=params.base_dir)
            target_dataset_eval = VideoDataset_EpicKitchens(csv_file='./video_splits/D3_test.pkl', transform=None, base_dir=params.base_dir, is_test=True)


    source_dataloader = DataLoader(source_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_segments)
    target_dataloader = DataLoader(target_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_segments)
    target_dataloader_eval = DataLoader(target_dataset_eval, batch_size=params.batch_size, shuffle=False, num_workers=params.num_segments)

    graph_model = Graph_Model(dataset_name=params.dataset_name)
    graph_model.cuda()

    print("=== Training started for CoMix ===")
    print('TemporalGraph:')
    print(graph_model)

    graph_model = train_comix(graph_model, source_dataloader, target_dataloader, target_dataloader_eval, num_iterations=params.num_iter_adapt)
