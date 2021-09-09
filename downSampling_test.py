import os

import numpy as np
import torch

import resnet
import spatial_transforms
from downsampling_arg import parse_args
from dataset import DAD
from dataset_test import DAD_Test
from model import generate_model
from downsampling_temporal_transforms import TemporalSequentialCrop
from downsampling_test import get_normal_vector, cal_score_single, get_normal_vector_head
from utils import evaluate, get_score

args = parse_args()

args.root_path = ''
args.save = args.root_path

args.n_split_ratio = 1.
args.a_split_ratio = 1.

args.n_train_batch_size = 40  # 3
args.a_train_batch_size = 40  # 25

args.shortcut_type = 'A'
args.n_scales = 5

if not os.path.exists(args.checkpoint_folder):
    os.makedirs(args.checkpoint_folder)
if not os.path.exists(args.log_folder):
    os.makedirs(args.log_folder)
if not os.path.exists(args.normvec_folder):
    os.makedirs(args.normvec_folder)
if not os.path.exists(args.score_folder):
    os.makedirs(args.score_folder)
torch.manual_seed(args.manual_seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.manual_seed)
if args.nesterov:
    dampening = 0
else:
    dampening = args.dampening
args.scales = [args.initial_scales]
for i in range(1, args.n_scales):
    args.scales.append(args.scales[-1] * args.scale_step)
assert args.train_crop in ['random', 'corner', 'center']
if args.train_crop == 'random':
    crop_method = spatial_transforms.MultiScaleRandomCrop(args.scales, args.sample_size)
elif args.train_crop == 'corner':
    crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size)
elif args.train_crop == 'center':
    crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])

before_crop_duration = int(args.sample_duration * args.downsample)
temporal_transform = TemporalSequentialCrop(before_crop_duration, args.downsample)

if not os.path.exists(args.normvec_folder):
    os.makedirs(args.normvec_folder)
score_folder = './score/'
if not os.path.exists(score_folder):
    os.makedirs(score_folder)
args.pre_train_model = False

model_front_d = generate_model(args)
resume_path_front_d = './checkpoints/best_model_' + args.model_type + '_top_IR.pth'
resume_checkpoint_front_d = torch.load(resume_path_front_d)
model_front_d.load_state_dict(resume_checkpoint_front_d['state_dict'])
model_front_d.eval()

model_front_d_head = resnet.ProjectionHead(args.feature_dim, args.model_depth)
resume_path_front_d_head = './checkpoints/best_model_' + args.model_type + '_front_depth_head.pth'
resume_checkpoint_front_d_head = torch.load(resume_path_front_d_head)
model_front_d_head.load_state_dict(resume_checkpoint_front_d_head['state_dict'])
model_front_d_head.cuda()
model_front_d_head.eval()

val_spatial_transform = spatial_transforms.Compose([
    spatial_transforms.Scale(args.sample_size),
    spatial_transforms.CenterCrop(args.sample_size),
    spatial_transforms.ToTensor(args.norm_value),
    spatial_transforms.Normalize([0], [1]),
])

print("========================================Loading Test Data========================================")
test_data_front_d = DAD_Test(root_path=args.root_path,
                             subset='validation',
                             view='top_IR',
                             sample_duration=before_crop_duration,
                             type=None,
                             spatial_transform=val_spatial_transform,
                             temporal_transform=temporal_transform
                             )

test_loader_front_d = torch.utils.data.DataLoader(
    test_data_front_d,
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=args.n_threads,
    pin_memory=True,
)
num_val_data_front_d = test_data_front_d.__len__()
print('Front depth view is done')

print("==========================================Loading Normal Data==========================================")
training_normal_data_front_d = DAD(root_path=args.root_path,
                                   subset='train',
                                   view='top_IR',
                                   sample_duration=before_crop_duration,
                                   type='normal',
                                   spatial_transform=val_spatial_transform,
                                   temporal_transform=temporal_transform
                                   )

training_normal_size = int(len(training_normal_data_front_d) * args.n_split_ratio)
training_normal_data_front_d = torch.utils.data.Subset(training_normal_data_front_d,
                                                       np.arange(training_normal_size))

train_normal_loader_for_test_front_d = torch.utils.data.DataLoader(
    training_normal_data_front_d,
    batch_size=args.cal_vec_batch_size,
    shuffle=True,
    num_workers=args.n_threads,
    pin_memory=True,
)
print(f'Front depth view is done (size: {len(training_normal_data_front_d)})')

print(
    "============================================START EVALUATING============================================")
# normal_vec_front_d = get_normal_vector(model_front_d, train_normal_loader_for_test_front_d,
#                                        args.cal_vec_batch_size,
#                                        args.feature_dim,
#                                        args.use_cuda)

normal_vec_front_d = get_normal_vector_head(model_front_d,
                                            model_front_d_head,
                                            train_normal_loader_for_test_front_d,
                                            args.cal_vec_batch_size,
                                            args.feature_dim,
                                            args.use_cuda)
np.save(os.path.join(args.normvec_folder, 'normal_vec_front_d.npy'), normal_vec_front_d.cpu().numpy())

np.save(os.path.join(args.normvec_folder, 'normal_vec_front_d.npy'), normal_vec_front_d.cpu().numpy())
normal_vec_front_d = torch.from_numpy(np.load(os.path.join(args.normvec_folder, 'normal_vec_top_ir.npy'))).cuda()


cal_score_single(model_front_d,
                 normal_vec_front_d,
                 test_loader_front_d,
                 score_folder,
                 args.use_cuda,
                 'top_d')

hashmap = {'top_d': 'Top(D)',
           'top_ir': 'Top(IR)',
           'fusion_top': 'Top(DIR)',
           'front_d': 'Front(D)',
           'front_ir': 'Front(IR)',
           'fusion_front': 'Front(DIR)',
           'fusion_d': 'Fusion(D)',
           'fusion_ir': 'Fusion(IR)',
           'fusion_all': 'Fusion(DIR)'
           }

mode = 'top_ir'

score = get_score(score_folder, mode)

gt = np.load(os.path.join(score_folder, 'label_' + mode + '.npy'))

best_acc, best_threshold, AUC = evaluate(score, gt, False)
print(
    f'Mode: {mode}:      Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
