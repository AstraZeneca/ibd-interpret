import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
import matplotlib.pyplot as plt

import math
import openslide

torch.manual_seed(0)

import random
random.seed(0)

np.random.seed(0)

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        # the location of the patch is defined here
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class BagDatasetHistolab():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]


        # the location of the patch is defined here
        row_pos_wsi = int(img_name.split('.')[0].split('_')[3].split("-")[0])
        col_pos_wsi = int(img_name.split('.')[0].split('_')[3].split("-")[1])
        img_pos = np.asarray([row_pos_wsi, col_pos_wsi])
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset_histolab(args, csv_file_path):
    transformed_dataset = BagDatasetHistolab(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def check_prediction(prediction, ground_truth):
    return prediction == ground_truth


def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    colors = [ np.array([255,255,255]) for _ in range(args.num_classes) ]
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.'+args.patch_ext))
        if args.patch_origin == 'DZ':
            dataloader, bag_size = bag_dataset(args, csv_file_path)
        elif args.patch_origin == 'histolab':
            dataloader, bag_size = bag_dataset_histolab(args, csv_file_path)
        else:
            raise NotImplementedError
        with torch.no_grad(): 
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()

            if len(bag_prediction.shape)==0 or len(bag_prediction.shape)==1:
                bag_prediction = np.atleast_1d(bag_prediction)
            
            # read in the ground truth
            ground_truth_df = pd.read_csv(args.gt_path, usecols=["IMAGE_VSI", "label"])

            image_vsi = [name.split(".")[0] for name in ground_truth_df["IMAGE_VSI"].to_list()]

            ground_truth_df["IMAGE_VSI"] = image_vsi

            try:
                # try and extract the ground truth for this slide
                ground_truth = int( ground_truth_df[ ground_truth_df["IMAGE_VSI"] == bags_list[i].split("/")[-1] ]["label"].values[0] )
                ground_truth_found = True
            except Exception as e:
                # if we cannot find ground truth
                ground_truth_found = False
                print("could not find ground truth for: ", bags_list[i])

            num_pos_classes = 0
            for c in range(args.num_classes):

                # get the attention scores for this class
                attentions = A[:, c].cpu().numpy()

                # if ground truth is found, we can print whether prediction was correct
                if ground_truth_found:

                    # if the prediction score is higher than the threshold for this class, it is detected
                    if bag_prediction[c] >= args.thres[c]:
                        num_pos_classes += 1

                        prediction = 1

                        prediction_status = check_prediction(prediction, ground_truth)

                        if prediction_status:

                            print(bags_list[i]+ ' is CORRECTLY PREDICTED as: ' + args.class_name[c])

                        else:
                            print(bags_list[i]+ ' is INCORRECTLY PREDICTED as: ' + args.class_name[c])

                        colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])

                    else:
                        num_pos_classes += 1

                        prediction = 0

                        prediction_status = check_prediction(prediction, ground_truth)

                        if prediction_status:

                            print(bags_list[i]+ ' is CORRECTLY PREDICTED as: NOT ' + args.class_name[c])

                        else:
                            print(bags_list[i]+ ' is INCORRECTLY PREDICTED as: NOT ' + args.class_name[c])

                        colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])
                else:
                    # if no ground truth is found, we just print the prediction

                    if bag_prediction[c] >= args.thres[c]:
                        
                        num_pos_classes += 1

                        print(bags_list[i]+ ' is predicted as: ' + args.class_name[c])

                        colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])

                    else:
                        num_pos_classes += 1

                        print(bags_list[i]+ ' is predicted as: NOT ' + args.class_name[c])

                        colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])

                # rescale scores between 0 and 1
                colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))

                slide_name = bags_list[i].split(os.sep)[-1]

                # read in the WSI
                slide = openslide.OpenSlide(os.path.join(args.wsi_path, slide_name+args.format))
                # set the level
                level = args.level

                # width, height (cols, rows)
                # get the dimensions of the slide at the current level
                level_dimensions = slide.level_dimensions[level]

                # get the dimensions of the WSI at level 0
                dimensions = slide.dimensions

                # define the size of patches
                patch_size = 224

                # create a large png of the slide
                png = slide.get_thumbnail((dimensions[0] // args.scale, dimensions[1] // args.scale))


                if args.patch_origin == "DZ":
                    color_map = np.zeros((np.amax(pos_arr, 0)[1]+1, np.amax(pos_arr, 0)[0]+1, 3))

                    for k, pos in enumerate(pos_arr):
                        color_map[pos[1], pos[0]] = colored_tiles[k]

                    # find the number of rows and columns in the WSI based on patches
                    c_max = math.floor(level_dimensions[0] / patch_size)
                    r_max = math.floor(level_dimensions[1] / patch_size)

                    # find the number of rows and columns in the attention map
                    c_a = max(pos_arr[:,0])
                    r_a = max(pos_arr[:,1])

                    # find the difference
                    c_difference = c_max - c_a
                    r_difference = r_max - r_a

                    # greyscale attention map
                    color_map = color_map[:,:,0]
                    if args.threshold > 0.0:
                        color_map[color_map < args.threshold] = 0
                    # pad with the differences
                    color_map = np.pad(color_map, [(0,r_difference), (0,c_difference)], mode='constant', constant_values=0)
                    greyscale_map = color_map.copy()
                    # scale up for detail
                    color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
                    # save attention map
                    plt.imsave(os.path.join(args.map_path, "output", slide_name+"_"+args.class_name[c]+'.png'), color_map, cmap='jet')
                    
                    # overlay the attention map on the WSI png
                    att_map = Image.open(os.path.join(args.map_path, "output", slide_name+"_"+args.class_name[c]+'.png'))
                    att_map = att_map.resize((png.size[0], png.size[1]), Image.Resampling.LANCZOS)

                    # convert to RGBA for blending
                    png = png.convert('RGBA')
                    att_map = att_map.convert('RGBA')

                    fused = Image.blend(png,att_map, 0.3)
                    fused.save(os.path.join(args.map_path, "fused", slide_name+"_"+args.class_name[c]+'.png'),"PNG")
                
                elif args.patch_origin == "histolab":

                    # get dims of png
                    png_rows = dimensions[1] // args.scale
                    png_cols = dimensions[0] // args.scale
                    
                    # get dims of downsized patch
                    png_patch_size = (patch_size * (args.level*2)) // args.scale if args.level != 0 else patch_size // args.scale
                    
                    # create an empty attention map the size of the png
                    att_map = np.zeros((png_rows, png_cols, 3))

                    # fill in attention map
                    for k, pos in enumerate(pos_arr):
                        row = pos[1]//args.scale
                        col = pos[0]//args.scale
                        att_map[row:row+png_patch_size, col:col+png_patch_size] = colored_tiles[k]

                    att_map = att_map[:,:,0]

                    # save the attention map
                    plt.imsave(os.path.join(args.map_path, "output", slide_name+"_"+args.class_name[c]+'.png'), att_map, cmap='jet')

                    # overlay the attention map on the WSI png
                    att_map = Image.open(os.path.join(args.map_path, "output", slide_name+"_"+args.class_name[c]+'.png'))

                    # convert to RGBA for blending
                    png = png.convert('RGBA')
                    att_map = att_map.convert('RGBA')

                    fused = Image.blend(png,att_map, 0.3)
                    fused.save(os.path.join(args.map_path, "fused", slide_name+"_"+args.class_name[c]+'.png'),"PNG")


                else:
                    raise NotImplementedError

            if args.export_scores:
                df_scores = pd.DataFrame(exposure.rescale_intensity(colored_tiles, out_range=(0, 1))[:,0])
                pos_arr_str = [str(s) for s in pos_arr]
                df_scores['pos'] = pos_arr_str
                df_scores.columns = ["attention", 'pos']
                df_scores.sort_values(by="attention", ascending=False, inplace=True)
                df_scores.to_csv(os.path.join(args.map_path, "score", bags_list[i].split(os.sep)[-1]+'.csv'), index=False)
                
                
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres', nargs='+', type=float, default=[0.7371, 0.2752])
    parser.add_argument('--class_name', nargs='+', type=str, default=None)
    parser.add_argument('--embedder_weights', type=str, default='test/weights/embedder.pth')
    parser.add_argument('--aggregator_weights', type=str, default='test/weights/aggregator.pth')
    parser.add_argument('--bag_path', type=str, default='test/patches')
    parser.add_argument('--wsi_path', type=str, help='path to wsis')
    parser.add_argument('--patch_ext', type=str, default='jpeg')
    parser.add_argument('--format', type=str, default=".tiff", help="slide format")
    parser.add_argument('--scale', type=int, default=16, help="how much to downscale the size of the heatmap, must be positive")
    parser.add_argument('--map_path', type=str, default='test/output')
    parser.add_argument('--threshold', type=float, default=0.0, help="threshold for attention map")
    parser.add_argument('--patch_origin', type=str, default='DZ', choices=['DZ', 'histolab'], help="which method was used to generate patches")
    parser.add_argument('--feats_path', type=str, default=None, help="path to computed embeddings")
    parser.add_argument('--level', type=int, default=0, help='slide magnification level')
    parser.add_argument('--export_scores', type=int, default=0)
    parser.add_argument('--gt_path', type=str, default=None, help="path to the ground truth labels")
    args = parser.parse_args()
    
    print("loading weights")
    if args.embedder_weights == 'ImageNet':
        print('Use ImageNet features')
        resnet = models.resnet18(pretrained=True, norm_layer=nn.BatchNorm2d)
    else:
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    if args.embedder_weights !=  'ImageNet':
        state_dict_weights = torch.load(args.embedder_weights)
        new_state_dict = OrderedDict()
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)

    state_dict_weights = torch.load(args.aggregator_weights) 
    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)

    # bag list comes from the bag dataset - so it may find bags that have no patches extracted
    print("generating")
    bags_list = glob.glob(os.path.join(args.bag_path, '*'))
    bags_list = os.listdir(args.feats_path)
    bags_list = [name.split(".")[0] for name in bags_list]
    bags_list = [os.path.join(args.bag_path, name) for name in bags_list]

    bags_list = sorted(bags_list)
    
    os.makedirs(args.map_path, exist_ok=True)
    if args.export_scores:
        os.makedirs( os.path.join(args.map_path, "score"), exist_ok=True)
    if args.class_name == None:
        args.class_name = ['class {}'.format(c) for c in range(args.num_classes)]
    if len(args.thres) != args.num_classes:
        raise ValueError('Number of thresholds does not match classes.')
    test(args, bags_list, milnet)