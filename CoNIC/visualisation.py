import os
import sys
import cv2
import glob
import yaml
import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.models.architecture.hovernet import HoVerNet as TIAHoVerNet

from patch_gen import open_slide_full

sys.path.append('../')


def plot_three(img, overlaid_true, overlaid_pred):
    plt.subplot(1, 3, 1)
    plt.title('Original H&E')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(overlaid_true)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid_pred)
    plt.title('Prediction')
    plt.axis('off')
    plt.show()


def plot_two(img, overlaid_pred):
    plt.subplot(1, 2, 1)
    plt.title('Original H&E')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_pred)
    plt.title('Prediction')
    plt.axis('off')
    plt.show()


def get_perceptive_colors():
    PERCEPTIVE_COLORS = [(0, 0, 0), (255, 165, 0), (0, 255, 0),
                         (255, 0, 0), (0, 255, 255), (0, 0, 255),
                         (255, 255, 0)]
    return PERCEPTIVE_COLORS


def get_colors_cells(cell_type):
    min_cells = {
        'lymphocyte':
        {
            'red': 10,
            'orange': 5,
            'yellow': 2
        },
        'neutrophil':
        {
            'red': 0,
            'orange': 0,
            'yellow': 0
        },
        'plasma':
        {
            'red': 6,
            'orange': 4,
            'yellow': 2
        },
        'eosinophil':
        {
            'red': 1,
            'orange': 0,
            'yellow': 0
        },
        'connective':
        {
            'red': 35,
            'orange': 25,
            'yellow': 18
        },
        'epithelial':
        {
            'red': 35,
            'orange': 25,
            'yellow': 18
        }
    }
    return min_cells[cell_type]


def overlay_pred(img_path, idx, semantic_pred, path_save, save=True, show=False):

    PERCEPTIVE_COLORS = get_perceptive_colors()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inst_map = semantic_pred[idx][..., 0]
    type_map = semantic_pred[idx][..., 1]
    pred_inst_dict = TIAHoVerNet.get_instance_info(inst_map, type_map)

    inst_type_colours = np.array([PERCEPTIVE_COLORS[v['type']] for v in pred_inst_dict.values()])
    overlaid_pred = overlay_prediction_contours(
        img, pred_inst_dict,
        inst_colours=inst_type_colours,
        line_thickness=1
    )
    if save:
        filename = path_save + img_path.split('/')[-1]
        Image.fromarray(overlaid_pred).save(filename)
    if show:
        # plot_three(img, overlaid_true, overlaid_pred)
        plot_two(img, overlaid_pred)


def overlap_heatmap(image, patch, cell_type, patch_size_level_one, diff_to_level_one, coeff):
    start_x = int(patch['start_y'] // coeff)
    start_y = int(patch['start_x'] // coeff)
    patch_size = int(patch_size_level_one / diff_to_level_one)
    min_cells = get_colors_cells(cell_type)
    red = (255, 0, 0)
    orange = (255, 130, 55)
    yellow = (255, 255, 100)
    alpha = 0.2
    if patch[cell_type] > min_cells['yellow']:
        overlay = image.copy()
        x, y, w, h = start_y, start_x, patch_size, patch_size  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), yellow, -1)  # A filled rectangle
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if patch[cell_type] > min_cells['orange']:
        overlay = image.copy()
        x, y, w, h = start_y, start_x, patch_size, patch_size  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), orange, -1)  # A filled rectangle
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    if patch[cell_type] > min_cells['red']:
        overlay = image.copy()
        x, y, w, h = start_y, start_x, patch_size, patch_size  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), red, -1)  # A filled rectangle
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


def get_df_cells(out_dir, path_output, image_name):
    path_to_df = out_dir + f'/{path_output}_pred_cell.csv'
    print('path to dataframe', path_to_df)
    df = pd.read_csv(path_to_df)
    df['image_name'] = df['filename'].str.split('/').str[0].str.split('_').str[-3]
    df = df[df['image_name'] == image_name]
    df['start_x'] = df['filename'].str.split('/').str[0].str.split('_').str[-2].astype(int)
    df['start_y'] = df['filename'].str.split('/').str[0].str.split('_').str[-1].str.replace('.png', '').astype(int)
    return df


def get_all_images_names(out_dir, path_output):
    df = pd.read_csv(out_dir + f'/{path_output}_pred_cell.csv')
    df['image_name'] = df['filename'].str.split('/').str[0].str.split('_').str[-3]
    images_names = list(set(df['image_name']))
    return images_names


def make_heatmap(out_dir, path_to_image, imgs_input, image_name, path_heatmaps, cell_type,
                 level, diff_to_level_one, coeff, show=True, save=True):
    print('image name:', image_name)
    image = open_slide_full(path_to_image, level=level, show=False)
    image = np.asarray(image)[:, :, :3].astype(np.uint8)
    df_cells = get_df_cells(out_dir, imgs_input, image_name)
    print('dataframe shape for an image:', df_cells.shape)
    df_cells = df_cells[[cell_type, 'start_x', 'start_y']]
    list_cells_patches = df_cells.T.to_dict().values()
    patch_size_level_one = 256
    for patch in list_cells_patches:
        image = overlap_heatmap(image, patch, cell_type, patch_size_level_one, diff_to_level_one, coeff)
    if show:
        plt.imshow(image)
        plt.show()
    if save:
        heatmap_name = image_name + '_' + cell_type + '.png'
        Image.fromarray(image).save(path_heatmaps + heatmap_name)


def mkdir_if_not_exist(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        print("created folder: ", folder_name)


def make_heatmaps(wsi_path, imgs_paths, path_output, output_heatmaps, imgs_input, cell_types,
                  level, diff_to_level_one, coeff, ext, test=True):
    imgs_names = list(set([imgs_path.split('/')[-1].split('_')[1] for imgs_path in imgs_paths]))
    print('total images: ', len(imgs_names))
    mkdir_if_not_exist(output_heatmaps)
    if test:  # then only for one cell type and one WSI
        cell_types = [cell_types[1]]  # epithelial  lymphocyte  plasma  eosinophil  connective
        imgs_names = [imgs_names[0]]
    for cell_type in cell_types:
        for image_file in imgs_names:
            path_to_image = f'{wsi_path}/{image_file}{ext}'
            print('path to image:', path_to_image)
            make_heatmap(path_output, path_to_image, imgs_input, image_file, output_heatmaps,
                         cell_type, level, diff_to_level_one=diff_to_level_one, coeff=coeff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualising the predictions by overlaying and heatmapping.')
    parser.add_argument('--exp_name', default='',  help='the name of the experiment.')
    parser.add_argument('--config_file', default='configs.yaml', help='config file name.')

    args = parser.parse_args()
    config_file_name = args.config_file
    stream = open(config_file_name, 'r')
    parameters = yaml.load(stream, Loader=yaml.Loader)
    exp_param = parameters[args.exp_name]

    imgs_folder = exp_param['imgs_input']
    images_names = get_all_images_names(exp_param["path_output"], imgs_folder)
    print(images_names)

    semantic_pred = np.load(f'{exp_param["path_output"]}/{imgs_folder}_pred_seg.npy')
    output_file = f'{exp_param["path_output"]}/{imgs_folder}_raw/file_map.dat'
    output_info = joblib.load(output_file)
    imgs_paths = glob.glob(exp_param["path_output"] + f'/{imgs_folder}/*.png')
    print('total images:', len(imgs_paths))
    print('overlaying path:', exp_param["imgs_overlap"])

    if exp_param["get_overlap"]:
        for idx, img_path in enumerate(sorted(imgs_paths)):
            print('overlaying', img_path)
            overlay_pred(img_path, idx, semantic_pred, path_save=exp_param["imgs_overlap"], save=True, show=False)
    if exp_param["get_heatmaps"]:
        make_heatmaps(exp_param["wsi_path"],
                      imgs_paths,
                      exp_param["path_output"],
                      exp_param["output_heatmaps"],
                      exp_param["imgs_input"],
                      exp_param["cell_types"],
                      exp_param["level"],
                      exp_param["diff_to_level_one"],
                      exp_param["coeff"],
                      exp_param["ext"],
                      exp_param["test"])


