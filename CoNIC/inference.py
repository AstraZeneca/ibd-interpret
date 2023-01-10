import sys
import logging
import cv2
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
import torch
from IPython.utils import io as IPyIO
from tqdm import tqdm
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from tiatoolbox.models.architecture.hovernet import HoVerNet as TIAHoVerNet
sys.path.append('../')
from net_desc import HoVerNetConic
from misc_util import cropping_center, recur_find_ext


def get_model(path_pretrained_model, num_types):
    pretrained_model = torch.load(path_pretrained_model)
    model = HoVerNetConic(num_types=num_types)
    model.load_state_dict(pretrained_model)
    return model


def get_predictor(model, num_loader_workers, batch_size):
    # Tile prediction
    predictor = SemanticSegmentor(
        model=model,
        num_loader_workers=num_loader_workers,
        batch_size=batch_size
    )
    return predictor


def get_ioconfig(resolution, patch_input_size):
    # Define the input/output configurations
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {'units': 'baseline', 'resolution': resolution}
        ],
        output_resolutions=[
            {'units': 'baseline', 'resolution': resolution},
            {'units': 'baseline', 'resolution': resolution},
            {'units': 'baseline', 'resolution': resolution}
        ],
        save_resolution={'units': 'baseline', 'resolution': resolution},
        patch_input_shape=[patch_input_size, patch_input_size],
        patch_output_shape=[patch_input_size, patch_input_size],
        stride_shape=[patch_input_size, patch_input_size]
    )
    return ioconfig


def predict_cells_per_patch(out_dir, predictor, infer_img_paths, ioconfig, pred_dir):
    # capture all the printing to avoid cluttering the console
    with IPyIO.capture_output() as captured:
        output_file = predictor.predict(
            infer_img_paths,
            masks=None,
            mode='tile',
            on_gpu=True,
            ioconfig=ioconfig,
            crash_on_exception=True,
            save_dir=f'{out_dir}/{pred_dir}/'
        )


def process_segmentation(model, np_map, hv_map, tp_map):
    # HoVerNet post-proc is coded at 0.25mpp so we resize
    np_map = cv2.resize(np_map, (0, 0), fx=2.0, fy=2.0)
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0)
    tp_map = cv2.resize(tp_map, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    inst_map = model._proc_np_hv(np_map[..., None], hv_map)
    inst_dict = TIAHoVerNet.get_instance_info(inst_map, tp_map)
    # Generating results match with the evaluation protocol
    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([[v['type']] * 3 for v in inst_dict.values()])
    type_map = overlay_prediction_contours(
        type_map, inst_dict,
        line_thickness=-1,
        inst_colours=inst_type_colours)
    pred_map = np.dstack([inst_map, type_map])
    # The result for evaluation is at 0.5mpp, so we scale back
    pred_map = cv2.resize(pred_map, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return pred_map


def process_composition(pred_map, num_types, central_crop_size=224):
    # Only consider the central 224x224 region,
    # as noted in the challenge description paper
    pred_map = cropping_center(pred_map, [central_crop_size, central_crop_size])
    inst_map = pred_map[..., 0]
    type_map = pred_map[..., 1]
    # ignore 0-th index as it is 0 i.e background
    uid_list = np.unique(inst_map)[1:]
    if len(uid_list) < 1:
        type_freqs = np.zeros(num_types)
        return type_freqs
    uid_types = [np.unique(type_map[inst_map == uid]) for uid in uid_list]
    type_freqs_ = np.unique(uid_types, return_counts=True)
    #!not all types exist within the same spatial location
    #!so we have to create a placeholder and put them there
    type_freqs = np.zeros(num_types)
    type_freqs[type_freqs_[0]] = type_freqs_[1]
    return type_freqs


def predict_all_patches(output_info, num_types, model):
    semantic_predictions = []
    composition_predictions = []
    for input_file, output_root in tqdm(output_info):
        #print(input_file)
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_map = np.load(f'{output_root}.raw.0.npy')
        hv_map = np.load(f'{output_root}.raw.1.npy')
        tp_map = np.load(f'{output_root}.raw.2.npy')

        pred_map = process_segmentation(model, np_map, hv_map, tp_map)
        type_freqs = process_composition(pred_map, num_types)
        semantic_predictions.append(pred_map)
        composition_predictions.append(type_freqs)

    return np.array(semantic_predictions), np.array(composition_predictions)


def save_predictions_to_df(out_dir, output_info, semantic_pred, composition_pred, type_names, segm_file, df_file):
    # Saving the results for segmentation
    np.save(f'{out_dir}/{segm_file}', semantic_pred)
    # Saving the results for composition prediction
    df = pd.DataFrame(composition_pred[:, 1:].astype(np.int32),)
    df.columns = type_names
    filenames = [f[0].split('/')[-1] for f in output_info]
    df['filename'] = filenames
    df.to_csv(f'{out_dir}/{df_file}', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicting six classes of cells.')
    parser.add_argument('--exp_name', default='', help='the name of the experiment.')
    parser.add_argument('--config_file', default='configs.yaml', help='config file name.')
    args = parser.parse_args()

    config_file_name = args.config_file
    stream = open(config_file_name, 'r')
    parameters = yaml.load(stream, Loader=yaml.Loader)
    exp_param = parameters[args.exp_name]

    SEED = 5
    type_names = ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]
    num_types = len(type_names) + 1

    imgs_folder = exp_param['imgs_input']
    pred_dir = f'{imgs_folder}_raw'

    logger = logging.getLogger()
    logger.disabled = True
    ioconfig = get_ioconfig(resolution=1.0, patch_input_size=exp_param['patch_input_size'])
    model = get_model(exp_param['path_model'], num_types)
    predictor = get_predictor(model, num_loader_workers=2, batch_size=6)
    infer_img_paths = recur_find_ext(f'{exp_param["path_output"]}/{imgs_folder}/', ['.png'])# change to the img path you need
    predict_cells_per_patch(exp_param['path_output'], predictor, infer_img_paths, ioconfig, pred_dir)

    output_file = f'{exp_param["path_output"]}/{pred_dir}/file_map.dat'
    output_info = joblib.load(output_file)
    semantic_pred, composition_pred = predict_all_patches(output_info, num_types, model)
    save_predictions_to_df(exp_param['path_output'],
                           output_info,
                           semantic_pred,
                           composition_pred,
                           type_names,
                           segm_file=f'{imgs_folder}_pred_seg.npy',
                           df_file=f'{imgs_folder}_pred_cell.csv')