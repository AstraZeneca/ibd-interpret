import sys
import torch
import argparse
from HIPT import HIPT
import os
from PIL import Image
import numpy as np
from HIPT_4K.hipt_model_utils import eval_transforms
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score


def run_hipt4k(imgs, model):
    """driver function for running hipt 4k on 4k regions from a WSI

    Args:
        imgs (list): list of PIL image
        model (HIPT): HIPT model

    Returns:
        torch.Tensor: 192 dim embedding for each 4k region
    """

    out = torch.zeros(len(imgs), 192)

    for i in range(len(imgs)):
        out[i] = model.hipt_4k(imgs[i].unsqueeze(0))

    return out

def check_prediction(prediction, ground_truth):
    return prediction == ground_truth

def f1_score(TP,TN,FP,FN):

    if TP == 0 and FP == 0:
        return 0.0

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)

    F1 = (2 * precision * recall) / (precision + recall)

    return F1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model256_path",  type=str, default=None, help="path to vit256")
    parser.add_argument("--model4k_path",   type=str, default=None, help="path to vit4k")
    parser.add_argument("--modelWSI_path",  type=str, default=None, help="path to vitWSI")
    parser.add_argument("--fold_num", type=int, default=0, help="which fold")
    parser.add_argument("--model_code", type=str, default=None, help="which class of model is running")
    parser.add_argument("--images",         type=str, default=None, help="path to data root")
    parser.add_argument("--ground_truth", type=str, default=None, help="path to ground truth")
    parser.add_argument("--save_predictions", type=str, default=None, help="where to save model predictions")
    parser.add_argument("--class_name", type=str, default=None, help="name of positive class")

    args = parser.parse_args()

    device=torch.device("cuda:0")

    # load in HIPT with 3 pretrained vits
    print("initialising HIPT model")

    fold_path = os.path.join(args.modelWSI_path, f"s_{args.fold_num}_checkpoint.pt")
    threshold = pd.read_csv(os.path.join(args.modelWSI_path, "summary_partial_0_5.csv")).iloc[args.fold_num, 3]

    model = HIPT(args.model256_path, args.model4k_path, fold_path, device=torch.device("cuda:0"))
    model.eval()
    model.to(device)

    slides = os.listdir(args.images)

    ground_truth_df = pd.read_csv(args.ground_truth)

    # for saving the model predictions
    y_trues = []
    y_pred_probas = []

    TP, TN, FP, FN = 0, 0, 0, 0

    for slide_idx in range(len(slides)):

        imgs = glob.glob(os.path.join(args.images, slides[slide_idx], "*.png"))
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs = [eval_transforms()(img) for img in imgs]

        features4k = run_hipt4k(imgs, model)
        features4k = features4k.to(device)

        y_pred_proba, y_pred = model(features4k)

        y_pred_proba = y_pred_proba.cpu().detach().numpy()[0][1]
        y_pred = y_pred.item()

        # check if prediction is correct, compute metrics F1 and AUROC
        try:
            ground_truth = int( ground_truth_df[ ground_truth_df["IMAGE_VSI"] == slides[slide_idx].split("/")[-1]+".svs" ]["label"].values[0] )
            print(ground_truth)
            y_pred_probas.append(y_pred_proba)
            y_trues.append(ground_truth)
            ground_truth_found = True
        except:
            ground_truth_found = False
            print("could not find ground truth for: ", slides[slide_idx])

        
        # if ground truth is found, we can print whether prediction was correct
        if ground_truth_found:

            # if the prediction score is higher than the threshold for this class, it is detected
            if y_pred_proba > threshold:

                prediction = 1

                prediction_status = check_prediction(prediction, ground_truth)

                if prediction_status:

                    print(slides[slide_idx]+ ' is CORRECTLY PREDICTED as: ' + args.class_name)
                    TP +=1

                else:
                    print(slides[slide_idx]+ ' is INCORRECTLY PREDICTED as: ' + args.class_name)
                    FP +=1

            else:

                prediction = 0

                prediction_status = check_prediction(prediction, ground_truth)

                if prediction_status:

                    print(slides[slide_idx]+ ' is CORRECTLY PREDICTED as: NOT ' + args.class_name)
                    TN += 1

                else:
                    print(slides[slide_idx]+ ' is INCORRECTLY PREDICTED as: NOT ' + args.class_name)
                    FN += 1

        else:
            # if no ground truth is found, we just print the prediction

            if y_pred_proba > threshold:
                
                print(slides[slide_idx]+ ' is predicted as: ' + args.class_name)

            else:

                print(slides[slide_idx]+ ' is predicted as: NOT ' + args.class_name)


    # save the model prediction for plotting ROC curves
    model_predictions = pd.DataFrame({"y_true" : y_trues, "y_pred_proba": y_pred_probas})
    model_predictions.to_csv(os.path.join(args.save_predictions, args.model_code, args.model_code + f"-{args.fold_num}.csv"), index=False)

    # print summary metrics
    print(f"F1 score: {f1_score(TP,TN,FP,FN)}")
    print(f"AUROC: {roc_auc_score(np.array(y_trues), np.array(y_pred_probas))}")


if __name__ == '__main__':
    main()