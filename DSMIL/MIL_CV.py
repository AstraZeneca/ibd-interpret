import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import scipy

torch.manual_seed(0)

import random
random.seed(0)

np.random.seed(0)

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats

def compute_mean_sd(data):
    m, se = np.mean(data), np.std(data, ddof=1) / len(data)
    return m, se

def get_CV_dataset(cv_splits, dataset, fold_num):

    # get the training slides and labels for this fold
    train_slides = pd.read_csv( os.path.join(cv_splits, str(fold_num), f"X_train_fold_{fold_num}.csv") ).iloc[:,0].to_list()
    train_labels = pd.read_csv( os.path.join(cv_splits, str(fold_num), f"y_train_fold_{fold_num}.csv") ).iloc[:,0].to_list()

    train_path = pd.DataFrame( {'slides' : train_slides, 'label' : train_labels} )
    # if a slide has no patches generated, there will not be a csv for that slide in the dataset,
    # so filter on the slides that do have embeddings
    bag_embeddings = os.listdir(os.path.join('datasets', dataset))
    bag_embeddings = [slide.split(".")[0] for slide in bag_embeddings]

    train_path = train_path.loc[ train_path["slides"].isin(bag_embeddings) ]

    # create the path to embeddings
    train_full_path = [os.path.join('datasets', dataset, slide_name+'.csv') for slide_name in train_path["slides"].to_list()]

    train_path["slides"] = train_full_path

    # get the training slides and labels for this fold
    test_slides = pd.read_csv( os.path.join(cv_splits, str(fold_num), f"X_test_fold_{fold_num}.csv") ).iloc[:,0].to_list()
    test_labels = pd.read_csv( os.path.join(cv_splits, str(fold_num), f"y_test_fold_{fold_num}.csv") ).iloc[:,0].to_list()

    test_path = pd.DataFrame( {'slides' : test_slides, 'label' : test_labels} )

    # if a slide has no patches generated, there will not be a csv for that slide in the dataset,
    # so filter on the slides that do have embeddings
    bag_embeddings = os.listdir(os.path.join('datasets', dataset))
    bag_embeddings = [slide.split(".")[0] for slide in bag_embeddings]

    test_path = test_path.loc[ test_path["slides"].isin(bag_embeddings) ]

    # create the path to embeddings
    test_full_path = [os.path.join('datasets', dataset, slide_name+'.csv') for slide_name in test_path["slides"].to_list()]

    test_path["slides"] = test_full_path

    return train_path, test_path

def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(train_df.iloc[i], args)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = Variable(Tensor([label]))
        bag_feats = Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            #test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)

    y_pred_proba = np.copy(test_predictions)

    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    
    y_true = np.copy(test_labels)
    print("test labels:", test_labels)
    print("test predictions:", test_predictions)
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, y_true, y_pred_proba

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Path to where all embeddings are stored')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument("--cv_splits", type=str, help="path to cross validation splits")
    parser.add_argument("--model_code", type=str, default=None, help="which class of model is running")
    parser.add_argument("--save_predictions", type=str, default=None, help="where to save model predictions")
    parser.add_argument("--results_dir", type=str, help="path to where MIL results are stored")
    parser.add_argument("--save_name", type=str, help="name of results file")
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    num_cv_folds = len(os.listdir(args.cv_splits))
    results_path = os.path.join(args.results_dir, args.save_name + '_results.txt')

    # open results file
    with open( results_path, 'w') as f:
        f.write("MIL results\n")

    # for storing AUCs for each of the CV runs
    all_AUCs = {}

    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"), str(datetime.datetime.now().hour) + "-" + str(datetime.datetime.now().minute))
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(num_cv_folds):

        # get the current fold
        train_path, test_path = get_CV_dataset(args.cv_splits, args.dataset, i)
        
        best_score = 0
        #save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"), str(datetime.datetime.now().hour) + "-" + str(datetime.datetime.now().minute), str(i))
        
        run = len(glob.glob(os.path.join(save_path, '*.pth')))
        
        best_auc = 0

        # re init every fold
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        if args.model == 'dsmil':
            state_dict_weights = torch.load('init.pth')
            try:
                milnet.load_state_dict(state_dict_weights, strict=False)
            except:
                del state_dict_weights['b_classifier.v.1.weight']
                del state_dict_weights['b_classifier.v.1.bias']
                milnet.load_state_dict(state_dict_weights, strict=False)
        criterion = nn.BCEWithLogitsLoss()
        
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        for epoch in range(1, args.num_epochs):
            
            train_path = shuffle(train_path).reset_index(drop=True)
            test_path = shuffle(test_path).reset_index(drop=True)
            train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
            test_loss_bag, avg_score, aucs, thresholds_optimal, y_true, y_pred_proba = test(test_path, milnet, criterion, optimizer, args)

            if args.dataset=='TCGA-lung':
                print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                    (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
            else:
                print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                    (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))

            scheduler.step()
            current_score = (sum(aucs) + avg_score)/2

            if np.mean(aucs) > best_auc:
                best_auc = np.mean(aucs)
                threshold_save = thresholds_optimal
                save_name = os.path.join(save_path, args.save_name + f"_weights_{i}.pth")
                torch.save(milnet.state_dict(), save_name)

                # save the best model's test set predictions
                model_predictions = pd.DataFrame({"y_true" : y_true, "y_pred_proba": y_pred_proba})
                model_predictions.to_csv(os.path.join(args.save_predictions, args.model_code, args.model_code + f"-{i}.csv"), index=False)

                if args.dataset=='TCGA-lung':
                    print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
                else:
                    print('Best model saved at: ' + save_name)
                    print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            
        # write the best AUC and threshold to file
        with open( results_path, 'a') as f:
            f.write(f"Fold {str(i)}: AUC: {str(best_auc)} optimal threshold: {str(threshold_save)} \n")

        # append AUC
        all_AUCs[best_auc] = [threshold_save, save_name]

    # calculate summary
    mean, sd = compute_mean_sd(list(all_AUCs.keys()))

    # write summary
    with open( results_path, 'a') as f:
        f.write(f"Performance summary: {str(mean)} ± {str(sd)}\n")
        f.write(f"best model is Fold {str(np.argmax(list(all_AUCs.keys())))} with AUC: {str(np.max(list(all_AUCs.keys())))} and threshold: {str( all_AUCs[ np.max(list(all_AUCs.keys()))][0] )} and saved at {str( all_AUCs[ np.max(list(all_AUCs.keys()))][1] )}")




if __name__ == '__main__':
    main()