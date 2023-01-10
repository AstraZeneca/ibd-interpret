import numpy as np
import sys
import torch
import torch.nn.functional as F
from utils.utils import *
import os
import torch.nn.functional as F
from datasets.dataset_generic import save_splits
#from models.model_dsmil import *
#from models.model_mil import MIL_fc, MIL_fc_mc
#from models.model_dgcn import DeepGraphConv
from models.model_clam import CLAM_MB, CLAM_SB
#from models.model_cluster import MIL_Cluster_FC
from models.model_hierarchical_mil import HIPT_None_FC, HIPT_LGP_FC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import auc as calc_auc
import matplotlib.pyplot as plt
import pandas as pd

#from utils.gpu_utils import gpu_profile, print_gpu_mem
os.environ['GPU_DEBUG']='0'

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]

        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def plot_curves(train_losses, train_errors, test_errors, test_aucs, args, fold_num):

    fig, axs = plt.subplots(4, 1, dpi=150, figsize=(10,10))

    axs[0].plot([x for x in range(len(train_losses))], train_losses)
    axs[0].set_title("train loss")
    axs[0].set_xticks(list(range(len(train_losses))), [x+1 for x in range(len(train_losses))])

    axs[1].plot([x for x in range(len(train_losses))], train_errors)
    axs[1].set_title("train error")
    axs[1].set_xticks(list(range(len(train_losses))), [x+1 for x in range(len(train_losses))])

    axs[2].plot([x for x in range(len(test_errors))], test_errors)
    axs[2].set_title("test error")
    axs[2].set_xticks(list(range(len(train_losses))), [x+1 for x in range(len(train_losses))])

    axs[3].plot([x for x in range(len(test_aucs))], test_aucs, label="test auc")
    axs[3].set_title("test auc")
    axs[3].set_xticks(list(range(len(train_losses))), [x+1 for x in range(len(train_losses))])

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, f"metrics_{str(fold_num)}.png"))
    plt.close()


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, test_split = datasets
    save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    #print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {'path_input_dim': args.path_input_dim, "dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif 'hipt' in args.model_type:
        if args.model_type == 'hipt_n':
            model = HIPT_None_FC(**model_dict)
        elif args.model_type == 'hipt_lgp':
            model = HIPT_LGP_FC(**model_dict, freeze_4k=args.freeze_4k, pretrain_4k=args.pretrain_4k, freeze_WSI=args.freeze_WSI, pretrain_WSI=args.freeze_WSI)
    elif args.model_type == 'dsmil':
        i_classifier = FCLayer(in_size=args.path_input_dim, out_size=model_dict['n_classes'])
        b_classifier = BClassifier(input_size=args.path_input_dim, output_class=model_dict['n_classes'], dropout_v=0.0)
        model = MILNet(i_classifier, b_classifier)
    elif args.model_type == 'dgcn':
        model_dict = {'path_input_dim': args.path_input_dim}
        model = DeepGraphConv(num_features=model_dict['path_input_dim'], n_classes=args.n_classes)
    elif args.model_type == 'mi_fcn':
        model = MIL_Cluster_FC(path_input_dim=args.path_input_dim, n_classes=args.n_classes)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    #val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    # early stopping parameters
    patience = 0
    max_patience = 5
    best_test_auc = 0
    global_optimal_threshold = 0

    train_losses, train_errors, test_errors, test_aucs = [], [], [], []

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, dropinput=args.dropinput)
            
            #stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
            # early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loss, train_error, test_error, test_auc, optimal_thresh, y_true, y_pred_proba = train_loop(args, epoch, model, train_loader, test_loader, optimizer, args.n_classes, writer, loss_fn)
            train_losses.append(train_loss)
            train_errors.append(train_error)
            test_errors.append(test_error)
            test_aucs.append(test_auc)

            # whenever there is a new best model, save auc, threshold, model predictions and model
            if test_auc >= best_test_auc:
                best_test_auc = test_auc
                global_optimal_threshold = optimal_thresh
                patience = 0
                # save the model prediction for plotting ROC curves
                model_predictions = pd.DataFrame({"y_true" : y_true, "y_pred_proba": y_pred_proba})
                model_predictions.to_csv(os.path.join(args.save_predictions, args.model_code, args.model_code + f"-{cur}.csv"), index=False)

                torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            else:
                patience += 1

            if patience > max_patience:
                print("early stopping...")
                print(f"best test auc achieved: {best_test_auc}")
                print(f"optimal threshold: {global_optimal_threshold}")
                break
            # stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                # early_stopping, writer, loss_fn, args.results_dir)
        
        #if stop: 
            #break

        plot_curves(train_losses, train_errors, test_errors, test_aucs, args, cur)
    
    print(f"best test auc achieved: {best_test_auc}")

    results_dict, test_error, test_auc, acc_logger, optimal_thresh, y_true, y_pred_proba = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, best_test_auc, 1-test_error, global_optimal_threshold


def train_loop(args, epoch, model, loader, test_loader, optimizer, n_classes, writer = None, loss_fn = None, gc=32):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')

    for batch_idx, batch in enumerate(loader):

        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
        else:
            data, label = batch
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            cluster_id = None
        
        logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
        #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss = loss / gc
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    results_dict, test_error, test_auc, acc_logger, optimal_thresh, y_true, y_pred_proba = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    return train_loss, train_error, test_error, test_auc, optimal_thresh, y_true, y_pred_proba

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
            else:
                data, label = batch
                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                cluster_id = None
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, dropinput=0.0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            data, label = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        if dropinput > 0:
            data = F.dropout(data, p=dropinput)

        logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device), cluster_id, label.to(device)
            else:
                data, label = batch
                data, label = data.to(device), label.to(device)
                cluster_id = None
            logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            data, label = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None

        #data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(data)

        # print("logits",logits)
        # print("yprob",Y_prob)
        # print("yhat", Y_hat)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    # print(all_labels)
    # print(all_probs)
    # print(np.argmax(all_probs, axis=1))

    np.save("labels", all_labels)
    np.save("probs", all_probs)

    if n_classes == 2:
        #auc = roc_auc_score(all_labels, all_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        fpr_opt, tpr_opt, optimal_threshold = optimal_thresh(fpr, tpr, thresholds)
        auc = calc_auc(fpr, tpr)
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, thresholds = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])

                fpr_opt, tpr_opt, optimal_threshold = optimal_thresh(fpr, tpr, thresholds)

                # print(thresholds)
                # print(fpr,tpr)
                plt.figure()
                plt.plot(fpr,tpr)
                plt.savefig("roc.png")
                plt.close()
                # print(fpr, tpr)
                # print(binary_labels[:, class_idx])
                # print(all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        # print(aucs)
        auc = np.nanmean(np.array(aucs))

    print("confusion matrix")
    print(confusion_matrix(all_labels, np.argmax(all_probs, axis=1)))


    return patient_results, test_error, auc, acc_logger, optimal_threshold, all_labels, all_probs[:,1]
