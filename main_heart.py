def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
import argparse
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from sklearn import metrics 
from sklearn.preprocessing import Binarizer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import *
from torch.utils.tensorboard import SummaryWriter
from models.GAT_Custom import Net
from models.GAT_Custom import MyEnsemble
from data_loading import load_graph_data
#from utils.data_loading import load_graph_data_2
from utils.constants import *
import utils.utils as utils
import os
from torch_geometric.data import Data
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.metrics import plot_confusion_matrix
from datetime import datetime
import datetime
current_time = datetime.datetime.now()
print(current_time)

# config['num_of_epochs']
# #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]= f"{config["gpu"]}"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import datetime
start = time.time()

#
#writer = SummaryWriter()
#_312_LR_5e-4_MACRO_EPOCH_2000
total_tn = []
total_fp = []
total_fn = []
total_tp = []

total_accuracy = []
total_recall = []
total_precision = []
total_macro_f1 = []
total_auc_score = []

# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def get_main_loop(config, gat0, gat1, gat2, gat, sigmoid_cross_entropy_loss, optimizer, scheduler, patience_period, time_start):
    writer = SummaryWriter(comment=f"trial{config['trial_no']}_gpuNO{config['gpu']}_epochs{config['num_of_epochs']}_lr{config['lr']}_batchsize{config['batch_size']}")
    device = next(gat.parameters()).device  # fetch the device info from the model instead of passing it as a param

    def main_loop(phase, data_loader, epoch=0):
        global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT
        global total_val_tn, total_val_fp, total_val_fn, total_val_tp
        global total_val_accuracy, total_val_recall, total_val_precision, total_val_macro_f1, total_val_auc_score, total_val_loss

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            current_lr = get_lr(optimizer)
            print("current_lr : ", current_lr)
            gat.train()
        else:
            
            #reset and prepared validation score list               
            total_val_tn = []
            total_val_fp = []
            total_val_fn = []
            total_val_tp = []

            total_val_accuracy = []
            total_val_recall = []
            total_val_precision = []
            total_val_macro_f1 = []
            total_val_auc_score = []
            total_val_loss = []
            
            gat.eval()

            

        # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
        # We merge them into a single graph with 2 connected components, that's the main idea. After that
        # the implementation #3 is agnostic to the fact that those are multiple and not a single graph!
        # for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
            # Push the batch onto GPU - note PPI is to big to load the whole dataset into a normal GPU
            # it takes almost 8 GBs of VRAM to train it on a GPU
            #edge_index2 = edge_index.transpose(0, 1).contiguous().to(device)
#             print("len(data_loader)")
#             print(len(data_loader))

            indices0 = torch.tensor([13,14,15,16,17,18,22,23,24,25,26,27,28,29,30,31,32,33]).to(device)
            indices1 = torch.tensor([0,1,2,3,4,5,6,8,9]).to(device)
            indices2 = torch.tensor([7,10,11,12,19,20,21]).to(device)   
            
            node_features = node_features.to(device)
            node_features0 = torch.index_select(node_features, 1, indices0).to(device)
            node_features1 = torch.index_select(node_features, 1, indices1).to(device)
            node_features2 = torch.index_select(node_features, 1, indices2).to(device)
            
            gt_node_labels = torch.squeeze(gt_node_labels, 0).to(device)
            edge_index = edge_index.to(device)
             
#             edge_index = torch.squeeze(edge_index)
#             nodes_features = torch.squeeze(node_features)            
#             nodes_features = torch.squeeze(node_features)
#             edge_indexs = torch.squeeze(edge_index)            

            # I pack data into tuples because GAT uses nn.Sequential which expects this format
            demo_graph = Data(x=node_features0, edge_index=edge_index)
            habit_graph = Data(x=node_features1, edge_index=edge_index)
            under_graph = Data(x=node_features2, edge_index=edge_index)
            
            # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
            # shape = (N, C) where N is the number of nodes in the batch and C is the number of classes (121 for PPI)
            # GAT imp #3 is agnostic to the fact that we actually have multiple graphs
            # (it sees a single graph with multiple connected components)
            
            nodes_unnormalized_scores = gat(demo_graph, habit_graph, under_graph)
            scaler = torch.cuda.amp.GradScaler()
            loss = sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)
            

            if phase == LoopPhase.TRAIN:
                
                optimizer.zero_grad()
                scaler.scale(loss).backward() # scaled gradients
                scaler.step(optimizer)   # unscaled gradients
                scaler.update()               # scaled update      
                scheduler.step()  

            # Calculate the main metric - micro F1

            # Convert unnormalized scores into predictions. Explanation:
            # If the unnormalized score is bigger than 0 that means that sigmoid would have a value higher than 0.5
            # (by sigmoid's definition) and thus we have predicted 1 for that label otherwise we have predicted 0.
            #pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()
            pred = torch.round(torch.sigmoid(nodes_unnormalized_scores)).cpu().detach().numpy()
            
            # Binarizer
            thresholds = 0.5
            
            binarizer = Binarizer(threshold = thresholds)
            
            binarizer = Binarizer(threshold = thresholds).fit(pred)
            pred_bi = binarizer.transform(pred)
            
            gt = gt_node_labels.cpu().numpy()
            
            #4 score function 
            accuracy = accuracy_score(gt, pred)
            recall = recall_score(gt, pred, average='macro', zero_division='warn') #micro pos_label = 1
            precision = precision_score(gt, pred, average='macro', zero_division='warn')           
            macro_f1 = f1_score(gt, pred, zero_division='warn', average='macro')
            con_matrix = confusion_matrix(gt, pred, labels=[0, 1], sample_weight=None, normalize=None)
            #tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
                      
            
            fpr, tpr, thresholds = metrics.roc_curve(gt, pred, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)

            #global_step = len(data_loader) * epoch + batch_idx
#             import copy
#             global step_train
#             step_train = copy.copy(global_step)
            
            #print(lr_scheduler.get_lr())
            
            if phase == LoopPhase.TRAIN:
                global_step = len(data_loader) * epoch + batch_idx
                import copy
                global step_val
                step_val = copy.copy(global_step)                
                # Log metrics
                if config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), global_step)
                    writer.add_scalar('training_accuracy', accuracy, global_step)
                    writer.add_scalar('training_recall', recall, global_step)
                    writer.add_scalar('training_precision', precision, global_step)
                    writer.add_scalar('training_macro_f1', macro_f1, global_step)
                    writer.add_scalar('training_auc_score', auc_score, global_step)
                    

                # Log to console
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | accuracy={accuracy:.4f} | precision={precision:.4f} | recall={recall:.4f} | macro-F1={macro_f1} | auc={auc_score:.4f}.')

                # Save model checkpoint
                if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0 and batch_idx == 0:
                    #config['trial_no']
                    ckpt_model_name = f'gat_{config["dataset_name"]}_{config["trial_no"]}_ckpt_epoch_{epoch + 1}.pth'
                    config['test_perf'] = -1  # test perf not calculated yet, note: perf means main metric micro-F1 here
                    torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            
            elif phase == LoopPhase.VAL:               
                    
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    #append per validation iteration
#                     total_val_tn.append(tn)
#                     total_val_fp.append(fp)
#                     total_val_fn.append(fn)
#                     total_val_tp.append(tp)

                    total_val_loss.append(loss.item())
                    total_val_accuracy.append(accuracy)
                    total_val_recall.append(recall)
                    total_val_precision.append(precision)
                    total_val_macro_f1.append(macro_f1)
                    total_val_auc_score.append(auc_score)
                    
                    print("total val loss length PRINT :", len(total_val_loss))
                    #print(f'tn={tn} | fp={fp} | fn={fn} | tp={tp}.')
                    print(f'GAT validation: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | accuracy={accuracy:.4f} | precision={precision:.4f} | recall={recall:.4f} | val macro-F1={macro_f1} | auc={auc_score:.4f}.')


                
                # The "patience" logic - should we break out from the training loop? If either validation micro-F1
                # keeps going up or the val loss keeps going down we won't stop
                if macro_f1 > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_PERF = max(macro_f1, BEST_VAL_PERF)  # keep track of the best validation micro_f1 so far
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  # and the minimal loss
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best micro_f1
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= patience_period * len(data_loader):
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

        
            else:
                # Log to console
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    #print(f'tn={tn} | fp={fp} | fn={fn} | tp={tp}.')
                    print(f'GAT test: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f'batch={batch_idx + 1} | accuracy={accuracy:.4f} | precision={precision:.4f} | recall={recall:.4f} | test macro-F1={macro_f1:.4f} | auc={auc_score:.4f}.')
                    
                
#                     total_tn.append(tn)
#                     total_fp.append(fp)
#                     total_fn.append(fn)
#                     total_tp.append(tp)

                    total_accuracy.append(accuracy)
                    total_recall.append(recall)
                    total_precision.append(precision)
                    total_macro_f1.append(macro_f1)
                    total_auc_score.append(auc_score)
              

    return main_loop  # return the decorated function


def train_gat_lending(config):
    """
    Very similar to Cora's training script. The main differences are:
    1. Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    2. Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    3. Model architecture and hyperparams are a bit different (as reported in the GAT paper)

    """
    global BEST_VAL_PERF, BEST_VAL_LOSS
    
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{config['gpu']}"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    writer = SummaryWriter(comment=f"trial{config['trial_no']}_gpuNO{config['gpu']}_epochs{config['num_of_epochs']}_lr{config['lr']}_batchsize{config['batch_size']}") 
    # Checking whether you have a strong GPU. Since PPI training requires almost 8 GBs of VRAM
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    device = torch.device("cuda" if torch.cuda.is_available() and not config['force_cpu'] else "cpu")
    cpu_device = torch.device('cpu')
    # Step 1: prepare the data loaders
    
    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)
#     train_loader_len = len(data_loader_train)
#     val_loader_len = len(data_loader_val)
    
#    step_val = train_loader_len * (epoch+1) - 1 
#     print("="*30)
#     print(len(data_loader_test))
#     print("="*30)
    
    # Step 2: prepare the model
    gat0 = Net(
        nfeat=config['num_features_per_layer'][0][0],
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False
        # no need to store attentions, used only in playground.py for visualizations
    ).to(device)
    
    gat1 = Net(
        nfeat=config['num_features_per_layer'][0][1],
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False
        # no need to store attentions, used only in playground.py for visualizations
    ).to(device)
    
    gat2 = Net(
        nfeat=config['num_features_per_layer'][0][2],
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False
        # no need to store attentions, used only in playground.py for visualizations
    ).to(device) 
    
    gat = MyEnsemble(gat0, gat1, gat2, config['num_features_per_layer']).to(device)
    
    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000000000000], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001, last_epoch=-1)
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat0, gat1, gat2, gat,
        loss_fn,
        optimizer,
        scheduler,
        config['patience_period'],
        time.time())

    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                
                main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
                #list to array
#                 total_val_tn_final = np.array(total_val_tn)
#                 total_val_fp_final = np.array(total_val_fp)
#                 total_val_fn_final = np.array(total_val_fn)
#                 total_val_tp_final = np.array(total_val_tp)

                total_val_loss_final = np.array(total_val_loss)

                total_val_accuracy_final = np.array(total_val_accuracy)
                total_val_recall_final = np.array(total_val_recall)
                total_val_precision_final = np.array(total_val_precision)
                total_val_macro_f1_final = np.array(total_val_macro_f1)
                total_val_auc_score_final = np.array(total_val_auc_score) 

                print("="*40,"VALIDATION EPOCH EVALUATION METRICS","="*45)

                print(f'Total Val accuracy={np.mean(total_val_accuracy_final):.4f} | precision={np.mean(total_val_precision_final):.4f} | recall={np.mean(total_val_recall_final):.4f} | Val-F1 = {np.mean(total_val_macro_f1):.4f} | auc={np.mean(total_val_auc_score_final):.4f}.')

                #print(f'Total tn={np.sum(total_val_tn_final)} | fp={np.sum(total_val_fp_final)} | fn={np.sum(total_val_fn_final)} | tp={np.sum(total_val_tp_final)}.')

                #print(f'Avg tn={np.mean(total_val_tn_final)} | fp={np.mean(total_val_fp_final)} | fn={np.mean(total_val_fn_final)} | tp={np.mean(total_val_tp_final)}.')
                print("="*40,"END","="*50)
                # Log metrics
                if config['enable_tensorboard']:                  
                    #step = len(data_loader) * epoch 
                    writer.add_scalar('val_loss', np.mean(total_val_loss_final), step_val)
                    writer.add_scalar('val_accuracy', np.mean(total_val_accuracy_final), step_val)
                    writer.add_scalar('val_recall', np.mean(total_val_recall_final), step_val)
                    writer.add_scalar('val_precision', np.mean(total_val_precision_final), step_val)
                    writer.add_scalar('val_macro_f1', np.mean(total_val_macro_f1_final), step_val)
                    writer.add_scalar('val_auc_score', np.mean(total_val_auc_score_final), step_val)

            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break
                  # break out from the training loop             

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and micro-F1 on the test dataset. Friends don't let friends overfit to the test data. <3
    if config['should_test']:
        with torch.no_grad():
            
            main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)
            config['test_perf'] = np.mean(total_macro_f1)
        
#         total_tn_final = np.array(total_tn)
#         total_fp_final = np.array(total_fp)
#         total_fn_final = np.array(total_fn)
#         total_tp_final = np.array(total_tp)

            total_accuracy_final = np.array(total_accuracy)
            total_recall_final = np.array(total_recall)
            total_precision_final = np.array(total_precision)
            total_macro_f1_final = np.array(total_macro_f1)
            total_auc_score_final = np.array(total_auc_score)

            print("*"*40,"TEST EVALUATION METRICS","*"*40)

            print(f'Total Test accuracy={np.mean(total_accuracy_final):.4f} | precision={np.mean(total_precision_final):.4f} | recall={np.mean(total_recall_final):.4f} | Test-F1 = {np.mean(total_macro_f1):.4f} | auc={np.mean(total_auc_score_final):.4f}.')
        
#         print(f'Total tn={np.sum(total_tn)} | fp={np.sum(total_fp)} | fn={np.sum(total_fn)} | tp={np.sum(total_tp)}.')

#         print(f'Avg tn={np.mean(total_tn)} | fp={np.mean(total_fp)} | fn={np.mean(total_fn)} | tp={np.mean(total_tp)}.')
            print("*"*40,"END","*"*40,'********************')
            
#         print("="*30,"TOTAL","="*30)      
#         print(f'Total Test accuracy={(total_accuracy/len(data_loader_test)):.4f} | precision={(total_precision/len(data_loader_test)):.4f} | recall={(total_recall/len(data_loader_test)):.4f} | micro-F1 = {(total_micro_f1/len(data_loader_test))} | auc={(total_auc_score/len(data_loader_test)):.4f}.')
#         print(f'tn={total_tn} | fp={total_fp} | fn={total_fn} | tp={total_tp}.')
    
    else:
        config['test_perf'] = -1

    # Save the latest GAT in the binaries directory
    final_model_name = f'gat_{config["dataset_name"]}_{config["trial_no"]}.pth'
    
    torch.save(
        utils.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, final_model_name))
    
def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--trial_no", type=int, help="number of training trial", default=1)
    parser.add_argument("--gpu", type=int, help="number of gpu to use", default=1)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=20000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-4)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--force_cpu", action='store_true', help='use CPU if your GPU is too small (no by default)')

    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.LENDING.name)
    #parser.add_argument("--dataset_type", help='which batch dataset to use for training', type=str, default="2000_rescaled")
    parser.add_argument("--batch_size", type=int, help='number of graphs in a batch', default=1)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq (None for no logging)", default=1)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=10)
    args = parser.parse_args()

    # I'm leaving the hyperparam values as reported in the paper, but I experimented a bit and the comments suggest
    # how you can make GAT achieve an even higher micro-F1 or make it smaller
    gat_config = {
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [[18, 9, 7], 8, 8, 8],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0,  # dropout hurts the performance (best to keep it at 0)
        "layer_type": LayerType.IMP2  # IMP3 = GAT, IMP2 = GATv2 
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        
        training_config[arg] = getattr(args, arg)
        print(training_config[arg])
    training_config['lending_load_test_only'] = False  # load both train/val/test data loaders (don't change it)

    # Add additional config information
    training_config.update(gat_config)

    return training_config

if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_lending(get_training_args())
    
#time consuming
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print(times)

current_time = datetime.datetime.now()
print(current_time)