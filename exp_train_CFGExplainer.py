import sys
import os
from util.config import args
from util.models import GCN
from util.metrics import accuracy
from util.graphprocessor import YANCFG

import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from Explainer import ExplainerModule

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------
# explanation function
# ----------------------------


def train_CFGExplainer():
    """
    will run the training for explainer
    """
    
    # 1. load pre-trained GCN model
    model = GCN(input_dim=args.d, output_dim=args.c)
    model.load_weights(args.save_path + args.dataset)  # load the weights

    # 2. load graph data
    data_loader = YANCFG()
    train, info, num_samples = data_loader.load_yancfg_data(args.path, 'padded_train', args.malware_list)
    
    device = '/gpu:0'
    # 3. fit explainer
    # 3.1. initialize writer
    name = 'CFGExplainer_' + args.explainer_name + args.model_name_flag + args.dataset
    writer = None
    if args.writer_path is not None:
        writer = SummaryWriter(args.writer_path + name)

    # initilize the explainer model
    explainer, optimizer = None, None
    with tf.device(device):
        explainer = ExplainerModule(model=model, output_dim=args.c)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.elr)
    print('+ gcn model:', model)
    print('+ explainer model:', explainer)

    # running the training epochs
    for epoch in tqdm(range(args.eepochs), disable=args.disable_tqdm):

        # run a minibatch for each epoch
        losses, exp_outputs, labels = [], [], []
        train_batch = train.shuffle(args.batch_size).batch(args.batch_size)  # removed the batch_size from here, the code only works for batch training?
        for batch_id, ts_batch in enumerate(train_batch):

            with tf.device(device):
                batch_adjs, batch_feats, batch_labels, batch_ids, batch_mask = ts_batch
                # get the embedding results for the graphs from pre-trained GCN
                # does not need gradient computation
                batch_embs = model.getNodeEmb((batch_feats, batch_adjs), training=False)
                
                with tf.GradientTape() as tape:
                    pred, _ = explainer((batch_feats, batch_embs, batch_adjs, batch_mask))
                    # print(pred)
                    loss = explainer.loss(pred, batch_labels)

                losses.append(tf.concat(loss, axis=0))
                exp_outputs.append(tf.concat(pred, axis=0))
                labels.append(tf.concat(batch_labels, axis=0))

                train_variables = [para for para in explainer.trainable_variables if para.name.startswith('explainer')]
                grads = tape.gradient(loss, train_variables)
                optimizer.apply_gradients(zip(grads, train_variables))

        train_loss = tf.reduce_mean(losses, axis=0)
        exp_outputs = tf.concat(exp_outputs, axis=0)
        label = tf.concat(labels, axis=0)
        train_acc = accuracy(exp_outputs, label)

        print('+ ep', epoch,' acc =', train_acc)
        if args.writer_path is not None:
            writer.add_scalar('CFGExplainer loss', train_loss.numpy(), epoch + 1)
            writer.add_scalar('CFGExplainer acc', train_acc.numpy(), epoch + 1)
            
        if (epoch % args.save_thresh == 0) or (epoch == args.eepochs - 1):
            if args.save_model:
                explainer.save_weights(args.explainer_path)
    
    if writer:
        writer.close()
    return

# --------------------------
# main function
# --------------------------


def main(arguments):
    """
    Main function
    ----------------
    Args:
    arguments: the sys.args for running code
    """
    # other arguments are left intact as defaults, check config.py
    # add new arguments: model
    args.d = 13  # the number of features (fixed)
    args.c = 12  # the number of classes (fixed)
    args.n = 4690  # the number of nodes in padded graph (fixed for experiment)
    args.batch_size = int(arguments[0])  # batch size
    args.path = str(arguments[1])  # the path to load the data
    args.hiddens = str(arguments[2])  # '1024-512-128'
    args.elr = float(arguments[3])  # [explainer] optimizer learning rate 0.005 (default)
    args.model_name_flag = str(arguments[4])  # 'trial_gcn_'
    args.save_path = './checkpoints/' + args.model_name_flag
    args.dataset = str(arguments[5])  # 'yancfg_test'
    args.eepochs = int(arguments[6])  # [explainer] epochs 1000
    args.embnormlize = False  # keep this False: else the output becomes NaN

    # add arguments: for logging results
    # args.writer_path = None  # change this back !!! if we want to save the model!!
    
    args.writer_path = './logs/explainer/'  # wont change
    args.disable_tqdm = True  # make True to hide progress bar
    args.save_thresh = 1  # save model state every 1 epoch

    # debugging argument
    args.debug = False  # prints out the data loading step + loads only 1 graph per sample

    # new params [explainer]
    args.explainer_name = str(arguments[7])
    args.explainer_path = './checkpoints/explainer_' + str(arguments[7]) + args.model_name_flag + args.dataset  # path to save the explainer model
    args.results_save_path = './interpretability_results'  # the path to save the results (add a git-ignore?)
    
    args.malware_list = {
        'Bagle': 0,
        'Benign': 1,
        'Bifrose': 2,
        'Hupigon': 3,
        'Ldpinch': 4,
        'Lmir': 5,
        'Rbot': 6,
        'Sdbot': 7,
        'Swizzor': 8,
        'Vundo': 9,
        'Zbot': 10,
        'Zlob': 11
    }

    # run explain code
    train_CFGExplainer()
    
    return


# running the code
if __name__ == "__main__":
    print("sys.args: ", sys.argv)
    main(sys.argv[1:])
