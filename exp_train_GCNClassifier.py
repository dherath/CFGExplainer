import sys

from util.config import args
from util.models import GCN
from util.metrics import accuracy, softmax_cross_entropy
from util.graphprocessor import YANCFG

import networkx as nx
import tensorflow as tf

# for writing results
from tensorboardX import SummaryWriter
from tqdm import tqdm

# ------------------------------
# function to train GCN model
# ------------------------------


def train_GCNClassifier():
    """ runs training for GCN model on yancfg """
    clip_value_min = -2.0
    clip_value_max = 2.0

    # loading the datasets
    data_loader = YANCFG()
    train, _, _ = data_loader.load_yancfg_data(args.path, 'padded_train', args.malware_list)
    print('+ loaded train dataset')
    test, _, _ = data_loader.load_yancfg_data(args.path, 'padded_test', args.malware_list)
    test_batch = test.batch(args.batch_size)
    del test
    print('+ loaded test dataset')

    # intializing the writer
    name = args.model_name_flag + args.dataset
    writer = None
    if args.writer_path is not None:
        writer = SummaryWriter(args.writer_path + name)

    # creating the model
    model = GCN(input_dim=args.d, output_dim=args.c)
    print('+ model: \n', model)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    print('+ all args: \n', args)
    
    # running the training epochs
    for epoch in tqdm(range(args.epochs), disable=args.disable_tqdm):

        # run minibatch training for each epoch
        outputs, labels, losses = [], [], []
        train_batch = train.shuffle(args.batch_size).batch(args.batch_size)
        for batch_id, ts_batch in enumerate(train_batch):
            # print('ep: ', epoch, ' batch: ', batch_id)
            with tf.device('/gpu:0'):
                batch_adjs, batch_feats, batch_labels, batch_ids, batch_masks = ts_batch
                with tf.GradientTape() as tape:
                    output = model.call((batch_feats, batch_adjs), training=True)
                    cross_loss = softmax_cross_entropy(output, batch_labels)
                    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
                    loss = cross_loss + args.weight_decay * lossL2
                    grads = tape.gradient(loss, model.trainable_variables)
                    cliped_grads = [tf.clip_by_value(t, clip_value_min, clip_value_max) for t in grads]
                optimizer.apply_gradients(zip(cliped_grads, model.trainable_variables))
                with tf.device('CPU'):
                    outputs.append(output)
                    labels.append(batch_labels)
                    losses.append(cross_loss)

        output = tf.concat(outputs, axis=0)  # will be of length = #tot-train-samples
        label = tf.concat(labels, axis=0)  # will be same shape as output
        train_loss = tf.reduce_mean(losses, axis=0)  # gets the mean loss for all batches
        train_acc = accuracy(output, label)
        # print("ep: ", epoch, "train loss: ", "{:.7f}".format(train_loss), "train acc: ", "{:.7f}".format(train_acc))
    
        # test iterations: will be done per epoch
        # test_batch = test.batch(args.batch_size)  # no need to shuffle
        results = evaluate_model(model, test_batch)

        if args.writer_path is not None:
            # logging for training
            writer.add_scalar('loss/train_loss', train_loss.numpy(), epoch + 1)
            writer.add_scalar('accuracy/train_acc', train_acc.numpy(), epoch + 1)
            # logging for val step
            writer.add_scalar('loss/test_loss', results['loss'].numpy(), epoch + 1)
            writer.add_scalar('accuracy/test_acc', results['accuracy'].numpy(), epoch + 1)

        if (epoch % args.save_thresh == 0) or (epoch == args.epochs - 1):
            if args.save_model:
                model.save_weights(args.save_path + args.dataset)
        
    return


def evaluate_model(model, batch_dataset):
    """
    step through all the data samples in test dataset
    and compute scores for accuracy and validation loss
    ----------------
    Args
    model (tf.model): the intance of the model
    batch_dataset (tf.data.Dataset): a dataset class, batched and shuffled
    ----------------
    Returns
    results (dict): a dictionary for {'loss', 'accuracy'}
    """
    # Not sure if the model could step throught the entire dataset
    # in one pass, so will break it into batches as well
    outputs, labels, losses = [], [], []
    # loop through all val. batches
    for batch_id, ts_batch in enumerate(batch_dataset):
        with tf.device('/gpu:0'):
            batch_adjs, batch_feats, batch_labels, batch_ids, batch_masks = ts_batch
            output = model.call((batch_feats, batch_adjs), training=False)
            val_cross_loss = softmax_cross_entropy(output, batch_labels)
            outputs.append(output)
            labels.append(batch_labels)
            losses.append(val_cross_loss)
    # compute the results
    all_outputs = tf.concat(outputs, axis=0)
    all_labels = tf.concat(labels, axis=0)
    loss = tf.reduce_mean(losses, axis=0)
    acc = accuracy(all_outputs, all_labels)
    # use in dictionary
    results = {
        "loss": loss,
        "accuracy": acc
    }
    return results


# -----------------------------
# Main function
# -----------------------------

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
    args.c = 13  # the number of classes (fixed)
    args.batch_size = int(arguments[0])  # batch size
    args.path = str(arguments[1])  # the path to load the data
    args.hiddens = str(arguments[2])  # '1024-512-128'
    args.lr = float(arguments[3])  # 0.00001
    args.model_name_flag = str(arguments[4])  # 'trial_gcn_'
    args.save_path = './checkpoints/' + args.model_name_flag
    args.dataset = str(arguments[5])  # 'yancfg_test'
    args.epochs = int(arguments[6])  # 1000
    args.embnormlize = False  # keep this False: else the output becomes NaN

    # add arguments: for logging results
    args.writer_path = './logs/classifier/'  # wont change
    args.disable_tqdm = True  # make False to see progress bar
    args.save_thresh = 5  # save model state every 5 epochs
    
    args.malware_list = {
        'Bagle': 0,
        'Benign': 1,
        'Bifrose': 2,
        'Hupigon': 3,
        'Koobface': 4,
        'Ldpinch': 5,
        'Lmir': 6,
        'Rbot': 7,
        'Sdbot': 8,
        'Swizzor': 9,
        'Vundo': 10,
        'Zbot': 11,
        'Zlob': 12
    }

    # debugging argument
    args.debug = False  # prints out the data loading step + loads only 1 graph per sample
    if args.debug:
        print("Experimenting in DEBUG mode!")
    
    # run train()
    train_GCNClassifier()
    
    return


# running the code
if __name__ == "__main__":
    print("sys.args: ", sys.argv)
    main(sys.argv[1:])
