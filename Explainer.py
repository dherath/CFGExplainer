import sys
import tensorflow as tf


class ExplainerModule(tf.keras.Model):
    def __init__(self, model, output_dim, **kwargs):
        super(ExplainerModule, self).__init__(**kwargs)
        with tf.name_scope("explainer") as scope:
            self.elayers = [tf.keras.layers.Dense(64, tf.nn.relu),
                            tf.keras.layers.Dense(32, tf.nn.relu),
                            tf.keras.layers.Dense(1)]

            self.predlayers = [tf.keras.layers.Dense(64, tf.nn.relu),
                               tf.keras.layers.Dense(32, tf.nn.relu),
                               tf.keras.layers.Dense(16, tf.nn.relu)]
            self.class_layer = tf.keras.layers.Dense(output_dim, tf.nn.softmax)
        
        return
        
    def call(self, inputs, training=True, debug=False):
        x, embed, adj, node_mask = inputs

        # call NN1 and get node importance
        importance = self.NN1(embed, node_mask, training=training, debug=debug)  # obtains node importance

        # call NN2 to get graph classification
        out = self.NN2(importance, embed, training=training, debug=debug)  # gets class probabilities, might change it use GNN model component in future
        if debug:
            print('+ call() node importance: ', tf.shape(importance))
            print('+ call() out', tf.shape(out))
        
        return out, importance

    def NN1(self, embed, node_mask=None, training=True, debug=False):
        """
        First part of model: obtains block importance
        """
        # loop through self.elayers and obtain output
        h = embed
        for elayer in self.elayers:
            h = elayer(h)
        if node_mask is not None:
            node_mask = tf.reshape(node_mask, tf.shape(h))
            if debug:
                print('+ MLP1() reshaped node mask: ', tf.shape(node_mask))
            h = tf.math.multiply(h, node_mask)
        # do sigmoid() activation to get probability
        return tf.sigmoid(h)

    def NN2(self, importance, embed, training=True, debug=False):
        """
        Second part of model: computes classification probability
        version 2: just use the node importance weighted by embeddings
        [future variations: load GNN model weights to initalize layers]
        """
        
        tf_shape = tf.shape(embed) # # usually [#batches, #nodes, #emb-length]
        _dim, _flatten = None, None
        if training is True:
            _flatten = (tf_shape[0], -1)  # used to reshape for final class_layer
            _dim = (tf_shape[0], tf_shape[1], 1)  # used to reshape the importance scores
        else:
            _flatten = (1, -1)  # during testing will call one graph at a time
            _dim = (tf_shape[0], -1)

        reshaped_importance = tf.reshape(importance, _dim) # so that each embedding is multiplied with importance
        # do scalar multiplication
        h = tf.math.multiply(embed, reshaped_importance)

        for layer in self.predlayers:
            h = layer(h)

        # h_flatten = tf.reshape(h, [tf_shape[0], -1]) # tf_shape[0] = batch_size during training
        h_flatten = tf.reshape(h, _flatten)
        out = self.class_layer(h_flatten)  # class labels
        
        if debug:
            print('+ MLP2 <tensor shapes>')
            print('+ embed:', tf.shape(embed))
            print('+ importance:', tf.shape(importance))
            print('+ reshaped-importance:', tf.shape(reshaped_importance))
            print('+ h:', tf.shape(h))
            print('+ flattend h:', tf.shape(h_flatten))
            print('+ out:', tf.shape(out))
        
        return out
    
    def loss(self, pred, pred_label, debug=False):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        # class probability of correct(target) class
        logit = tf.reduce_sum(tf.math.multiply(pred, pred_label), axis=1)

        # then will take negative log likelihood, and take  -1* reduce_mean(logit)
        loss = -1 * tf.reduce_mean(tf.math.log(logit + 1e-10))  # in case ln(0) appears

        if debug:
            print(': loss() pred (with-sigmoid): ', pred, '\n label:', pred_label)
            print(': loss() logit:', logit)
            print(': loss() loss:', loss)
        
        return loss
