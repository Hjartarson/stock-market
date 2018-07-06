
import numpy as np
import tensorflow as tf
print('tensorflow version:', tf.__version__)


class AutoEncoder:
    epochs = 20
    learning_rate = 0.001
    encoder = None
    sess = None
    shape = None

    def __init__(self, shape=None):
        self.shape = shape
        self.inputs_ = tf.placeholder(dtype=tf.float32, shape=(None, shape, 1), name="input")
        self.targets_ = tf.placeholder(dtype=tf.float32, shape=(None, shape, 1), name="target")

    def build_autoencoder(self, conv_layers, pool_layers):
        enc_layer = self.inputs_
        shape_layers = {}
        print(enc_layer.get_shape())
        for conv, pool in zip(conv_layers.keys(), pool_layers.keys()):
            print(conv,pool)
            samples, width, length = enc_layer.get_shape()
            shape_layers[pool] = width
            ### Encoder
            enc_layer = tf.layers.conv1d(inputs=enc_layer,
                                         filters=conv_layers[conv]['filters'],
                                         kernel_size=conv_layers[conv]['kernel_size'],
                                         strides=conv_layers[conv]['strides'],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name=conv)
            enc_layer = tf.layers.max_pooling1d(enc_layer,
                                                pool_size=pool_layers[pool]['pool_size'],
                                                strides=pool_layers[pool]['strides'],
                                                padding='same')
            print(enc_layer.get_shape())


        self.encoder = enc_layer
        dec_layer = self.encoder

        for conv, pool in zip(reversed(list(conv_layers.keys())), reversed(list(pool_layers.keys()))):
            print(conv, pool)
            dec_layer = tf.image.resize_images(dec_layer,
                                               size=(1, shape_layers[pool]),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            dec_layer = tf.layers.conv1d(dec_layer,
                                         filters=conv_layers[conv]['filters'],
                                         kernel_size=conv_layers[conv]['kernel_size'],
                                         strides=conv_layers[conv]['strides'],
                                         padding='same',
                                         activation=tf.nn.relu)
            print(dec_layer.get_shape())

        ### Logits, activation
        logits = tf.layers.conv1d(inputs=dec_layer,
                                  filters=1,
                                  kernel_size=7,
                                  padding='same',
                                  activation=None)
        # print(logits.get_shape())

        decoded = tf.nn.sigmoid(logits)
        return logits

    def train(self, df_player, conv_layers, pool_layers):
        """
        Train model
        """
        # self.encoder()
        # logits = self.decoder()
        logits = self.build_autoencoder(conv_layers, pool_layers)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_, logits=logits)

        # Get cost and define the optimizer
        cost = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        ### Train model
        self.sess = tf.Session()
        # sess.partial_run
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.epochs):
            batch_cost, _ = self.sess.run([cost, opt],
                                          feed_dict={self.inputs_: df_player.values.reshape((-1, self.shape, 1)),
                                                     self.targets_: df_player.values.reshape((-1, self.shape, 1))})
            if e % 10 == 0:
                print("Epoch: {}/{}...".format(e + 1, self.epochs),
                      "Training loss: {:.4f}".format(batch_cost))

    def encode(self, df_player):
        """
        encode
        """
        encoded = self.sess.run(self.encoder, feed_dict={self.inputs_: df_player.values.reshape((-1, self.shape, 1))})
        return np.reshape(encoded, (-1, self.encoder.get_shape()[1] * self.encoder.get_shape()[2]))