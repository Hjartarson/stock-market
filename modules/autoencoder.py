
import numpy as np
import tensorflow as tf
print('tensorflow version:', tf.__version__)

class AutoEncoder:
    epochs = 60
    learning_rate = 0.001
    encoder = None
    sess = None
    days = None

    def __init__(self, shape=None):
        self.days = shape
        self.inputs_ = tf.placeholder(dtype=tf.float32, shape=(None, shape, 1), name="input")
        self.targets_ = tf.placeholder(dtype=tf.float32, shape=(None, shape, 1), name="target")

    def encoder(self):
        ### BUILD MODEL

        ### Encoder
        enc_layer1 = tf.layers.conv1d(inputs=self.inputs_,
                                      filters=8,
                                      kernel_size=7,
                                      strides=1,
                                      padding='same',
                                      activation=tf.nn.relu,
                                      name="conv1")
        print(enc_layer1.get_shape())
        enc_layer2 = tf.layers.max_pooling1d(enc_layer1,
                                             pool_size=7,
                                             strides=7,
                                             padding='same')

        print(enc_layer2.get_shape())
        self.encoder = enc_layer2

    def decoder(self):
        ### Decoder
        dec_layer1 = tf.image.resize_images(self.encoder,
                                            size=(1 ,self.days),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        print(dec_layer1.get_shape())
        dec_layer2 = tf.layers.conv1d(dec_layer1,
                                      filters=8,
                                      kernel_size=7,
                                      strides=1,
                                      padding='same',
                                      activation=tf.nn.relu)
        print(dec_layer2.get_shape())
        ### Logits, activation
        logits = tf.layers.conv1d(inputs=dec_layer2,
                                  filters=1,
                                  kernel_size=7,
                                  padding='same',
                                  activation=None)
        print(logits.get_shape())
        self.logits = logits
        decoded = tf.nn.sigmoid(logits)
        print(decoded.get_shape())

    def train(self, df_stocks):
        """
        Train model
        """
        self.encoder()
        logits = self.decoder()
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_, logits=self.logits)

        # Get cost and define the optimizer
        cost = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)


        ### Train model
        self.sess = tf.Session()
        # sess.partial_run
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.epochs):
            batch_cost, _ = self.sess.run([cost, opt], feed_dict={self.inputs_: df_stocks.values.reshape((-1, self.days, 1)),
                                                                  self.targets_: df_stocks.values.reshape
                                                                      ((-1, self.days, 1))})
            if e% 10 == 0:
                print("Epoch: {}/{}...".format(e + 1, self.epochs),
                      "Training loss: {:.4f}".format(batch_cost))

    def encode(self, df_stocks):
        """
        encode
        """
        encoded = self.sess.run(self.encoder, feed_dict={self.inputs_: df_stocks.values.reshape((-1, self.days, 1))})
        return np.reshape(encoded, (-1, self.encoder.get_shape()[1] * self.encoder.get_shape()[2]))