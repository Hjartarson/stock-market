
import numpy as np
import pandas as pd

import tensorflow as tf
print('tensorflow version:', tf.__version__)

class Clustering:

    def __init__(self, points, num_clusters):
        self.points = points
        self.kmeans = tf.contrib.factorization.KMeansClustering(
            num_clusters=num_clusters, use_mini_batch=False)


    def input_fn(self):
        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.points, dtype=tf.float32), num_epochs=1)

    def train(self):
        num_iterations = 6
        previous_centers = None
        for _ in range(num_iterations):
            self.kmeans.train(self.input_fn)
            cluster_centers = self.kmeans.cluster_centers()
            #if previous_centers is not None:
            #    print('delta:', cluster_centers - previous_centers)
            previous_centers = cluster_centers
            print('score:', self.kmeans.score(self.input_fn))
        #print('cluster centers:', cluster_centers)

    def predict(self):
        return list(self.kmeans.predict_cluster_index(self.input_fn))