import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SOM_Network():
    def __init__(self, input_dim, dim=10, sigma=None, learning_rate=0.1, tay2=1000, dtype=tf.float32):
        '''
        input_dim: dimention of input data
        dim: grid step
        sigma: effective width (can be changible while learning)
        and some constants for the algorithm
        '''
        
        #если сигма не определена, устанавливаем ее равной половине размера решетки
        if not sigma:
            sigma = dim / 2

        self.dtype = dtype
        #определяем константы использующиеся при обучении
        self.dim = tf.constant(dim, dtype=tf.int64)
        self.learning_rate = tf.constant(learning_rate, dtype=dtype, name='learning_rate')
        self.sigma = tf.constant(sigma, dtype=dtype, name='sigma')
        #тау 1 (формула 6)
        self.tay1 = tf.constant(1000/np.log(sigma), dtype=dtype, name='tay1')
        #минимальное значение сигма на шаге 1000 (определяем по формуле 3)
        self.minsigma = tf.constant(sigma * np.exp(-1000/(1000/np.log(sigma))), dtype=dtype, name='min_sigma')
        self.tay2 = tf.constant(tay2, dtype=dtype, name='tay2')
        #input vector
        self.x = tf.placeholder(shape=[input_dim], dtype=dtype, name='input')
        #iteration number
        self.n = tf.placeholder(dtype=dtype, name='iteration')
        #матрица синаптических весов
        self.w = tf.Variable(tf.random_uniform([dim*dim, input_dim], minval=-1, maxval=1, dtype=dtype),
                             dtype=dtype, name='weights')
        #матрица позиций всех нейронов, для определения латерального расстояния
        self.positions = tf.where(tf.fill([dim, dim], True))

    
    def __competition(self, info=''):
        with tf.name_scope(info+'competition') as scope:
        #вычисляем минимум евклидова расстояния для всей сетки нейронов
        distance = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.w), axis=1))
        #возвращаем индекс победившего нейрона (формула 1)
        return tf.argmin(distance, axis=0)

    
    def training_op(self):
        #определяем индекс победившего нейрона
        win_index = self.__competition('train_')

        with tf.name_scope('cooperation') as scope:
            #вычисляем латеральное расстояние d
            #для этого переводим инедкс победившего нейрона из 1d координаты в 2d координату
            coop_dist = tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.cast(self.positions -
                                [win_index//self.dim,
                                 win_index-win_index//self.dim*self.dim],
                                dtype=self.dtype)),
                    axis=1))
            
            #корректируем сигма (используя формулу 3)
            sigma = tf.cond(
                self.n > 1000,
                lambda: self.minsigma,
                lambda: self.sigma * tf.exp(-self.n/self.tay1))
            sigma_summary = tf.summary.scalar('Sigma', sigma)
            #вычисляем топологическую окрестность (формула 2)
            tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(sigma)))

        with tf.name_scope('adaptation') as scope:
            #обновляем параметр скорости обучения (формула 5)
            lr = self.learning_rate * tf.exp(-self.n/self.tay2)
            minlr = tf.constant(0.01, dtype=self.dtype, name='min_learning_rate')
            lr = tf.cond(lr <= minlr, lambda: minlr, lambda: lr)
            lr_summary = tf.summary.scalar('Learning rate', lr)
            #вычисляем дельта весов и обновляем всю матрицу весов (формула 4)
            delta = tf.transpose(lr * tnh * tf.transpose(self.x - self.w))
            training_op = tf.assign(self.w, self.w + delta)
        return training_op
            

# testing
