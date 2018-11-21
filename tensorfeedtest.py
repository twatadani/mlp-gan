import tensorflow as tf
import numpy as np


input = tf.placeholder(dtype=tf.float32, shape=(1, 1), name='input')
ones = tf.ones(dtype=tf.float32, shape=(1, 1), name='ones')

input_from_numpy = tf.placeholder(dtype=tf.bool, shape=(), name='input_from_numpy')

input_selected = tf.cond(input_from_numpy,
                         lambda: input,
                         lambda: ones,
                         name = 'input_selected')

calc = tf.matmul(input_selected, input_selected)

nptensor = np.full(shape=(1, 1), fill_value=18, dtype=np.float)
dummy_nparray = np.zeros(shape=(1, 1), dtype=np.float)

with tf.Session() as session:
    print('NumPyからのarrayをfeedします。')
    print(session.run(calc, feed_dict = { input: nptensor,
                                          input_from_numpy: True}))                                    
    print('Tensorをfeedします。')
    print(session.run(calc, feed_dict = { input: dummy_nparray,
                                          input_from_numpy: False }))

