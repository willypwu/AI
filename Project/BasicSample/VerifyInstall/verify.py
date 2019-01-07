import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.enable_eager_execution()
print(tf.reduce_sum(tf.random_normal([1000, 1000])))
