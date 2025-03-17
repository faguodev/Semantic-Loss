import argparse
import sys
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from mnist_input import read_data_sets

def standarization(x_images):
    x_images = tf.reshape(x_images, [-1, 28, 28, 1])
    x_images = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x_images)
    return tf.reshape(x_images, [-1, 784])

def image_processing(x_images, batch_size):
    noise = tf.random.normal(tf.shape(x_images), mean=0.0, stddev=0.3, dtype=tf.float32)
    x_images = x_images + noise
    x_images = tf.reshape(x_images, [-1, 28, 28, 1])
    x_images = tf.image.random_crop(x_images, [batch_size, 25, 25, 1])
    x_images = tf.image.resize_with_crop_or_pad(x_images, 28, 28)
    return tf.reshape(x_images, [-1, 784])

def build_model():
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Lambda(standarization),
        layers.Dense(1000, activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(250, activation='relu'),
        layers.Dense(250, activation='relu'),
        layers.Dense(250),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10)  # No activation (logits)
    ])
    return model

def compute_loss(logits, labels, label_mask):
    cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1)
    normalized_logits = tf.nn.sigmoid(logits)
    batch_number = tf.shape(logits)[0]

    wmc_tmp = tf.zeros([batch_number,])
    for i in range(10):
        one_situation = tf.concat([
            tf.concat([tf.ones([batch_number, i]), tf.zeros([batch_number, 1])], axis=1),
            tf.ones([batch_number, 10 - i - 1])
        ], axis=1)
        wmc_tmp += tf.reduce_prod(one_situation - normalized_logits, axis=1)

    wmc_tmp = tf.abs(wmc_tmp)
    log_wmc = tf.math.log(wmc_tmp + 1e-10)  # avoid log(0)

    unlabeled_mask = 1.0 - label_mask
    loss = -0.0005 * unlabeled_mask * log_wmc - 0.0005 * label_mask * log_wmc + label_mask * cross_entropy
    return tf.reduce_mean(loss), tf.reduce_mean(wmc_tmp)

def compute_accuracy(logits, labels, label_mask):
    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    correct = tf.cast(correct, tf.float32) * label_mask
    return tf.reduce_sum(correct) / tf.reduce_sum(label_mask)

def train_model(args):
    mnist = read_data_sets(args.data_path, n_labeled=args.num_labeled, one_hot=True)
    model = build_model()
    optimizer = optimizers.Adam(1e-4)

    log_path = "log.txt"
    open(log_path, 'w').close()  # clear log file

    for step in range(50000):
        images, labels = mnist.train.next_batch(args.batch_size)
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        with tf.GradientTape() as tape:
            processed_images = image_processing(images, args.batch_size)
            logits = model(processed_images, training=True)
            label_mask = tf.cast(tf.reduce_max(labels, axis=1) > 0, tf.float32)
            loss, wmc = compute_loss(logits, labels, label_mask)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc = compute_accuracy(logits, labels, label_mask)

        if step % 100 == 0:
            with open(log_path, 'a') as f:
                f.write(f"step {step}, training_accuracy {acc.numpy()}, train_loss {loss.numpy()}, wmc {wmc.numpy()}\n")
                print(f"step {step}, training_accuracy {acc.numpy()}, train_loss {loss.numpy()}, wmc {wmc.numpy()}")

        if step % 500 == 0:
            test_logits = model(mnist.test.images, training=False)
            test_labels = tf.convert_to_tensor(mnist.test.labels, dtype=tf.float32)
            test_mask = tf.cast(tf.reduce_max(test_labels, axis=1) > 0, tf.float32)
            test_acc = compute_accuracy(test_logits, test_labels, test_mask)
            with open(log_path, 'a') as f:
                f.write(f"test accuracy {test_acc.numpy()}\n")
                print(f"test accuracy {test_acc.numpy()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='mnist_data/', help='Directory for storing input data')
    parser.add_argument('--num_labeled', type=int, required=True, help='Num of labeled examples for semi-supervised learning.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training.')
    args = parser.parse_args()

    train_model(args)
