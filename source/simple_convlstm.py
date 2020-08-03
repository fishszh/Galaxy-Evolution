# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

import matplotlib.pyplot as plt
import numpy as np
import os
import time

from base_framework import Config
from data_processing import glob_image_path, process_img, gen_images_set

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SubConfig(Config):
    def __init__(self):
        super(SubConfig, self).__init__()

    def set_model_name(self):
        self.model_name = 'SimpleConvLSTM'


cfg = SubConfig()

# %%
img_path = glob_image_path('../data/colliding/', '*.jpg')
images = gen_images_set(img_path)
def generator():
    for i in tf.range(1000):
        end_i = i + cfg.tempro_steps * cfg.tempro_steps_interval
        image_frame = images[i : end_i : cfg.tempro_steps_interval]
        # image_y = images[i+1 : i+FRAMES_NUM+1]
        yield (image_frame)

# %%
train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32)) \
                .shuffle(cfg.buffer_size).batch(cfg.batch_size) \
                .prefetch(tf.data.experimental.AUTOTUNE)

# %%
convlstm = keras.Sequential([
    layers.InputLayer(cfg.image_size),
    layers.ConvLSTM2D(10, 5, 1, 'same', return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(10, 5, 1, 'same', return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(10, 5, 1, 'same', return_sequences=True),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(10, 5, 1, 'same', return_sequences=True),
    layers.BatchNormalization(),
    layers.Conv3D(3, 5, 1, activation='sigmoid', padding='same', data_format='channels_last')
])

# convlstm.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4))

convlstm.summary()

optimizers = tf.keras.optimizers.Adam(lr=cfg.lr)
train_loss = tf.keras.metrics.Mean()
checkpoint = tf.train.Checkpoint(model=convlstm)
ckpt_manager = tf.train.CheckpointManager(checkpoint, cfg.ckpt_path, max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)

# %%
def train_model(loss_model='ssim'):
    if loss_model == 'ssim':
        loss_func = ssim_loss_func
    else:
        loss_func = mse_loss_func
    for epoch in range(1, cfg.epochs+1):
        start_time = time.time()
        for train_batch in train_ds:
            train_step(train_batch, loss_func)
        end_time = time.time()
        cus = (end_time - start_time)/60
        
        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            cfg.process_gif(epoch, 250, loss_model, images)
            cfg.process_gif(epoch, 300, loss_model, images)
            cfg.process_gif(epoch, 500, loss_model, images)
            cfg.process_gif(epoch, 1200, loss_model, images)
            cfg.process_gif(epoch, 1500, loss_model, images)
            cfg.process_gif(epoch, 2000, loss_model, images)
            

        if epoch % 1 == 0:
            print('Epoch:%d|%d, train loss: %f, %.2f min/epoch-->Total: %d min' 
                    % (epoch, cfg.epochs, train_loss.result(), cus, cfg.epochs*cus))
        train_loss.reset_states()

@tf.function
def mse_loss_func(y, y_pred):
    loss = tf.square(y_pred-y)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def ssim_loss_func(y, y_pred):
    loss = tf.image.ssim(y,y_pred, max_val=1.0)
    loss = tf.reduce_mean(loss)
    return 1-loss

@tf.function
def train_step(train_batch, loss_func):
    with tf.GradientTape() as tape:
        y_pred = convlstm(train_batch[:,:-1])
        loss = loss_func(train_batch[:,1:], y_pred)
    grads = tape.gradient(loss, convlstm.trainable_variables)
    optimizers.apply_gradients(zip(grads, convlstm.trainable_variables))

    train_loss.update_state(loss)


# %%

if __name__ == "__main__":
    train_model(loss_model='ssim')
    
# %%
