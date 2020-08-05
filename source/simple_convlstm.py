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
    index = [i for i in range(2100) if i % 100 < 40]
    for i in index:

        steps = cfg.tempro_steps * cfg.tempro_steps_interval
        image_x = images[i : i+steps*2 : cfg.tempro_steps_interval]
        # image_y = images[i+steps : i+2*steps: cfg.tempro_steps_interval]
        yield (image_x)

# %%
train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32)) \
                .shuffle(cfg.buffer_size).batch(cfg.batch_size) \
                .prefetch(tf.data.experimental.AUTOTUNE)

# %%
convlstm = keras.Sequential([
    layers.InputLayer(cfg.image_size),
    layers.ConvLSTM2D(32, 3, 2, 'same', return_sequences=True, use_bias=False),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(64, 3, 2, 'same', return_sequences=True, use_bias=False),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(128, 3, 2, 'same', return_sequences=True, use_bias=False),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(128, 1, 1, 'same', return_sequences=True, use_bias=False),
    layers.BatchNormalization(),
    layers.Conv3DTranspose(64, 3, [1,2,2], 'same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv3DTranspose(64, 1, 1, 'same', activation='relu'),
    layers.Conv3DTranspose(32, 3, [1,2,2], 'same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv3DTranspose(32, 1, 1, 'same', activation='relu'),
    layers.Conv3DTranspose(3, 3, [1,2,2], activation='sigmoid', padding='same', data_format='channels_last')
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
def train_model(loss_model='mix'):
    if loss_model == 'mix':
        loss_func = mix_loss_func
    else:
        loss_func = ssim_loss_func
    for epoch in range(1, cfg.epochs+1):
        start_time = time.time()
        for train_batch in train_ds:
            train_step(train_batch, loss_func)
        end_time = time.time()
        cus = (end_time - start_time)/60
        
        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            cfg.process_gif(convlstm, images, loss_model, epoch, 200)
            cfg.process_gif(convlstm, images, loss_model, epoch, 500)
            cfg.process_gif(convlstm, images, loss_model, epoch, 700)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1000)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1250)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1500)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1800)
            cfg.process_gif(convlstm, images, loss_model, epoch, 2000)

            cfg.process_gif(convlstm, images, loss_model, epoch, 281)
            cfg.process_gif(convlstm, images, loss_model, epoch, 581)
            cfg.process_gif(convlstm, images, loss_model, epoch, 781)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1081)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1281)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1581)
            cfg.process_gif(convlstm, images, loss_model, epoch, 1881)
            cfg.process_gif(convlstm, images, loss_model, epoch, 2081)
            

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
def mae_loss_func(y, y_pred):
    loss = tf.math.abs(y-y_pred)
    loss = tf.reduce_mean(loss, axis=(2,3,4))
    return loss

@tf.function
def ssim_loss_func(y, y_pred):
    loss = tf.image.ssim(y,y_pred, max_val=1.0, filter_size=5)
    return 1-loss

@tf.function
def mix_loss_func(y, y_pred):
    loss = 0.85*ssim_loss_func(y, y_pred) + 0.15*mae_loss_func(y, y_pred)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(train_batch, loss_func):
    x, y = tf.split(train_batch, num_or_size_splits=2, axis=1)
    with tf.GradientTape() as tape:
        y_pred = convlstm(x, training=True)
        loss = mix_loss_func(y, y_pred)
    grads = tape.gradient(loss, convlstm.trainable_variables)
    optimizers.apply_gradients(zip(grads, convlstm.trainable_variables))

    train_loss.update_state(loss)


# %%

if __name__ == "__main__":
    # train_model(loss_model='mix')
    cfg.process_gif(convlstm, images, 'mix', 1, 200)
# %%
