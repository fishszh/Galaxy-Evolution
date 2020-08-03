import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import glob

from gen_gif import gen_gif

class Config:
    def __init__(self):
        self.tempro_steps = 30  # temporal frame number to train
        self.tempro_steps_interval = 2 # temporal frame interval
        self.tempro_steps_gen = 20 # temporal frame number to genertate
        self.image_size = [None, None, None, 3]

        self.batch_size = 6
        self.buffer_size = self.batch_size * 5
        self.epochs = 20
        self.lr = 1e-4

        self.set_model_name()

        self.log_dir = '../logs/{name}'.format(name=self.model_name)
        self.ckpt_path = '../ckpts/{name}'.format(name=self.model_name)
        self.image_save_path = '../imgs/{name}'.format(name=self.model_name)


    def set_model_name(self):
        raise Exception('Model name is not specified in Config!!!')

    def process_gif(self, model, images, loss_model, epoch, start_frame):
        pg = ProcessGif(self)
        return pg.process_gif(model, images, loss_model, epoch, start_frame)

class ProcessGif():
    def __init__(self, config):
        self.cfg = config

    def com_plot(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the short term prediction images
        '''
        end_frame = start_frame + self.cfg.tempro_steps * self.cfg.tempro_steps_interval
        test_inputs = tf.expand_dims(images[start_frame:end_frame:self.cfg.tempro_steps_interval], axis=0)
        logits = model(test_inputs[:,:-1])
        fig, axes = plt.subplots(1,2, figsize=(10,5))
        fig.subplots_adjust(wspace=0, top=1, left=0, bottom=0, right=1)
        for i in range(self.cfg.tempro_steps-1):
            plt.suptitle('Comparison: Epoch %4d, Frame %d--%d' % (epoch, start_frame, start_frame+i+1),color='white',fontsize='xx-large')
            axes[0].imshow(test_inputs[0,i])
            axes[0].axis('off')
            axes[1].imshow(logits[0,i])
            axes[1].axis('off')
            plt.savefig(self.cfg.image_save_path + self.cfg.model_name+ '/com_%04d_%04d' %(start_frame,i))

    def gen_plot(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the long term prediction images
        x: initial state
        '''
        x = tf.expand_dims(images[start_frame:start_frame+1], axis=0)
        y = x
        fig, axes = plt.subplots(1,2, figsize=(10,5))
        fig.subplots_adjust(wspace=0, top=1, left=0, bottom=0, right=1)
        for i in range(self.cfg.tempro_steps_gen):
            if i % self.cfg.tempro_steps_gen//50 == 0:
                y = model(y)
                index = start_frame+i*self.cfg.tempro_steps_interval
                x = images[index:index+1]
                plt.suptitle('Epoch %4d, Frame %d--%d' % (epoch, start_frame, start_frame+i+1),color='white', fontsize='xx-large')
                axes[0].imshow(x[0])
                axes[0].axis('off')
                axes[1].imshow(y[0,0])
                axes[1].axis('off')
                plt.savefig(self.cfg.image_save_path + self.cfg.model_name +'/gen_%04d_%04d' %(start_frame,i))

    def process_gif(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the short term and long term prediction gif
        '''
        self.com_plot(model, images, loss_model, epoch, start_frame)
        com_pattern = [
            self.cfg.image_save_path, 
            self.cfg.model_name +'/com_%04d_*.png' % start_frame,
            self.cfg.model_name +'/com_%s_%04d.gif' %(loss_model, start_frame)]
        gen_gif(*com_pattern)

        self.gen_plot(model, images, loss_model, epoch, start_frame)
        gen_pattern = [
            self.cfg.image_save_path, 
            '/gen_%04d_*.png' % start_frame,
            '/gen_%s_%04d_%d.gif' %(loss_model, start_frame, self.cfg.tempro_steps_gen)]
        gen_gif(*gen_pattern)
