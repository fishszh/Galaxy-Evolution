import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import glob

from gen_gif import gen_gif

class Config:
    def __init__(self):
        self.tempro_steps = 20  # temporal frame number to train
        self.tempro_steps_interval = 1 # temporal frame interval
        self.tempro_steps_gen = 40 # temporal frame number to genertate
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
        steps = self.cfg.tempro_steps * self.cfg.tempro_steps_interval
        test_x = tf.expand_dims(images[start_frame:start_frame+steps:self.cfg.tempro_steps_interval], axis=0)
        test_y = tf.expand_dims(images[start_frame+steps:start_frame+2*steps:self.cfg.tempro_steps_interval], axis=0)
        logits = model(test_x)
        for i in range(self.cfg.tempro_steps):
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            fig.subplots_adjust(wspace=0, top=1, left=0, bottom=0, right=1)
            plt.suptitle('Epoch(%d), Initial Input(%d:%d:%d), Output(%d)' % (epoch, start_frame, start_frame+steps, self.cfg.tempro_steps_interval, start_frame+steps+(i+1)*self.cfg.tempro_steps_interval),color='white',fontsize='xx-large')
            loc_x, loc_y = 0.05*test_x[0,0].shape[0], 0.95*test_x[0,0].shape[1]
            ssim = tf.image.ssim(test_y[0,i], logits[0,i], max_val=1.0, filter_size=5)
            axes[0].imshow(test_x[0,i])
            axes[0].text(loc_x, loc_y, 'Ground truth', color='white', fontsize='xx-large')
            axes[0].axis('off')
            axes[1].imshow(logits[0,i])
            axes[1].text(loc_x, loc_y, 'Prediction', color='white', fontsize='xx-large')
            axes[1].text(10*loc_x, loc_y, 'SSIM: '+str(round(ssim.numpy(), 3)), color='white', fontsize='xx-large')
            axes[1].axis('off')
            plt.savefig(self.cfg.image_save_path + '/com_%04d_%04d' %(start_frame,i))

    def gen_plot(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the long term prediction images
        x: initial state
        '''
        steps = self.cfg.tempro_steps* self.cfg.tempro_steps_interval
        loc_x, loc_y = 0.05*images[0].shape[0], 0.95*images[0].shape[1]
        x = tf.expand_dims(images[start_frame:start_frame+steps:self.cfg.tempro_steps_interval], axis=0)
        for i in range(self.cfg.tempro_steps_gen):
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            fig.subplots_adjust(wspace=0, top=1, left=0, bottom=0, right=1)
            plt.suptitle('Epoch(%d), Initial input(%d:%d:%d)  Output(%d)' % (epoch, start_frame, start_frame+steps, self.cfg.tempro_steps_interval, target_frame),color='white', fontsize='xx-large')
            if i % self.cfg.tempro_steps_gen//50 == 0:
                y = model(x)

                target_frame = start_frame + steps + (i+1)*self.cfg.tempro_steps_interval
                x = tf.concat([x, y], axis=1)
                x = x[:,1:self.cfg.tempro_steps+1]
                ssim = tf.image.ssim(images[target_frame], y[0,0], max_val=1.0, filter_size=5)
                
                axes[0].imshow(images[target_frame])
                axes[0].text(loc_x, loc_y, 'Ground truth', color='white', fontsize='xx-large')
                axes[0].axis('off')
                axes[1].imshow(y[0,0])
                axes[1].text(loc_x, loc_y, 'Prediction', color='white', fontsize='xx-large')
                axes[1].text(10*loc_x, loc_y, 'SSIM: '+str(round(ssim.numpy(), 3)), color='white', fontsize='xx-large')
                axes[1].axis('off')
                plt.savefig(self.cfg.image_save_path +'/gen_%04d_%04d' %(start_frame,i))

    def process_gif(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the short term and long term prediction gif
        '''
        self.com_plot(model, images, loss_model, epoch, start_frame)
        com_pattern = [
            self.cfg.image_save_path, 
            '/com_%04d_*.png' % start_frame,
            '/com_%s_%04d.gif' %(loss_model, start_frame)]
        gen_gif(*com_pattern)

        self.gen_plot(model, images, loss_model, epoch, start_frame)
        gen_pattern = [
            self.cfg.image_save_path, 
            '/gen_%04d_*.png' % start_frame,
            '/gen_%s_%04d_%d.gif' %(loss_model, start_frame, self.cfg.tempro_steps_gen)]
        gen_gif(*gen_pattern)
