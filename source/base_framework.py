import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import glob

from gen_gif import gen_gif

class Config:
    def __init__(self):
        self.tempro_steps = 20  # temporal frame number to train
        self.tempro_steps_interval = 2 # temporal frame interval
        self.tempro_steps_gen = 30 # temporal frame number to genertate
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
        # initial input and output frames
        steps_input = self.cfg.tempro_steps * self.cfg.tempro_steps_interval
        steps_output = self.cfg.tempro_steps_gen * self.cfg.tempro_steps_interval
        test_data = tf.expand_dims(images[start_frame:start_frame+steps_input+steps_output:self.cfg.tempro_steps_interval], axis=0)
        x, y = tf.split(test_data, num_or_size_splits=[self.cfg.tempro_steps, self.cfg.tempro_steps_gen], axis=1)
        x_start_frame = start_frame
        x_end_frame = start_frame + steps_input
        y_start_frame = start_frame + steps_input
        y_end_frame = y_start_frame + steps_input+steps_output

        # short term prediction
        y_short_term = model(x)
        ssim_short_term = tf.image.ssim(y[0,:self.cfg.tempro_steps], y_short_term[0], max_val=1.0, filter_size=5)
        # initial long term input
        x_long_term = x
        y_long_term = []
        ssim_long_term = []

        loc_x, loc_y = 0.05*x[0,0].shape[0], 0.95*x[0,0].shape[1]
        
        for i in range(self.cfg.tempro_steps_gen):
            # update y_long_term and x_long_term
            y_temp = model(x_long_term)
            x_long_term = tf.concat([x_long_term, y_temp], axis=1)
            x_long_term = x_long_term[:,1:1+self.cfg.tempro_steps_gen]
            y_long_term.append(y_temp[0,0])

            # update ssim_long_term
            ssim_temp = tf.image.ssim(y[0,i], y_temp[0,0], max_val=1.0, filter_size=5)
            ssim_long_term.append(ssim_temp.numpy())

        for i in range(self.cfg.tempro_steps_gen):
            input_frame = start_frame + i*self.cfg.tempro_steps_interval
            output_frame = input_frame + steps_input

            fig, axes = plt.subplots(1, 4, figsize=(20,5))
            fig.subplots_adjust(wspace=0, top=1, left=0, bottom=0, right=1)
            if i < self.cfg.tempro_steps:
                axes[0].imshow(x[0,i])
                axes[1].imshow(y_short_term[0,i])
            else:
                axes[0].imshow(x[0,self.cfg.tempro_steps-1])
                axes[1].imshow(y_short_term[0,self.cfg.tempro_steps-1])

            axes[0].text(loc_x, loc_y, 'Input %d' % input_frame, color='white', fontsize='xx-large')
            axes[1].text(loc_x, loc_y, 'Short term Prediction', color='white', fontsize='xx-large')
            axes[0].axis('off')
            axes[1].axis('off')
            
            axes[2].imshow(y[0,i])
            axes[2].text(loc_x, loc_y, 'Ground truth %d' % output_frame, color='white', fontsize='xx-large')
            axes[2].axis('off')
            
            axes[3].imshow(y_long_term[i])
            axes[3].text(loc_x, loc_y, 'Long term Prediction', color='white', fontsize='xx-large')
            axes[3].axis('off')

            # anotate the SSIM for short term prediction
            axin = fig.add_axes([0.63, 0.08, 0.1, 0.17], frameon=False, facecolor='none')
            axin.plot(ssim_short_term.numpy(), 'w', lw=3, alpha=0.3)
            if i < self.cfg.tempro_steps:
                axin.plot(i, ssim_short_term.numpy()[i], 'yo', alpha=0.7)
            axin.tick_params(axis='y',colors='gold')
            axin.set_ylabel('SSIM', color='gold')
            axin.set_xticks([])
            axin.grid(True)
            # anotate the SSIM for long term prediction
            axin = fig.add_axes([0.88, 0.08, 0.1, 0.17], frameon=False, facecolor='none')
            axin.plot(ssim_long_term, 'w', lw=3, alpha=0.3)
            axin.plot(i, ssim_long_term[i], 'yo', alpha=0.7)
            axin.tick_params(axis='y',colors='gold')
            axin.set_ylabel('SSIM', color='gold')
            axin.set_xticks([])
            axin.set_xlim(0, self.cfg.tempro_steps)
            axin.grid(True)
            plt.savefig('/content/com_%04d_%04d' %(start_frame,i))
            plt.close()


    def process_gif(self, model, images, loss_model, epoch, start_frame):
        '''
        generate the short term and long term prediction gif
        '''
        self.com_plot(model, images, loss_model, epoch, start_frame)
        com_pattern = [
            self.cfg.image_save_path, 
            '/content/com_%04d_*.png' % start_frame,
            '/com_%s_%04d.gif' %(loss_model, start_frame)]
        gen_gif(*com_pattern)
