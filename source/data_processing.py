# %%
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def glob_image_path(path, pattern):
    '''
    glob all image path 
    '''
    files = glob.glob(path + pattern)
    files = sorted(files)
    return files

def process_img(img_path):
    '''
    process the image to normed tensors
    '''
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)

    ratio = image.shape[1]/image.shape[0]
    image = tf.image.resize(image, [128, 128])
    image /= 255.
    return image

def gen_images_set(img_paths):
    image_set = []
    for path in img_paths:
        image_set.append(process_img(path))
    image_set = tf.stack(image_set, axis=0)
    return image_set

def gen_train_data(img_paths, frame_num, i):
    path_x = img_paths[i:i+frame_num-1]
    path_y = img_paths[i+1:i+frame_num]
    return gen_images_set(path_x), gen_images_set(path_y)




# %%
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    img_paths = glob_image_path('../../Galaxy/data/colliding/', '*.jpg')
    img = process_img(img_paths[0])
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
 # %%


# %%
