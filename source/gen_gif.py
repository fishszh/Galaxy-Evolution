import glob
import imageio
import os

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def gen_gif(img_path, labels, gif_name):
    image_list = glob.glob(img_path+labels)
    image_list = sorted(image_list)
    duration = 0.3
    create_gif(image_list, img_path+gif_name, duration)
    for f in image_list:
        os.remove(f)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # minst_dense = ['../imgs/VAE/', '*minst_dense*','minst_dense']
    minst_cnn = ['../imgs/', 'VAE/*minst_cvae*.png','minst_cvae']
    gen_gif(*minst_cnn)