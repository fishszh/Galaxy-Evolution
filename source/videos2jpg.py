import cv2
import os

'''
transform videos to images
'''

def mp4_avi(vedio_path, img_path):
    VideoCapture = cv2.VideoCapture(vedio_path)
    success, frame = VideoCapture.read()
    i = 0
    while success:
        i += 1
        save_img(frame, img_path, i)
        if success:
            print('save_img: %s %d' % (img_path, i))
        success, frame = VideoCapture.read()
    VideoCapture.release()

def save_img(image, img_path, i):
    address = img_path + '%04d.jpg' % i
    cv2.imwrite(address, image)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    yang = ['../data/yang.mp4', '../data/yang/']
    collision = ['../data/collision.mpg', '../data/collision/']
    colliding = ['../data/colliding_galaxies_with_BH.avi', '../data/colliding/']
    # mp4_avi(*yang)
    mp4_avi(*colliding)
    # mp4_avi(*collision)