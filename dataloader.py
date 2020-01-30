import os
from os.path import isdir, exists, abspath, join
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageEnhance import Contrast
# from skimage import data, io, filters, util, exposure
import torchvision.transforms.functional as TF



class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.2):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

        self.output = 388
        self.input = 572

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            start = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            start = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels

            data_image = Image.open(self.data_files[current]).resize((self.output, self.output))
            label_image = Image.open(self.label_files[current]).resize((self.output, self.output))

            if self.mode=="train":
                data_image, label_image = self.applyDataAugmentation(data_image, label_image)


            # ----------------- overlap tile-------------------------------------
            data_image_LR = data_image.transpose(Image.FLIP_LEFT_RIGHT)  # left to right

            # convert images to array
            data_image = np.asarray(data_image, dtype=np.float32) / 255.
            data_image_LR = np.asarray(data_image_LR, dtype=np.float32) / 255.

            # tile
            center_row = np.hstack((data_image_LR, data_image, data_image_LR))
            TB_row = np.flipud(center_row)
            tile = np.vstack((TB_row, center_row, TB_row))

            tile_crop = (3 * self.output - self.input) // 2
            data_image = tile[tile_crop:tile.shape[0] - tile_crop, tile_crop:tile.shape[1] - tile_crop]

            # ----------------------------------------------------------------

            label_image = np.asarray(label_image, dtype=np.float32)


            current += 1
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def applyDataAugmentation(self, image, label):
        w, h = image.size

        # gamma
        # select= random.randint(0,1)
        # if select:
        range = random.uniform(0.5, 1.5)
        image = TF.adjust_gamma(image, gamma=range, gain=.5)

        # crop
        # select = random.randint(0, 1)
        # if select:

        range = random.randint(250, 300)
        image = TF.resized_crop(image, 10, 10, range, range, (w,h))
        label = TF.resized_crop(label, 10, 10, range, range, (w, h))

        # flip
        select = random.randint(0, 4)
        if select>0:
            image = TF.hflip(image)
            label = TF.hflip(label)

        select = random.randint(0, 4)
        if select>0:
            image = TF.vflip(image)
            label = TF.vflip(label)




        # Enhance image
        # image_enhanced = cv2.equalizeHist(image)
        # image=  Image.new("RGBA", image.size)
        # image = TF.adjust_contrast(image,2)

        # flip
        # =============================================================================
        #         select= random.randint(0,1)
        #         print("select flip=", select)
        #         if select:
        #             image=image[:, ::-1]
        #             label= label[:, ::-1]

        # =============================================================================
        # crop
        # select= random.randint(0,1)
        # print("select crop=", select)
        # if select:
        #
        #     random_select= random.randint(20,50)
        #     image= resize(util.crop(image,random_select),(572,572))
        #     label = resize(util.crop(label,random_select),(388,388))
        #
        #
        # #gamma
        #
        # select = random.randint(0, 1)
        # print("select gamma=", select)
        # if select:
        #     random_select= random.uniform(.5, 1.5)
        #     image= exposure.adjust_gamma(image, gamma=random_select, gain=1)
        #     label = exposure.adjust_gamma(label, gamma=random_select, gain=1)
        #
        # #invert
        # select = random.randint(0, 1)
        # print("select invert=", select)
        # if select:
        #
        #     random_select=random.randint(-1,1)
        #     if random_select:
        #         image= util.invert(image)
        #         label =util.invert(label)

        # intensity
        # random_select = random.randint(0, 1)
        # print("select intense=", select)
        # if random_select:
        #     v_min, v_max = np.percentile(image, (.1, 99.8))
        #     image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        #     v_min, v_max = np.percentile(label, (1, 99.8))
        #     label = exposure.rescale_intensity(label, in_range=(v_min, v_max))

        # Zoom
        # random_select= random.randint(0,1)
        # if random_select==1:
        #     random_crop = random.randint(5, 10)
        #     width, height = image.size
        #     image = ImageOps.crop(image, image.size[1] // random_crop).resize((width, height))
        #     label = ImageOps.crop(label, label.size[1] // random_crop).resize((width, height))

        # random_flip = random.randint(-1, 5)  # -1 is None
        # image = image.transpose(random_flip) if random_flip > -1 else image
        # label = label.transpose(random_flip) if random_flip > -1 else label
        # Flip
        # method = random.randint(-1, 5)
        # if method != -1:
        #
        #     image = image.transpose(method)
        #     label = label.transpose(method)
        #
        # else:
        #     image = image

        return image, label

