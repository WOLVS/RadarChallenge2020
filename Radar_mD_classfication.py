"""
This kernel is used to classify radar signals.

Author: Shufan Yang, shufany@gmail.com
Date: 25/10/2019
"""

import keras
from keras.models import load_model
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import shutil
from tqdm import tqdm
from random import seed
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import click
import time
import json
from scipy.stats import binom

from sf_mltool.sf_mltool import Sf_mltool
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

GENERAL_IMAGE_OFFSET = 5

MODEL_NAMES = {
    'NM': 'NASNetMobile.h5',
    'NL': 'NASNetLarge.h5',
    'IR2': 'InceptionResNetV2.h5',
    'XC': 'Xception.h5',
    'D21': 'DenseNet201.h5',
    'IV3': 'InceptionV3.h5',
    'SC': 'SimpleCNN.h5',
}

system_config = {
    "Author": "Shufan Yang",
    "Email": "sfyang@gmail.com",
    "Version": "1.0.0",
    "first_model": "NM",
    "second_model": "IV3",
    "IR2_Accuracy": 0.96,
    "NL_Accuracy": 0.96,
    "XC_Accuracy": 0.96,
    "D21_Accuracy": 0.96,
    "IV3_Accuracy": 0.96,
    "crop_size": 256,
    "image_wh": 128,
    "first_thd_offset": 2,
    "first_confidence": 0.99,
    "second_thd_offset": 2,
    "second_confidence": 0.80,
}

launch_folder = Path(os.getcwd())
script_folder = Path(os.path.dirname(os.path.realpath(__file__)))


def create_tv_list(folder, ext='jpg', split=0.10):
    """
    This function get all file names from the folder and creates two lists for
    training and validation

    If a file name contains 'LevelA', its classification mark is '1'; If a file name
    contains 'LevelB', its classification mark is '0'.

    Example:
    train_list, valid_list = create_tv_list(source_folder.joinpath('label'))
    train_df = pd.DataFrame(train_list,columns=['fname','class'])

    :param folder: The folder holding all images
    :param ext: image file extension
    :param split: train, validation split ratio
    :return: train_list, valid_list
    """
    # Get a file list and classification
    labels = set()
    flist = []
    files = Path(folder).glob(f'**/*.{ext}')
    for i in files:
        temp = i.stem
        flag = 'Dataset'
        label_idx = temp.index(flag)
        label_idx = label_idx + len(flag)

        flist.append([str(i), temp[label_idx]])
        labels.add(temp[label_idx])
    pass

    train_list = []
    valid_list = []
    for i in labels:
        temp = [element for element in flist if element[1] == i]
        temp = shuffle(temp, random_state=seed())
        tl, vl = train_test_split(temp, test_size=split)
        train_list = train_list + tl
        valid_list = valid_list + vl

    return train_list, valid_list


def create_ml_list(folder, ext='jpg', split=0.33):
    """
    This function get all file names from the folder and creates three lists for
    machine learning (ml).

    If a file name contains 'LevelA', its classification mark is '1'; If a file name
    contains 'LevelB', its classification mark is '0'.

    Example:
    train_list, valid_list, test_list = create_ml_list(source_folder.joinpath('label'))
    train_df = pd.DataFrame(train_list,columns=['fname','class'])

    :param folder: The folder holding all images
    :param ext: image file extension
    :param split: train, validation, test dataset split ratio
    :return: train_list, valid_list, test_list
    """
    # Get a file list and classification
    flist = []
    files = Path(folder).glob(f'**/*.{ext}')
    for i in files:
        if 'LevelA' in i.stem:
            flist.append([str(i), '1'])
        elif 'LevelB' in i.stem:
            flist.append([str(i), '0'])
        else:
            pass
    pass

    # shuffle flist
    flist = shuffle(flist, random_state=seed())
    train_list, temp_list = train_test_split(flist, test_size=split)
    test_list, valid_list = train_test_split(temp_list, test_size=split)

    #print(f'flist size: {len(flist)}, train size: {len(train_list)}, valid size: {len(valid_list)}, test size: {len(test_list)}')
    return train_list, valid_list, test_list


def extract_radar_image(folder, fname):
    """
    This function read the image with fname and extract, or crop marked areas into images to be trained,
    validated, or tested for machine learning.

    According to the marks, these marked areas can be extracted into an image of class '1' if
    it is marked as 'LevelA', or an image of class '0' if it is marked as 'LevelB'.

    :note A: folder should exist, function could throw exception
    :param folder: destination folder where new extracted files are saved
    :param fname: marked image file name
    :return: extracted images in folder
    """
    extracted_image_wh = system_config['image_wh']

    # Create new image
    with Image.open(fname) as ifd:
        ws, hs = ifd.size
        # Crop image from original radar image with fixed offset and length
        xstart = 100
        ystart = 200
        box = [xstart, ystart, xstart + 590, ystart + 250]
        if box[2] < (ws - 100) and box[3] < (hs - 50):
            # Crop it to designated size
            ofd = ifd.crop(box)
            ofwsize, oflsize = ofd.size
            if ofwsize != extracted_image_wh or oflsize != extracted_image_wh:
                ofd = ofd.resize((extracted_image_wh, extracted_image_wh), Image.HAMMING)

            # Save it
            cvofd = np.asarray(ofd)
            cvofd = cv2.cvtColor(cvofd, cv2.COLOR_BGR2GRAY)
            ofd = Image.fromarray(cvofd)
            ofname = folder.joinpath(f'{fname.stem}.jpg')
            ofd.save(ofname)


def extract_images(dst_folder, src_folder, extracted_image_wh=128, ext='png'):
    """
    Extract all marked images in src_folder and save them into dst_folder.

    Example:

    src = Path('C:/Users/Desktop/ml/ml/ImgTree/0614-1')
    dst = Path('C:/Users/Desktop/ml/ml/ImgTree/test')

    extract_images(dst, src)

    :param dst_folder: full path destination folder for extracted images
    :param src_folder: full path source folder where keeps the original images
    :param extracted_image_wh: extracted image size
    :param ext: image file extension
    :return: None
    """
    if extracted_image_wh != system_config['image_wh']:
        system_config['image_wh'] = extracted_image_wh
        save_config()

    # Remove previous files
    shutil.rmtree(str(dst_folder), ignore_errors=True)
    dst_folder.mkdir(exist_ok=True)

    # Copy region of interest according to label
    files = list(Path(src_folder).glob('**/*.' + ext))
    total = len(files)
    with tqdm(total=total) as pbar:
        for i in files:
            try:
                extract_radar_image(dst_folder, i)
            except:
                print(f'file extraction error: {str(i)}')
                pass
            pbar.set_description(str(i))
            pbar.update(1)


def data_generator():
    """
    This augments image in real time.
    :return: image data generator
    """
    return ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=True,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0.0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.0,
        # set range for random zoom
        zoom_range=0.2,
        # set range for random channel shifts
        channel_shift_range=0.0,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def simple_cnn(input_shape=None, classes=0):
    """
    Simple CNN to train Radar network
    :param input_shape:
    :return:
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model

def unet_cnn(input_shape=None, classes=0):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Reshape((14, 14, 8)))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

def radar_train(folder, batch_size, epoch_size, model_name):
    """
    Train network with the parameters specified.

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 1)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model = None
    if 'NM' in model_name:
        model_name = 'NM'
        model = NASNetMobile(include_top=True,
                             weights=None,
                             input_tensor=None,
                             input_shape=image_shape,
                             pooling='max',
                             classes=6)
    elif 'NL' in model_name:
        model_name = 'NL'
        model = NASNetLarge(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'XC' in model_name:
        model_name = 'XC'
        model = Xception(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=image_shape,
                         pooling='max',
                         classes=6)
    elif 'D21' in model_name:
        model_name = 'D21'
        model = DenseNet201(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'IV3' in model_name:
        model_name = 'IV3'
        model = InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'SC' in model_name:
        model_name = 'SC'
        model = unet_cnn(input_shape=image_shape,
                           classes=6)
    else:
        model_name = 'IR2'
        model = InceptionResNetV2(include_top=True,
                                  weights=None,
                                  input_tensor=None,
                                  input_shape=image_shape,
                                  pooling='max',
                                  classes=6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'
    print(f'{filepath}\n')

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing=False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def autoencoder_train(folder, batch_size, epoch_size, model_name):
    """
    Autoencoding, inherently UNET, is a data compression algorithm where the compression and decompression functions are:
    - data specific, ie, only compress data similar to what they have been trained on
    - lossy, ie, decompressed output will be degraded
    - learned automatically from examples.

    Two practical applications of autoencoders are data removal and dimensionality reduction

    There is an implementation from scikit-learn:
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 1)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model = None
    if 'NM' in model_name:
        model_name = 'NM'
        model = NASNetMobile(include_top=True,
                             weights=None,
                             input_tensor=None,
                             input_shape=image_shape,
                             pooling='max',
                             classes=6)
    elif 'NL' in model_name:
        model_name = 'NL'
        model = NASNetLarge(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'XC' in model_name:
        model_name = 'XC'
        model = Xception(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=image_shape,
                         pooling='max',
                         classes=6)
    elif 'D21' in model_name:
        model_name = 'D21'
        model = DenseNet201(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'IV3' in model_name:
        model_name = 'IV3'
        model = InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=image_shape,
                            pooling='max',
                            classes=6)
    elif 'SC' in model_name:
        model_name = 'SC'
        model = simple_cnn(input_shape=image_shape,
                           classes=6)
    else:
        model_name = 'IR2'
        model = InceptionResNetV2(include_top=True,
                                  weights=None,
                                  input_tensor=None,
                                  input_shape=image_shape,
                                  pooling='max',
                                  classes=6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'
    print(f'{filepath}\n')

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing=False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def transfer_learn(folder, batch_size, epoch_size, model_name):
    """
    Transfer learning is a procedure with pre-trained models.

    :param folder: image folder for training
    :param batch_size: training batch size
    :param epoch_size: training epoch size
    :param model_name: IR2, InceptionResNetV2; NL, NASNetLarge; NM, NASNetLarge
    :return: None
    """
    image_wh = system_config['image_wh']

    image_size = (image_wh, image_wh)
    image_shape= (image_wh, image_wh, 1)

    train_list, valid_list = create_tv_list(folder)
    print(f'Train size: {len(train_list)}, valid size: {len(valid_list)}')

    train_df = pd.DataFrame(train_list, columns=['fname', 'class'])
    valid_df = pd.DataFrame(valid_list, columns=['fname', 'class'])

    model_fname = f'models/{MODEL_NAMES[model_name]}'
    model_fname = script_folder.joinpath(model_fname)
    model = load_model(str(model_fname), compile=False)

    # Set first 50 layer non trainable
    for layer in model.layers[50:]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Image generator does data augmentation:
    datagen = data_generator()

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    valid_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=folder,
        x_col="fname",
        y_col="class",
        class_mode="categorical",
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False)

    # Prepare model model saving directory.
    save_dir = Path(os.path.dirname(os.path.realpath(__file__))).joinpath('models')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)
    filepath = f'{str(save_dir)}/{MODEL_NAMES[model_name]}'

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    steps_per_epoch = int(len(train_list)/batch_size)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=steps_per_epoch,
        epochs=epoch_size,
        use_multiprocessing = False,
        verbose=1,
        workers=4,
        callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate_generator(generator=valid_gen, steps=steps_per_epoch, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save score in configuration file
    system_config[f'{model_name}_Accuracy'] = scores[1]
    save_config()

    return history


def load_config(fname='radar.json'):
    """
    Configuration file is in JSON format.

    :param fname: Configuration file name
    :return: system configuration
    """
    config = {}
    try:
        with open(script_folder.joinpath(fname)) as f:
            config = json.load(f)
    except:
        # Create default configuration file
        pass
    finally:
        print(config)

    return config


def save_config(fname='radar.json'):
    """
    save configuration file is in JSON format.

    :param fname: Configuration file name
    :return:
    """
    try:

        with open(script_folder.joinpath(fname), 'w', encoding='utf-8') as f:
            json.dump(system_config, f, ensure_ascii=False, indent=4)
    except:
        # Create default configuration file
        pass


def decision_threshold(overall=0.99, accuracy=0.95, samples=1000):
    """
    Calculate the decision threshold. This is based on Binomial Distribution.
    :param overall: Certainty we can sure the image is LevelA
    :param accuracy: Accuracy of predicting levelB is LevelB
    :param samples: How many sub images in total
    :return: integer value; if more than this value of sub images are LevelA, this image is LevelA
    """
    k = binom.ppf(overall, samples, 1-accuracy)
    return int(k)




def radar_verify(image_folder='verify', ext='jpg'):
    """
    verification of the software module to classify images in a folder

    :param image_folder: verification file folder holding images
    """
    # Prepare manual check folder
    '''
    Check folder architecture is:

    check/
         first/
         second/
    '''
    check_dir = script_folder.joinpath('check')
    if not check_dir.is_dir():
        check_dir.mkdir(exist_ok=True)

    first_dir = check_dir.joinpath('first')
    if not first_dir.is_dir():
        first_dir.mkdir(exist_ok=True)

    second_dir = check_dir.joinpath('second')
    if not second_dir.is_dir():
        second_dir.mkdir(exist_ok=True)

    # Load model, probably find a latest file in saved model folder
    model_indexes = ('IV3', 'NM')
    # first prediction model
    model_fname = f'models/{MODEL_NAMES[model_indexes[0]]}'
    model_fname = script_folder.joinpath(model_fname)
    model_first = load_model(str(model_fname), compile=False)
    # second prediction model
    model_fname = f'models/{MODEL_NAMES[model_indexes[1]]}'
    model_fname = script_folder.joinpath(model_fname)
    model_second = load_model(str(model_fname), compile=False)

    # Image generator does data augmentation
    datagen = data_generator()
    folder = script_folder.joinpath(image_folder)
    files = folder.glob(f'**/*.{ext}')
    for i in files:
        image = cv2.imread(str(i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        isLevelA = double_check(image, model_first, model_second, datagen)
        print(f'{i.name} image class is A: {isLevelA}')
    pass


def infer_live(camera_interface=0, debug=False):
    """
    Infer the class of an iPad from captured image. For one image, 2 different
    models are used to predict its class. This ensures prediction accuracy.

    :param camera_interface: camera interface index
    :param debug: Debug flag
    :return: True, is LevelA; False not.
    """
    save_dir = script_folder.joinpath('predicted')
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True)

    # Load model, probably find a latest file in saved model folder
    model_indexes = ('NM', 'IV3')
    # first prediction model
    model_fname = f'models/{MODEL_NAMES[model_indexes[0]]}'
    model_fname = script_folder.joinpath(model_fname)
    model_first = load_model(str(model_fname), compile=False)
    # second prediction model
    model_fname = f'models/{MODEL_NAMES[model_indexes[1]]}'
    model_fname = script_folder.joinpath(model_fname)
    model_second = load_model(str(model_fname), compile=False)

    # Image generator does data augmentation
    datagen = data_generator()

    # Open camera
    video_capture = cv2.VideoCapture(camera_interface)
    # Check success
    if not video_capture.isOpened():
        raise IOError("Could not open video device")

    # Read picture. ret === True on success
    while True:
        # Wait for signal from external device or a socket server?
        try:
            isLevelA = True
            ret, image = video_capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if ret:
                isLevelA = double_check(image, model_first, model_second, datagen)
                print(f'image class is A: {isLevelA}')

                if debug:
                    # Keep files for later analysis
                    ts = time.time()
                    label = 'LevelA' if isLevelA else 'LevelB'
                    fname = f'{ts}_{label}.jpg'

                    ofd = Image.fromarray(image)
                    ofd.save(fname)

                # Decision
                if isLevelA is True:
                    #this one is LevelA
                    pass

            else:
                print(f'Cannot get photo')
                time.sleep(1)

        except:
            # Just sleep one second
            time.sleep(1)
            pass

    # Close device
    video_capture.release()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def radar_ic_cli():
    pass


@radar_ic_cli.command()
@click.option('-s', '--src-folder', required=True, help='source image folder, relative to current folder')
@click.option('-d', '--dst-folder', default='radarExtracted', help='destination image folder, relative to current folder')
@click.option('-i', '--image-size', default=128, type=int, show_default=True, help='extracted image size resized from cropped image size')
def extract(**kwargs):
    src = script_folder.joinpath(kwargs['src_folder'])
    dst = script_folder.joinpath(kwargs['dst_folder'])
    extract_images(dst, src, kwargs['image_size'])


@radar_ic_cli.command()
@click.option('-s', '--image-folder', default='radarExtracted', help='image folder for training, relative to current folder')
@click.option('-b', '--batch-size', default=128, type=int, show_default=True, help='batch size')
@click.option('-e', '--epoch-size', default=500, type=int, show_default=True, help='epoch size')
@click.option('-m', '--model-name', default='IR2', show_default=True, help='IR2, NM, IV3, XC, D21')
def train(**kwargs):
    folder = script_folder.joinpath(kwargs['image_folder'])
    history = radar_train(folder,
                        kwargs['batch_size'],
                        kwargs['epoch_size'],
                        kwargs['model_name'])

    # summarize history for accuracy
    ml_tool = Sf_mltool()
    ml_tool.draw_performance(history=history, nname=kwargs['model_name'])


@radar_ic_cli.command()
@click.option('-s', '--image-folder', default='radarExtracted', help='image folder for training, relative to current folder')
@click.option('-b', '--batch-size', default=128, type=int, show_default=True, help='batch size')
@click.option('-e', '--epoch-size', default=500, type=int, show_default=True, help='epoch size')
@click.option('-m', '--model-name', default='IR2', show_default=True, help='IR2, NM, IV3, XC, D21')
def transfer(**kwargs):
    folder = script_folder.joinpath(kwargs['image_folder'])
    history = transfer_learn(folder,
                             kwargs['batch_size'],
                             kwargs['epoch_size'],
                             kwargs['model_name'])

    # summarize history for accuracy
    # ml_tool = Sf_mltool()
    # ml_tool.draw_performance(history=history, nname=kwargs['model_name'])


@radar_ic_cli.command()
@click.option('-f', '--camera-interface', default=0, type=int, show_default=True, help='camera device interface')
@click.option('-d', '--debug', is_flag=True, default=False, show_default=True, help='debug output of captured image and cropped images')
def infer(**kwargs):
    infer_live(kwargs['camera_interface'],
               kwargs['debug'])


@radar_ic_cli.command()
@click.option('-s', '--image-folder', required=True, help='image verification folder')
def verify(**kwargs):
    radar_verify(kwargs['image_folder'])


if __name__ == '__main__':
    # Load and merge configuration
    config = load_config()
    if config != system_config:
        system_config = {**system_config, **config}
        save_config()

    # Execute commands
    radar_ic_cli()

    # TODO:
    # Verify camera interface
    # input and output interface
