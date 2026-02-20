"""
utils.py
Shared visualisation and image helper functions used across
notebooks, training, and inference.
"""
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import re


def get_class_names(target_dir):
    """
    Returns a list of class names from the target_dir.

    :param target_dir: The target directory within which the different classes are stored
    """
    class_names = os.listdir(target_dir)
    class_names = sorted(class_names, key=lambda x: int(re.match(r'^\d+', x).group()))
    return class_names


def view_random_image(target_dir, class_names):
    """
    Plots a random image from each class in class_names located in target_dir.
    Also prints the shape of each image.

    :param target_dir: The target directory within which the different classes are stored
    :param class_names: The list of class names to be plotted
    """
    num_classes = len(class_names)
    if num_classes == 7:
        rows, cols = 1, 7
    elif num_classes == 35:
        rows, cols = 5, 7
    else:
        raise ValueError(f"Expected 7 or 35 classes, got {num_classes}")

    fig_width  = cols * 3
    fig_height = rows * 3

    plt.figure(figsize=(fig_width, fig_height))

    for i, class_name in enumerate(class_names):
        target_folder = os.path.join(target_dir, class_name)
        random_image = random.sample(os.listdir(target_folder), 1)[0]
        img = mpimg.imread(os.path.join(target_folder, random_image))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(class_name, fontsize=8)
        print(f"Image shape for {class_name}: {img.shape}")

    plt.tight_layout()
    plt.show()

def return_random_image(target_dir, class_name):
    """
    Takes a random image from the target_dir/class_name folder. 
    Returns the image as tensorflow tensor and its shape. 

    :param target_dir: The target directory within which the different classes are stored
    :param class_name: The class name for which a random image will be produced
    """
    target_folder = os.path.join(target_dir, class_name)
    random_image = random.sample(os.listdir(target_folder), 1)[0]
    img = mpimg.imread(os.path.join(target_folder, random_image))
    return tf.constant(img), img.shape


def read_rescale(filepath, shape):
    '''
    Takes image from filepath and reshapes it to a 4-dim tensor
    for appropriate input into model prediction. Also scales.

    :param filepath: The path of the image file to be read and rescaled
    '''
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=3)
    print(f"Initial image size: {img.shape}")
    img = tf.image.resize(img, shape)
    img = img/255.
    img = tf.expand_dims(img, axis=0)
    print(f"Final image size: {img.shape}")
    return img


def pred_plot(model, filename, class_names, shape):
    """
    Imports an image located at filename, makes a prediction with model
    and plot the image with predicted class as title

    :param model: The model with which to predict the class of the image
    :param filename: The image's file name
    :param class_names: The classes within which the image will be classified
    :param shape: The shape to which we want to reshape the image to
    """
    img = read_rescale(filename, shape=shape)
    pred = model.predict(img)
    pred_class = class_names[pred.argmax()]
    pred_class = pred_class.capitalize()
    print(f"The predicted class for {filename} is {pred_class}")
    plt.imshow(tf.squeeze(img))
    plt.axis(False)
    plt.title(label = f"Predicted class: {pred_class}")
    plt.show()