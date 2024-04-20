import os
from PIL import Image
import numpy as np
import chardet
from keras.preprocessing.image import ImageDataGenerator


def get_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def get_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def get_labels(path):
    encoding = get_encoding(path)
    with open(path, 'r', encoding=encoding) as f:
        labels = f.read().splitlines()
    labels = [list(map(float, label.split())) for label in labels]
    return labels


def conversion(path):
    img = Image.open(path)
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    return img_array


def get_data_generators(train_image_dir, train_label_dir, test_image_dir, test_label_dir, batch_size):
    train_image_paths = get_paths(train_image_dir)
    train_label_paths = get_paths(train_label_dir)
    get_paths(test_image_dir)
    get_paths(test_label_dir)

    image_datagen = ImageDataGenerator(rescale=1. / 255)
    label_datagen = ImageDataGenerator()

    train_image_generator = image_datagen.flow_from_directory(
        train_image_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode=None,
    )

    train_label_generator = label_datagen.flow_from_directory(
        train_label_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode=None,
        color_mode="grayscale",
    )

    train_generator = zip(train_image_generator, train_label_generator)

    val_samples = int(0.2 * len(train_image_paths))
    validation_generator = zip(
        train_image_generator.flow(
            np.array(train_image_paths[:-val_samples]),
            batch_size=batch_size,
            shuffle=False,
        ),
        train_label_generator.flow(
            np.array(train_label_paths[:-val_samples]),
            batch_size=batch_size,
            shuffle=False,
        ),
    )

    test_generator = zip(
        image_datagen.flow_from_directory(
            test_image_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode=None,
        ),
        label_datagen.flow_from_directory(
            test_label_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode=None,
            color_mode="grayscale",
        ),
    )

    return train_generator, validation_generator, test_generator
