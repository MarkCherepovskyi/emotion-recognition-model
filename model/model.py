import cv2
from pathlib import Path
import os
import tensorflow as tf

from  tensorflow import keras
import numpy as np

import logging


class Data:
    def __init__(self, base_path:  Path ):
        self.classes = self.get_classes(base_path)
        self.links = self.get_files_links(base_path)

    def load_img(self, path: Path):
        return cv2.imread(str(path))

    def get_files_links(self,  base_path) -> list[(Path, int)]:
        return [(base_path/folder/item,  i) for i, folder in  enumerate(os.listdir(base_path)) for  item in os.listdir(base_path/folder)]

    def load_files(self, index: int, batch_size: int):
        start_index = index * batch_size
        end_index = start_index + batch_size
        return [(self.load_img(link[0]), link[1]) for link in self.links[start_index:end_index]]

    def get_classes(self,  base_path: Path):
        return [name for name in os.listdir(base_path) if os.path.isdir(base_path / Path(name))]


class CustomModelDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        base_path: Path,
        shuffle: bool = False,
    ) -> None:


        self._batch_size = 16
        self._shuffle = shuffle
        self._base_path = base_path
        self._data = Data(base_path=Path('./dataset/train'))
        self._data.get_files_links(base_path=Path('./dataset/train'))
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Shuffles the paths to the images at the end of each epoch if shuffling is enabled.
        """

        if self._shuffle:
            np.random.shuffle(self._data)

    def __len__(self):
        """
        Returns the number of batches per epoch.

        ### Returns:
        - int: The number of batches in the dataset for each epoch.
        """

        return len( self._data.links) // self._batch_size

    def __getitem__(self, index):
        batch_data = self._data.load_files(index, self._batch_size)
        images = np.array([data[0] for data in batch_data])
        labels = np.array([data[1] for data in batch_data])
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(
            self._data.classes))
        return images, labels


class Model:
    def __init__(self):
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((48, 48, 3), input_shape=(48, 48, 3)),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(7, activation='softmax')  # Use softmax activation for multi-class classification
        ])

        self._model.compile(optimizer='adam',
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=['accuracy'])

        self._generator = CustomModelDataGenerator('./dataset/train')
        self._model.summary()

        self._is_modal_loaded = False

    def fit(self):
        self._model.fit(self._generator, epochs=70)

        print("fited")
        self._model.save('./test.h5')
        print("saved")


    def evaluate(self, x_test, y_test):
        self._model.evaluate(x_test, y_test, verbose=2)

    def load_modal(self):
        if self._is_modal_loaded: return
        self._model.load_weights("./test.h5")
        self._is_modal_loaded = True

    def predict(self,  path:  Path):
        self.load_modal()
        img = cv2.imread(str(path))
        img = cv2.resize(img, (48, 48))

        # Reshape the image to add the batch dimension
        img = np.expand_dims(img, axis=0)
        res =  self._model.predict(img)
        print(res)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to only display errors
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to only display errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Set TensorFlow logger to only display errors

# Initialize the model and fit it
model = Model()
model.fit()
model.predict('/home/dl/Desktop/Projects/AI/emotion_detection/dataset/test/fearful/im1.png')