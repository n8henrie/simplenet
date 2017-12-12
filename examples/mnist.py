"""MNist data loader.

Much of this taken from:
- https://github.com/hdmetor/NeuralNetwork/blob/master/data_load.py
- http://martin-thoma.com/classify-mnist-with-pybrain/
"""

import gzip
import os
import time
from struct import unpack
from typing import Tuple
from urllib.request import urlopen

import numpy as np

import simplenet as sn

np.set_printoptions(precision=10, suppress=True)

FOLDER = 'data/mnist'

LECUN = 'http://yann.lecun.com/exdb/mnist/'


def get_images_and_labels(train_or_test: str, folder: str = FOLDER) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Read input-vector and target class and return it as list of tuples.

    Arguments:
        train_or_test: Whether to get "train" or "test" data
        folder: folder that holds data files
    Returns:
        Tuple of (input, target) as ndarray
    """
    urlmap = {
            'train': {
                "data": 'train-images-idx3-ubyte.gz',
                "labels": 'train-labels-idx1-ubyte.gz'
            },
            'test': {
                "data": 't10k-images-idx3-ubyte.gz',
                "labels": 't10k-labels-idx1-ubyte.gz'
            },
        }

    if not os.path.exists(folder):
        os.makedirs(folder)

    for dataset in urlmap[train_or_test].values():
        check_or_download(dataset, data_folder=folder)

    return (read_images(urlmap[train_or_test]["data"], folder),
            read_labels(urlmap[train_or_test]["labels"], folder))


def read_images(file_name: str, data_folder: str) -> np.ndarray:
    """Unzip and unpack image data.

    Arguments:
        file_name: Name of file
        data_folder: Folder containing `file_name`
    Returns:
        ndarray of image data
    """
    file_location = os.path.join(data_folder, file_name)
    with gzip.open(file_location, 'rb') as images:
        images.read(4)
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]
        x = np.zeros((number_of_images, rows, cols), dtype=np.uint8)

        for i in range(number_of_images):
            if i % int(number_of_images / 10) == \
                    int(number_of_images / 10) - 1:
                print("Reading images progress ",
                      int(100 * (i + 1) / number_of_images), "%")
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = tmp_pixel

    return x


def read_labels(file_name: str, data_folder: str) -> np.ndarray:
    """Unzip and unpack image labels.

    Arguments:
        file_name: Name of file
        data_folder: Folder containing `file_name`
    Returns:
        ndarray of image labels
    """
    file_location = os.path.join(data_folder, file_name)
    with gzip.open(file_location, 'rb') as labels:
        labels.read(4)
        number_of_labels = labels.read(4)
        number_of_labels = unpack('>I', number_of_labels)[0]
        y = np.zeros((number_of_labels, 1), dtype=np.uint8)

        for i in range(number_of_labels):
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]

    return y


def check_or_download(file_name: str, data_folder: str, url: str = LECUN) \
        -> None:
    """Download the data if it isn't already present.

    Arguments:
        file_name: Name of file
        data_folder: Folder containing `file_name`
    """
    file_location = os.path.join(data_folder, file_name)
    if not os.path.exists(file_location):
        print("Downloading ", file_name)

        with urlopen(url + file_name) as req, open(file_location, 'wb') as fp:
            fp.write(req.read())


def main(import_progress: str = None, save_progress: str = None) -> None:
    """Download mnist and test a simple MLP.

    Arguments:
        import_progress: If given, import weights from here
        save_progress: If given, save weights to here (epoch number will be
                       appended to the filename)
    """
    print("Starting...")
    try:
        data = np.load(FOLDER + '/data.npz')
        X_train, y_train, X_test, y_test = (v for _, v in sorted(data.items()))
    except FileNotFoundError:

        print("Gathering the training data")
        X_train, y_train = get_images_and_labels('train', folder=FOLDER)

        errmsg = "Train images were loaded incorrectly"
        assert (X_train.shape, y_train.shape) == ((60000, 28, 28),
                                                  (60000, 1)), errmsg
        X_train = X_train.reshape(60000, 784)  # noqa

        print("Gathering the test data")
        X_test, y_test = get_images_and_labels('test', folder=FOLDER)

        errmsg = "Test images were loaded incorrectly"
        assert (X_test.shape, y_test.shape) == ((10000, 28, 28),
                                                (10000, 1)), errmsg
        X_test = X_test.reshape(10000, 784)  # noqa

        np.savez(FOLDER + '/data', X_train, y_train, X_test, y_test)

    y_train = np.eye(10)[y_train.reshape(y_train.shape[0],)]
    y_test = np.eye(10)[y_test.reshape(y_test.shape[0],)]

    nn = sn.SimpleNet(
        input_shape=(None, 784),
        output_shape=(None, 10),
        hidden_layer_sizes=(256, 128, 32),
        activation_function=sn.relu,
        learning_rate=0.008,
        output_activation=sn.softmax,
        loss_function=sn.cross_entropy,
        seed=42,
        )

    epochs = 5
    batch_size = 200

    if import_progress:
        nn.import_model(import_progress)

    print('batch size {}'.format(batch_size))
    times = []
    for e in range(epochs):
        batch_errors = []
        start = time.time()
        for idx in range(0, X_train.shape[0], batch_size):
            nn.learn(inputs=X_train[idx:idx+batch_size],
                     targets=y_train[idx:idx+batch_size])
            batch_err = nn.err
            batch_errors.append(batch_err)

        times.append(time.time() - start)
        epoch_err = sum(batch_errors) / len(batch_errors)
        print('epoch {}: {} seconds, cost: {}'.format(e, times[-1], epoch_err))

        train_acc = np.sum(
                nn.predict(X_train).argmax(axis=1) == y_train.argmax(axis=1)
                ) / X_train.shape[0]
        print('train_accuracy:', train_acc)

        test_acc = np.sum(
                nn.predict(X_test).argmax(axis=1) == y_test.argmax(axis=1)
                ) / X_test.shape[0]
        print('test_accuracy:', test_acc, '\n')

        if save_progress:
            nn.export_model(f"{save_progress}_{e}")

    print("Total time: {}".format(sum(times)))


if __name__ == "__main__":
    main()
