import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def origin_image_plot(train_dataset):
    """
    Plot the original images from the dataset.

    Args:
        train_dataset: The dataset containing the images and labels.

    Returns:
        fig1: The figure containing different disease images.
        fig2: The figure containing the healthy image.
    """
    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

    labels = [0, 1, 2, 3, 4]
    labels_name = {
        "0": "Cassava Bacterial Blight (CBB)",
        "1": "Cassava Brown Streak Disease (CBSD)",
        "2": "Cassava Green Mottle (CGM)",
        "3": "Cassava Mosaic Disease (CMD)",
        "4": "Healthy",
    }
    found_images = {}

    for image, label in train_dataset:
        label = label.numpy()
        if label in labels and label not in found_images:
            found_images[label] = image
            labels.remove(label)
        if len(labels) == 0:
            break

    # plot the healthy image
    ax2.imshow(found_images[4])
    ax2.set_xlabel(labels_name["4"], fontsize=25)
    # ax2.set_title("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    found_images.pop(4)

    # plot the rest of the images
    for i, (label, image) in enumerate(found_images.items()):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(image)
        axes[row, col].set_xlabel(labels_name[str(label)], fontsize=15)
        # axes[row, col].set_title("")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].spines["top"].set_visible(False)
        axes[row, col].spines["bottom"].set_visible(False)
        axes[row, col].spines["left"].set_visible(False)
        axes[row, col].spines["right"].set_visible(False)

    return fig1, fig2


def learning_curve(history):
    """
    Plot the learning curve of the model.

    Args:
        history: The history object returned by the model's fit method.

    Returns:
        fig: The figure containing the learning curve.
    """
    history_frame = pd.DataFrame(history.history)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    epochs = range(1, len(history_frame) + 1)

    ax.plot(epochs, history_frame["loss"], label="Training Loss")
    ax.plot(epochs, history_frame["val_loss"], label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xticks(range(1, len(history_frame) + 1, 2))

    return fig


def plot_cive_result(dataset):
    """
    Plot the original image, mask image, and processed image.

    Args:
        dataset: The dataset containing the images and masks.

    Returns:
        fig: The figure containing the original image, mask image, and processed image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    for image, mask in dataset.skip(14).take(1):
        # for image, mask in dataset.take(3):
        axes[0].imshow(image)
        axes[0].set_xlabel("Original Image", fontsize=15)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["bottom"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].imshow(mask)
        axes[1].set_xlabel("Mask Image", fontsize=15)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["bottom"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        image_pro = image * mask
        axes[2].imshow(image_pro)
        axes[2].set_xlabel("Processed Image", fontsize=15)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["bottom"].set_visible(False)
        axes[2].spines["left"].set_visible(False)
        axes[2].spines["right"].set_visible(False)

    return fig


def origin_Unet_result_plot(origin_image: list, mask_image: list, image_name: list):
    """
    Plot the original image and the processed image including all the diseases.

    Args:
        origin_image: The original images.
        mask_image: The mask images.
        image_name: The names of the images.

    Returns:
        fig: The figure containing the original image and the processed image.
    """
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for i in range(4):
        axes[0, i].imshow(origin_image[i])
        axes[0, i].set_xlabel(image_name[i], fontsize=15)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].spines["top"].set_visible(False)
        axes[0, i].spines["bottom"].set_visible(False)
        axes[0, i].spines["left"].set_visible(False)
        axes[0, i].spines["right"].set_visible(False)

        axes[1, i].imshow(origin_image[i] * mask_image[i])
        axes[1, i].set_xlabel(f"Processed {image_name[i]}", fontsize=15)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].spines["top"].set_visible(False)
        axes[1, i].spines["bottom"].set_visible(False)
        axes[1, i].spines["left"].set_visible(False)
        axes[1, i].spines["right"].set_visible(False)
        plt.tight_layout(pad=0.3)

    return fig


def classfication_result(y_true, y_pred):
    """
    Plot the confusion matrix and calculate the accuracy of the EfficientNetB1 model.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        fig: The confusion matrix figure.
        acc: The accuracy of the model.
    """
    # calculate the accuracy
    acc = accuracy_score(y_true, y_pred)

    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format=".4g")
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.set_ylabel("True Label", fontsize=15)

    # set the x and y ticks
    xticks = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    yticks = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=12)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=12)

    plt.tight_layout(pad=0.3)

    return fig, acc



from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def UNet_preprocessing_pro(dataset, batch_size, gen):
    """
    Preprocess the dataset including healthy and CBB images for UNet training.

    Args:
        dataset: tf.data.Dataset object.
        batch_size: int, batch size for the dataset.
        gen: tf.random.Generator object.

    Returns:
        unet_trainset: tf.data.Dataset object for training.
        unet_valset: tf.data.Dataset object for validation.
    """
    healthy_leaf_set = get_healthy_image(dataset)
    unet_dataset = build_Unet_dataset(healthy_leaf_set)
    cbb_leaf_set = get_CBB_image(dataset)
    cbb_dataset = build_cbb_dataset(cbb_leaf_set)

    # fig = vs.plot_cive_result(unet_dataset)
    # fig.savefig("figures/Image after CIVE.png")

    image_list = []
    mask_list = []

    for img, msk in unet_dataset:
        image_list.append(img)
        mask_list.append(msk)

    for img, msk in cbb_dataset:
        image_list.append(img)
        mask_list.append(msk)
        if gen.uniform(()) > 0.8:
            img_flip = tf.image.flip_left_right(img)
            msk_flip = tf.image.flip_left_right(msk)
            image_list.append(img_flip)
            mask_list.append(msk_flip)
        if gen.uniform(()) > 0.8:
            img_flip = tf.image.flip_up_down(img)
            msk_flip = tf.image.flip_up_down(msk)
            image_list.append(img_flip)
            mask_list.append(msk_flip)

    image_list = np.array(image_list)
    mask_list = np.array(mask_list)

    print(image_list.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        image_list, mask_list, test_size=0.2, random_state=711
    )

    unet_trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    unet_valset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return unet_trainset, unet_valset


def build_cbb_dataset(train_dataset):
    """
    Build the dataset for CBB images.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
    """
    cbb_dataset = train_dataset.map(
        create_mask_pair_cbb, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return cbb_dataset


def UNet_preprocessing(dataset, batch_size):
    """
    Preprocess the dataset including healthy images for UNet training.

    Args:
        dataset: tf.data.Dataset object.
        batch_size: int, batch size for the dataset.

    Returns:
        unet_trainset: tf.data.Dataset object for training.
        unet_valset: tf.data.Dataset object for validation.
    """
    healthy_leaf_set = get_healthy_image(dataset)
    unet_dataset = build_Unet_dataset(healthy_leaf_set)

    # fig = vs.plot_cive_result(unet_dataset)
    # fig.savefig("figures/Image after CIVE.png")

    image = []
    mask = []

    for img, msk in unet_dataset:
        image.append(img)
        mask.append(msk)

    image = np.array(image)
    mask = np.array(mask)

    print(image.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        image, mask, test_size=0.2, random_state=711
    )

    unet_trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    unet_valset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return unet_trainset, unet_valset


def compute_cive(image):
    """
    Compute the CIVE index for a given healthy cassava leaf image.

    Args:
        image: np.ndarray, input image.

    Returns:
        cive: np.ndarray, CIVE index.
    """
    # CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return cive


def compute_cive_cbb(image):
    """
    Compute the CIVE index for a given CBB cassava leaf image.

    Args:
        image: np.ndarray, input image.

    Returns:
        cive: np.ndarray, CIVE index.
    """
    # CIVE = 10.441 * R + 0.611 * G - 1.885 * B - 48.787
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 10.441 * R + 0.611 * G - 1.885 * B - 48.787
    return cive


def get_mask(cive, threshold=None):
    """
    Create the mask based on the CIVE index.

    Args:
        cive: np.ndarray, CIVE index.
        threshold: float, threshold value for mask creation.

    Returns:
        mask: np.ndarray, binary mask.
    """
    if threshold is None:
        # threshold = tf.reduce_mean(cive)
        threshold = 18.7  # threshold is set to 18.7
    mask = tf.cast(cive < threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def get_mask_cbb(cive, threshold=None):
    """
    Create the mask based on the CIVE index for CBB images.

    Args:
        cive: np.ndarray, CIVE index.
        threshold: float, threshold value for mask creation.

    Returns:
        mask: np.ndarray, binary mask.
    """
    if threshold is None:
        threshold = tf.reduce_mean(cive)
        # threshold = -48.63
    mask = tf.cast(cive > threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def create_mask_pair(image, label):
    """
    Create a mask pair for the given image and label.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.

    Returns:
        image: np.ndarray, input image.
        mask: np.ndarray, binary mask.
    """
    cive = compute_cive(image)
    mask = get_mask(cive, threshold=None)
    return image, mask


def create_mask_pair_cbb(image, label):
    """
    Create a mask pair for the given CBB image and label.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.

    Returns:
        image: np.ndarray, input image.
        mask: np.ndarray, binary mask.
    """
    cive = compute_cive_cbb(image)
    mask = get_mask_cbb(cive, threshold=None)
    return image, mask


def build_Unet_dataset(train_dataset):
    """
    Build the dataset including images and masks for UNet training.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        Unet_dataset: tf.data.Dataset object for UNet training.
    """
    unet_dataset = train_dataset.map(
        create_mask_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return unet_dataset


def which_classes(image, label, classes):
    """
    Filter the dataset based on the given classes.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.
        classes: int, class to filter.

    Returns:
        bool: True if the label matches the classes, False otherwise.
    """
    return tf.equal(label, classes)


def get_healthy_image(train_dataset):
    """
    Get the healthy images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        healthy_dataset: tf.data.Dataset object for healthy images.
    """
    filter_fn = partial(which_classes, classes=4)
    healthy_dataset = train_dataset.filter(filter_fn)
    # healthy_dataset_num = count_data_items(healthy_dataset)
    # print(f"Number of healthy images: {healthy_dataset_num}")
    return healthy_dataset


def get_all_diseases_image(train_dataset):
    """
    Get all the diseases images including CBB, CBSD, CGM, and CMD from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
        cbsd_dataset: tf.data.Dataset object for CBSD images.
        cgm_dataset: tf.data.Dataset object for CGM images.
        cmd_dataset: tf.data.Dataset object for CMD images.
    """
    cbb_dataset = get_CBB_image(train_dataset)
    cbsd_dataset = get_CBSD_image(train_dataset)
    cgm_dataset = get_CGM_image(train_dataset)
    cmd_dataset = get_CMD_image(train_dataset)
    return cbb_dataset, cbsd_dataset, cgm_dataset, cmd_dataset


def get_CBB_image(train_dataset):
    """
    Get the CBB images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
    """
    filter_fn = partial(which_classes, classes=0)
    cbb_dataset = train_dataset.filter(filter_fn)
    # cbb_dataset_num = count_data_items(cbb_dataset)
    # print(f"Number of CBB images: {cbb_dataset_num}")
    return cbb_dataset


def get_CBSD_image(train_dataset):
    """
    Get the CBSD images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbsd_dataset: tf.data.Dataset object for CBSD images.
    """
    filter_fn = partial(which_classes, classes=1)
    cbsd_dataset = train_dataset.filter(filter_fn)
    # cbsd_dataset_num = count_data_items(cbsd_dataset)
    # print(f"Number of CBSD images: {cbsd_dataset_num}")
    return cbsd_dataset


def get_CGM_image(train_dataset):
    """
    Get the CGM images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cgm_dataset: tf.data.Dataset object for CGM images.
    """
    filter_fn = partial(which_classes, classes=2)
    cgm_dataset = train_dataset.filter(filter_fn)
    # cgm_dataset_num = count_data_items(cgm_dataset)
    # print(f"Number of CGM images: {cgm_dataset_num}")
    return cgm_dataset


def get_CMD_image(train_dataset):
    """
    Get the CMD images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cmd_dataset: tf.data.Dataset object for CMD images.
    """
    filter_fn = partial(which_classes, classes=3)
    cmd_dataset = train_dataset.filter(filter_fn)
    # cmd_dataset_num = count_data_items(cmd_dataset)
    # print(f"Number of CMD images: {cmd_dataset_num}")
    return cmd_dataset


def data_acquisition(gcs_path, train_file_path, image_size):
    """
    Acquire the training and validation datasets for image segmentation.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.

    Returns:
        train_dataset: tf.data.Dataset object for training.
        val_dataset: tf.data.Dataset object for validation.
    """
    trainfile, valfile = split_data(gcs_path, train_file_path)
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    # # visualise the images
    # fig1, fig2 = vs.origin_image_plot(train_dataset)
    # fig2.savefig("figures/Healthy Leaf.png")
    # fig1.savefig("figures/Diseases Leaf.png")
    return train_dataset, val_dataset


def data_acquisition_classification(
    gcs_path, train_file_path, image_size, train_ratio, val_ratio
):
    """
    Acquire the training, validation, and test datasets to build a new dataset for classification.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.

    Returns:
        train_dataset: tf.data.Dataset object for training.
        val_dataset: tf.data.Dataset object for validation.
        test_dataset: tf.data.Dataset object for testing.
    """
    trainfile, valfile, testfile = split_data_classification(
        gcs_path, train_file_path, train_ratio, val_ratio
    )
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    test_dataset = load_testset(testfile, image_size=image_size, labeled=True)
    return train_dataset, val_dataset, test_dataset


def decode_raw_image(image_data, image_shape):
    """
    Decode the raw image data and resize it to the specified shape.

    Args:
        image_data: bytes, raw image data.
        image_shape: list, shape of the image.

    Returns:
        image: tf.Tensor, decoded and resized image.
    """
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*image_shape, 3])
    return image


def read_tfrecord(example, image_size, labeled):
    """
    Read a single TFRecord example and decode the image data.

    Args:
        example: tf.train.Example, TFRecord example.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.

    Returns:
        images: tf.Tensor, decoded and resized image.
        labels: tf.Tensor, labels for the images (if labeled).
    """
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    images = decode_raw_image(example["image"], image_shape=image_size)
    if labeled:
        labels = tf.cast(example["target"], tf.int32)
        return images, labels
    image_name = example["image_name"]
    return images, image_name


def split_data(gcs_path, train_file_path):
    """
    Split the tfrecords into training and validation use.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.

    Returns:
        train_filename: list, list of training filenames.
        val_filename: list, list of validation filenames.
    """
    train_filename, val_filename = train_test_split(
        tf.io.gfile.glob(gcs_path + train_file_path), train_size=0.8, random_state=711
    )
    return train_filename, val_filename


def split_data_classification(gcs_path, train_file_path, train_ratio, val_ratio):
    """
    Split the tfrecords into training, validation, and test use.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.

    Returns:
        train_filename: list, list of training filenames.
        val_filename: list, list of validation filenames.
        test_filename: list, list of test filenames.
    """
    train_filename, val_test_filename = train_test_split(
        tf.io.gfile.glob(gcs_path + train_file_path),
        train_size=train_ratio,
        random_state=711,
    )

    val_test_ratio = val_ratio / (1 - train_ratio)
    val_filename, test_filename = train_test_split(
        val_test_filename, train_size=val_test_ratio, random_state=711
    )

    return train_filename, val_filename, test_filename


def load_trainset(filenames, image_size, labeled, ordered=False):
    """
    Load the training dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for training.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size, labeled=labeled),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def load_valset(filenames, image_size, labeled, ordered=False):
    """
    Load the validation dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for validation.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size, labeled=labeled),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def load_testset(filenames, image_size, labeled, ordered=False):
    """
    Load the test dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for testing.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size, labeled=labeled),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def count_data_items(trainset):
    """
    Count the number of samples in the dataset.

    Args:
        trainset: tf.data.Dataset object.

    Returns:
        sample_count: int, number of samples in the dataset.
    """
    sample_count = sum(1 for _ in trainset)
    return sample_count


# --------preprocessing for classification task--------


def create_classification_dataset_batch(model, dataset, batch_size=64, threshold=0.4):
    """
    Create a new dataset after image segmentation.

    Args:
        model: tf.keras.Model object, trained UNet model for segmentation.
        dataset: tf.data.Dataset object, input dataset.
        batch_size: int, batch size for the dataset.
        threshold: float, threshold value for mask creation.

    Returns:
        dataset: tf.data.Dataset object, new dataset after segmentation.
    """
    images, labels = [], []

    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels)

    # predict the mask
    masks = model.predict(images, batch_size=batch_size)
    masks = masks > threshold
    masks = tf.cast(masks, tf.float32)

    # segment the image
    segmented_images = images * masks

    return tf.data.Dataset.from_tensor_slices((segmented_images, labels))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image, label, image_type=tf.float32):
    """
    Create a TFRecord example from the image and label.

    Args:
        image: tf.Tensor, input image.
        label: tf.Tensor, input label.
        image_type: tf.DType, type of the image.

    Returns:
        example: tf.train.Example, TFRecord example.
    """
    # image_raw = tf.io.serialize_tensor(tf.cast(image, image_type)).numpy()
    image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    image_raw = tf.image.encode_jpeg(image_uint8).numpy()
    label = int(label.numpy())

    feature = {
        "image": _bytes_feature(image_raw),
        "target": _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_tfrecord(dataset, filename):
    """
    Save the dataset to a TFRecord file.

    Args:
        dataset: tf.data.Dataset object, input dataset.
        filename: str, path to the output TFRecord file.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            example = create_example(image, label)
            writer.write(example.SerializeToString())
    print(f"{filename} saved")


def count_data_items_from_tfrecord(filenames):
    """
    Count the number of samples in the TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.

    Returns:
        count: int, number of samples in the TFRecord files.
    """
    count = 0
    for f in filenames:
        for _ in tf.data.TFRecordDataset(f):
            count += 1
    return count


def preprocess_image_for_classification(
    gcs_path,
    train_file_path,
    val_file_path,
    test_file_path,
    image_size,
    batch_size,
    autotune,
):
    """
    Preprocess the dataset for cassava leaf image classification based on new dataset.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        val_file_path: str, path to the validation file.
        test_file_path: str, path to the test file.
        image_size: list, size of the images.
        batch_size: int, batch size for the dataset.
        autotune: tf.data.experimental.AUTOTUNE object.

    Returns:
        trainset: tf.data.Dataset object for training.
        valset: tf.data.Dataset object for validation.
        testset: tf.data.Dataset object for testing.
        num_trainset: int, number of samples in the training dataset.
        num_valset: int, number of samples in the validation dataset.
    """
    trainset_path = gcs_path + train_file_path
    trainset_files = tf.io.gfile.glob(trainset_path)
    num_trainset = count_data_items_from_tfrecord(trainset_files)
    valset_path = gcs_path + val_file_path
    valset_files = tf.io.gfile.glob(valset_path)
    num_valset = count_data_items_from_tfrecord(valset_files)
    testset_path = gcs_path + test_file_path

    trainset = load_trainset(trainset_files, image_size=image_size, labeled=True)
    # trainset = trainset.repeat()
    trainset = trainset.shuffle(2048, seed=711)
    trainset = trainset.repeat()
    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(autotune)
    valset = load_valset(valset_path, image_size=image_size, labeled=True)
    valset = valset.repeat().batch(batch_size).prefetch(autotune)
    testset = load_testset(testset_path, image_size=image_size, labeled=True)
    testset = testset.batch(batch_size).prefetch(autotune)

    return trainset, valset, testset, num_trainset, num_valset


def preprocess_image_for_compare(
    gcs_path,
    train_file_path,
    image_size,
    train_ratio,
    val_ratio,
    batch_size,
    autotune,
):
    """
    Preprocess the dataset for cassava leaf image classification based on original dataset.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.
        batch_size: int, batch size for the dataset.
        autotune: tf.data.experimental.AUTOTUNE object.

    Returns:
        trainset: tf.data.Dataset object for training.
        valset: tf.data.Dataset object for validation.
        testset: tf.data.Dataset object for testing.
        num_trainset: int, number of samples in the training dataset.
        num_valset: int, number of samples in the validation dataset.
    """
    trainfile, valfile, testfile = split_data_classification(
        gcs_path, train_file_path, train_ratio, val_ratio
    )
    num_trainset = count_data_items_from_tfrecord(trainfile)
    num_valset = count_data_items_from_tfrecord(valfile)

    trainset = load_trainset(trainfile, image_size=image_size, labeled=True)
    trainset = trainset.repeat()
    trainset = trainset.shuffle(2048, seed=711)
    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(autotune)
    valset = load_valset(valfile, image_size=image_size, labeled=True)
    valset = valset.repeat().batch(batch_size).prefetch(autotune)
    testset = load_testset(testfile, image_size=image_size, labeled=True)
    testset = testset.batch(batch_size).prefetch(autotune)

    return trainset, valset, testset, num_trainset, num_valset


import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers, models
from tensorflow.keras.applications import EfficientNetB1


def set_seed(seed=711):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        tf.random.Generator: A TensorFlow random generator with the specified seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    return tf.random.Generator.from_seed(seed)


def Unet_Arch(dropout_num, input_shape=(256, 256, 3)):
    """
    Define the U-Net architecture.

    Args:
        dropout_num (float): Dropout rate.
        input_shape (tuple): Shape of the input image.

    Returns:
        tf.keras.Model: U-Net model.
    """
    inputs = keras.Input(shape=input_shape)
    conv_init = initializers.HeNormal(seed=711)
    bias_init = initializers.Zeros()

    # contracting path
    c1 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        inputs
    )  # 256 x 256 x 16
    c2 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c1
    )  # 256 x 256 x 16
    p1 = layers.MaxPooling2D((2, 2))(c2)  # 128 x 128 x 16
    p1 = layers.Dropout(dropout_num)(p1)

    c3 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        p1
    )  # 128 x 128 x 32
    c4 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c3
    )  # 128 x 128 x 32
    p2 = layers.MaxPooling2D((2, 2))(c4)  # 64 x 64 x 32
    p2 = layers.Dropout(dropout_num)(p2)

    # bottleneck
    c5 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        p2
    )  # 64 x 64 x 64
    c6 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c5
    )  # 64 x 64 x 64

    # expansive path
    u1 = layers.UpSampling2D((2, 2))(c6)  # 128 x 128 x 64
    u1 = layers.Concatenate()([u1, c4])  # 128 x 128 x 96
    u1 = layers.Dropout(dropout_num)(u1)
    c7 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        u1
    )  # 128 x 128 x 32
    c8 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c7
    )  # 128 x 128 x 32

    u2 = layers.UpSampling2D((2, 2))(c8)  # 256 x 256 x 32
    u2 = layers.Concatenate()([u2, c2])  # 256 x 256 x 48
    u2 = layers.Dropout(dropout_num)(u2)
    c9 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        u2
    )  # 256 x 256 x 16
    c10 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c9
    )  # 256 x 256 x 16

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c10)  # 256 x 256 x 1

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def Unet_train(model, train, val, epochs=10, learning_rate=0.001):
    """
    Build the U-Net model.

    Args:
        model (tf.keras.Model): U-Net model.
        train (tf.data.Dataset): Training dataset.
        val (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate.

    Returns:
        tf.keras.Model: Trained U-Net model.
        matplotlib.figure.Figure: Learning curve figure.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
    )

    fig = learning_curve(history)

    return model, fig


def Unet_test(model, dataset):
    """
    Test the U-Net model.

    Args:
        model (tf.keras.Model): U-Net model.
        dataset (tf.data.Dataset): Dataset to test.

    Returns:
        matplotlib.figure.Figure: Figure showing the original image, mask, and processed image.
    """
    cbb_dataset, cbsd_dataset, cgm_dataset, cmd_dataset = get_all_diseases_image(
        dataset
    )
    # get the test sample
    for image, _ in cbb_dataset.skip(56).take(1):  # skip(56).take(1)  11 71 116
        cbb_image_test = image
        cbb_image_test_batch = tf.expand_dims(cbb_image_test, axis=0)

    for image, _ in cbsd_dataset.skip(55).take(1):
        cbsd_image_test = image
        cbsd_image_test_batch = tf.expand_dims(cbsd_image_test, axis=0)

    for image, _ in cgm_dataset.skip(45).take(1):  # 45
        cgm_image_test = image
        cgm_image_test_batch = tf.expand_dims(cgm_image_test, axis=0)

    for image, _ in cmd_dataset.take(1):
        cmd_image_test = image
        cmd_image_test_batch = tf.expand_dims(cmd_image_test, axis=0)

    # predict the test sample
    cbb_mask_test = model.predict(cbb_image_test_batch)
    cbb_mask_test = tf.squeeze(cbb_mask_test, axis=0)
    cbb_mask_test = (cbb_mask_test > 0.4).numpy().astype("uint8")
    cbsd_mask_test = model.predict(cbsd_image_test_batch)
    cbsd_mask_test = tf.squeeze(cbsd_mask_test, axis=0)
    cbsd_mask_test = (cbsd_mask_test > 0.4).numpy().astype("uint8")
    cgm_mask_test = model.predict(cgm_image_test_batch)
    cgm_mask_test = tf.squeeze(cgm_mask_test, axis=0)
    cgm_mask_test = (cgm_mask_test > 0.4).numpy().astype("uint8")
    cmd_mask_test = model.predict(cmd_image_test_batch)
    cmd_mask_test = tf.squeeze(cmd_mask_test, axis=0)
    cmd_mask_test = (cmd_mask_test > 0.4).numpy().astype("uint8")

    # plot the test sample
    fig = origin_Unet_result_plot(
        [cbb_image_test, cbsd_image_test, cgm_image_test, cmd_image_test],
        [cbb_mask_test, cbsd_mask_test, cgm_mask_test, cmd_mask_test],
        ["CBB", "CBSD", "CGM", "CMD"],
    )

    return fig


def EfficientNetB1_Arch(
    dropout_ratio, layers_freezed, input_shape=(224, 224, 3), classes=5
):
    """
    Build a EfficientNetB1 model based on transfer learning.

    Args:
        dropout_ratio (float): Dropout rate.
        layers_freezed (int): Number of layers to freeze.
        input_shape (tuple): Shape of the input image.
        classes (int): Number of classes.

    Returns:
        tf.keras.Model: EfficientNetB1 model.
    """
    base_model = EfficientNetB1(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = True

    # Freeze some layers
    # cheak the number of layers in the base model
    print(len(base_model.layers))
    # Freeze the first 50 layers
    for layer in base_model.layers[:layers_freezed]:
        layer.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_ratio),
            layers.Dense(classes, activation="softmax"),
        ]
    )

    return model


def Effi_B1_train(
    model,
    trainset,
    valset,
    epochs,
    learning_scheduler,
    num_samples,
    num_valset,
    batch_size,
):
    """
    Train the EfficientNetB1 model.

    Args:
        model (tf.keras.Model): EfficientNetB1 model.
        trainset (tf.data.Dataset): Training dataset.
        valset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        learning_scheduler: Learning rate schedule.
        num_samples (int): Number of samples in the training set.
        num_valset (int): Number of samples in the validation set.
        batch_size (int): Batch size.

    Returns:
        tf.keras.Model: Trained EfficientNetB1 model.
        matplotlib.figure.Figure: Learning curve figure.
        tf.keras.callbacks.History: Training history.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_scheduler, epsilon=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        steps_per_epoch=num_samples // batch_size,
        validation_steps=num_valset // batch_size,
    )

    fig = learning_curve(history)

    return model, fig, history


def Effi_B1_test(model, testset):
    """
    Test the EfficientNetB1 model.

    Args:
        model (tf.keras.Model): EfficientNetB1 model.
        testset (tf.data.Dataset): Test dataset.

    Returns:
        matplotlib.figure.Figure: Figure of confusion matrix.
        float: Classification accuracy.
    """
    y_true = []
    y_pred = []

    # get the image and label from the testset
    for images, labels in testset:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # convert to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # get the classification result
    fig, acc = classfication_result(y_true, y_pred)

    return fig, acc


set_seed()


import tensorflow as tf
import os, re
import numpy as np
import matplotlib.pyplot as plt
import importlib
from tensorflow.keras import optimizers


autotune = tf.data.experimental.AUTOTUNE
gcs_path = "/kaggle/input"
trainset_path = "/cassava-leaf-disease-classification/train_tfrecords/ld_train*.tfrec"
# parameters for image segmentation
batch_size = 256
batch_size_unet = 64
image_size = [256, 256]
classes = ["0", "1", "2", "3", "4"]
epochs = 18
epochs_pro = 26
Unet_lr = 0.0005
Unet_lr_pro = 0.0002
dropout_num_pro = 0.12
train_ratio_cnn = 0.8
val_ratio_cnn = 0.1
learning_scheduler = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9
)
  
# data path for classfication
trainset_path_enet = "/new-dataset/New_Dataset/trainset_*.tfrec"
valset_path_enet = "/new-dataset/New_Dataset/validation_set.tfrec"
testset_path_enet = "/new-dataset/New_Dataset/test_set.tfrec"

image_size_enet = [240, 240]
dropout_ratio_ori = 0.7
dropout_ratio_enet = 0.6
layer_freezed = 50
epochs_ori = 12
epochs_enet = 12
batch_size_enet = 64
learning_scheduler_ori = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00008, decay_steps=500, decay_rate=0.4
)
learning_scheduler_enet = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, decay_steps=500, decay_rate=0.5
) 


trainset_ori, valset_ori, testset_ori, num_train_ori, num_val_ori = preprocess_image_for_compare(
    gcs_path, 
    trainset_path, 
    image_size_enet, 
    train_ratio_cnn, 
    val_ratio_cnn, 
    batch_size_enet, 
    autotune,
)


# print(num_train_ori)
# print(num_val_ori)


trainset_enet, valset_enet, testset_enet, num_train_enet, num_val_enet = preprocess_image_for_classification(
    gcs_path, 
    trainset_path_enet, 
    valset_path_enet, 
    testset_path_enet, 
    image_size_enet, 
    batch_size_enet, 
    autotune,
)


print(num_train_enet)


print(num_val_enet)


Enet_model_enet = EfficientNetB1_Arch(
    dropout_ratio=dropout_ratio_enet, 
    layers_freezed=layer_freezed, 
    input_shape=(*image_size_enet, 3), 
    classes=5,
)


Enet_model_enet.summary()


Enet_model_done, Enet_lr_fig, enet_history = Effi_B1_train(
    Enet_model_enet, 
    trainset_enet, 
    valset_enet, 
    epochs=epochs_enet, 
    learning_scheduler=learning_scheduler_enet, 
    num_samples=num_train_enet, 
    num_valset=num_val_enet, 
    batch_size=batch_size_enet,
)


tf.keras.backend.clear_session()
Enet_model_ori = EfficientNetB1_Arch(
    dropout_ratio=dropout_ratio_ori, 
    layers_freezed=layer_freezed, 
    input_shape=(*image_size_enet, 3), 
    classes=5,
)


Enet_model_ori.summary()


ori_model_done, ori_lr_fig, ori_history = Effi_B1_train(
    Enet_model_ori, 
    trainset_ori, 
    valset_ori, 
    epochs=epochs_ori, 
    learning_scheduler=learning_scheduler_ori, 
    num_samples=num_train_ori, 
    num_valset=num_val_ori, 
    batch_size=batch_size_enet,
)


fig_enet, acc_enet = Effi_B1_test(Enet_model_done, testset_enet)


print(acc_enet)


figure_dir = "/kaggle/working/figure"
model_dir = "/kaggle/working/model"

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# # save the model
# Enet_model_done.save(os.path.join(model_dir, "EfficientNetB1_model.h5"))
# Enet_model_done.save(os.path.join(model_dir, "EfficientNetB1_model.keras"))
# ori_model_done.save(os.path.join(model_dir, "EfficientNetB1_model_ori.h5"))
# ori_model_done.save(os.path.join(model_dir, "EfficientNetB1_model_ori.keras"))


# # save the learning curve
# Enet_lr_fig.savefig(os.path.join(figure_dir, "EfficientNetB1_model_learning_curve.png"))
# fig_enet.savefig(os.path.join(figure_dir, "EfficientNetB1_model_test_result.png"))

