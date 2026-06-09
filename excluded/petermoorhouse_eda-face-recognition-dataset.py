import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os


class TrainingDataset:
    def __init__(self, title, images_npz_path, labels_csv_path = "none"):

        self.images = self.get_images_from_npz(images_npz_path)
        self.n = len(list(self.images))
        self.labels = self.get_labels_from_csv(self.n, labels_csv_path)

        print(f"Loaded '{title}' data, containing {self.n} samples")

    def get_images_from_npz(self, filename):

        data = np.load(filename)
        images = data['arr_0']
        return images

    def get_labels_from_csv(self, n, filename):

        if not os.path.exists(filename):
            return ["Unknown"] * n

        train_y = pd.read_csv(filename)
        train_y = train_y["Predicted"].values
        return train_y


DATASET_DIRECTORY = "/kaggle/input/cnn-face-recognition-25"
TRAIN_IMAGES_NPZ = f"{DATASET_DIRECTORY}/faces_train_x.npz"
TRAIN_LABELS_CSV = f"{DATASET_DIRECTORY}/faces_train_y.csv"
TEST_IMAGES_NPZ = f"{DATASET_DIRECTORY}/faces_test_x.npz"


TRAINING_DATASET = TrainingDataset("train", TRAIN_IMAGES_NPZ, TRAIN_LABELS_CSV)
TESTING_DATASET = TrainingDataset("test", TEST_IMAGES_NPZ)


class ImagePresenter:
    def __init__(self, images, labels):

        self.images = images
        self.labels = labels
        self.n = len(list(self.images))

    def present_random_image(self):

        i = random.randint(0, self.n)
        (image, label) = (self.images[i], self.labels[i])

        plt.imshow(image, cmap='gray')
        plt.title(f"Image {i} (Label: {label})")
        plt.axis('off')
        plt.show()

    def get_random_identifiers_with_label(self, label):

        all_identifiers = self.get_all_identifiers_with_label(label)

        chosen = set([])
        while len(chosen) < 3:
            i = random.choice(all_identifiers)
            chosen.add(i)

        return list(chosen)

    def get_all_identifiers_with_label(self, label):
        
        return [i for i, lab in enumerate(self.labels) if lab == label]

    def present_random_images_with_label(self, label):

        identifiers = self.get_random_identifiers_with_label(label)
        self.present_three_images(identifiers)
        
    def present_three_images(self, identifiers):
        
        images = [self.images[i] for i in identifiers]
        labels = [self.labels[i] for i in identifiers]
    
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, label, idx in zip(axes, images, labels, identifiers):
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Image {idx} (Label: {label})")
            ax.axis('off')
        plt.tight_layout()
        plt.show()


presenter = ImagePresenter(TRAINING_DATASET.images, TRAINING_DATASET.labels)
presenter.present_three_images([0, 1, 2])


for i in range(8):
    presenter.present_random_images_with_label(i)


presenter = ImagePresenter(TESTING_DATASET.images, TESTING_DATASET.labels)
presenter.present_three_images([0, 1, 2])


class ClassRepresentationAnalyst:
    def __init__(self, labels):

        self.labels = labels
        self.class_labels = list(set(labels))

    def get_total_number_of_samples(self):

        total = 0
        for label in self.class_labels:
            total += self.count_instances_of_class(label)

        return total

    def plot_class_representation(self):

        df = self.create_class_representation_dataframe()

        sns.set_theme(style="darkgrid")
        
        sns.barplot(x="num_samples", y="class_label", data=df)
        
        plt.title("Class Representation")
        plt.ylabel("class label")
        plt.xlabel("num. samples")
        plt.show()
                
    def create_class_representation_dataframe(self):

        table = {"class_label": [], "num_samples": []}

        for label in self.class_labels:
            count = self.count_instances_of_class(label)
            table["class_label"].append(str(label))
            table["num_samples"].append(count)

        return pd.DataFrame(table)

    def count_instances_of_class(self, class_label):

        total = 0
        for label in self.labels:
            if class_label == label:
                total += 1

        return total


analyst = ClassRepresentationAnalyst(TRAINING_DATASET.labels)
analyst.plot_class_representation()

