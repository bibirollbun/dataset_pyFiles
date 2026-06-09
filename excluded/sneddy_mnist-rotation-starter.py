import pandas as pd
import os

work_dir = "../input/mnist-rotation"
os.listdir(work_dir)


import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def show_image(image, title=None):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    if title is not None:
        plt.title(title)
    plt.show()

def rotate(img: np.ndarray, angle: int) -> np.ndarray:
    if not (-120 <= angle <= 120):
        raise ValueError("Angle must be between -120 and 120 degrees.")
    
    rotated_img = scipy_rotate(
        img,
        angle=angle,
        reshape=False,      # сохраняем размер 28x28
        order=1,            # билинейная интерполяция для аккуратного поворота
        mode='constant',    # пиксели вне исходного изображения заполняются константой
        cval=0.0            # заполняем черным (0)
    )
    return rotated_img


class NoizeGenerator:
    def __init__(
        self,
        discrete_noise_proba=0.02,
        beta_alpha=0.3,
        beta_beta=0.3,
        gaussian_sigma=0.0,
        shift_prob=1.0,
        seed=None,
    ):
        """
        discrete_noise_proba: вероятность заменить пиксель значением из бета-распределения
        beta_alpha, beta_beta: параметры бета-распределения (бимодальность при < 1)
        gaussian_sigma: стандартное отклонение для нормального шума
        shift_prob: вероятность случайного сдвига изображения на 1 пиксель
        """
        self.discrete_noise_proba = discrete_noise_proba
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.gaussian_sigma = gaussian_sigma
        self.shift_prob = shift_prob
        self.rng = np.random.default_rng(seed)

    def apply_beta_noise(self, img):
        mask = self.rng.random(img.shape) < self.discrete_noise_proba
        beta_noise = (
            self.rng.beta(self.beta_alpha, self.beta_beta, size=img.shape) * 255
        )
        noisy_img = img.copy()
        noisy_img[mask] = beta_noise[mask]
        return noisy_img

    def apply_gaussian_noise(self, img):
        if self.gaussian_sigma > 0:
            noise = self.rng.normal(loc=0.0, scale=self.gaussian_sigma, size=img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)
        return img

    def apply_random_shift(self, img):
        direction = self.rng.choice(["up", "down", "left", "right"])
        shifted = np.zeros_like(img)

        if direction == "up":
            shifted[:-1, :] = img[1:, :]
        elif direction == "down":
            shifted[1:, :] = img[:-1, :]
        elif direction == "left":
            shifted[:, :-1] = img[:, 1:]
        elif direction == "right":
            shifted[:, 1:] = img[:, :-1]

        return shifted

    def transform_image(self, image):
        tmp_image = image.astype(np.float32)
        tmp_image = self.apply_beta_noise(tmp_image)
        tmp_image = self.apply_gaussian_noise(tmp_image)

        if self.rng.random() < self.shift_prob:
            tmp_image = self.apply_random_shift(tmp_image)
        return tmp_image.astype(np.uint8)

    def transform_dataset(self, X):
        n_samples = X.shape[0]
        X_aug = np.zeros_like(X)

        for i in range(n_samples):
            img = self.transform_image(X[i].reshape(28, 28))
            X_aug[i] = img.flatten()
        return X_aug
    

noize_gen = NoizeGenerator(
    discrete_noise_proba=0.2,
    beta_alpha=0.3,
    beta_beta=0.3,
    gaussian_sigma=40,
    shift_prob=1.0,
    seed=1,
)


train = pd.read_pickle(os.path.join(work_dir, 'train.pkl'))
train.head()


show_image(train.image.iloc[1], title=f"Label {train.label.iloc[1]}")


show_image(noize_gen.transform_image(train.image.iloc[1]), title=f"Label {train.label.iloc[1]}")


test = pd.read_pickle(os.path.join(work_dir, 'test.pkl'))
test.head()


show_image(test.image.iloc[2], title=f"Label {test.label.iloc[2]}")


allowed_angles = np.arange(-120, 121, 30)

print(f"Разрешенные углы поворота: {allowed_angles}")

for angle in allowed_angles:
    show_image(rotate(test.image.iloc[2], angle), title=f"Label {test.label.iloc[2]}, Angle: {angle}")


sample_submission = pd.read_csv(os.path.join(work_dir, 'sample_submission.csv'))
sample_submission




