import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  RANSACRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import cv2


try:
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    from tqdm import tqdm as tqdm_notebook
    import warnings
    warnings.warn("tqdm.notebook bulunamadı, standart tqdm kullanılıyor. Görsel sorunlar yaşanabilir.")


class HistologyDataLoader:
    def __init__(self, h5_file_path="/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", save_logic=True):
        self.h5_file_path = h5_file_path
        self.train_slide_ids = []
        self.train_spot_tables = {}
        self.train_images = {}
        self.test_slide_ids = []
        self.test_spot_tables = {}
        self.test_images = {}
        save_file = save_logic

    def load_train_spot_data(self):
        """
        Load training spot data from the H5 file and store each slide as a DataFrame.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            train_spots = f["spots/Train"]
            for slide_name in train_spots.keys():
                spot_array = np.array(train_spots[slide_name])
                df = pd.DataFrame(spot_array, columns=["x", "y"] + [f"C{i}" for i in range(1, 36)])
                self.train_spot_tables[slide_name] = df
                self.train_slide_ids.append(slide_name)
        print("Training spot data loaded successfully.", self.train_slide_ids)
        
    def load_train_images(self):
        """
        Load training HE images from the H5 file.
        Adjust the key if your H5 file uses a different naming convention.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            # Adjust key if necessary (e.g., f["images/Train"] if needed)
            train_imgs = f["images/Train"]
            for slide_name in train_imgs.keys():
                image_array = np.array(train_imgs[slide_name])
                self.train_images[slide_name] = image_array
        print("Training images loaded successfully.",self.train_slide_ids)

    def load_test_spot_data(self, slide_id="S_7"):
        """
        Load test spot data for a given slide.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_spots = f["spots/Test"]
            if slide_id not in test_spots:
                print("ATTENTION:", test_spots.keys())
                raise ValueError(f"Slide {slide_id} not found in test spot data.")
            spot_array = np.array(test_spots[slide_id])
            test_df = pd.DataFrame(spot_array, columns=["x", "y"])
            self.test_slide_ids.append(slide_id)
            self.test_spot_tables[slide_id] = test_df
        print(f"Test spot data for slide {slide_id} loaded successfully.")
        return test_df

    def load_test_images(self, slide_id="S_7"):
        """
        Load test HE image for a given slide.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            test_imgs = f["images/Test"]
            if slide_id not in test_imgs:
                raise ValueError(f"Slide {slide_id} not found in test images.")
            image_array = np.array(test_imgs[slide_id])
            self.test_images[slide_id] = image_array
        print(f"Test image for slide {slide_id} loaded successfully.")
        return image_array

    def load_train_data(self):
        self.load_train_spot_data()
        self.load_train_images()

    def load_test_data(self):
        self.load_test_spot_data()
        self.load_test_images()

    def visualize(self, image, image2=None, coor=None, coor2=None, title="Sample", 
               title2=None, figsize=(6,6), comparison=False):
        """
        Bir ya da iki görüntüyü görselleştirir ve isteğe bağlı olarak koordinat noktalarını gösterir.
        
        Parametreler:
            image: Görselleştirilecek ana görüntü
            image2: Karşılaştırma için ikinci görüntü (isteğe bağlı)
            coor: İlk görüntü üzerinde işaretlenecek koordinatlar {"x": x_coords, "y": y_coords}
            coor2: İkinci görüntü üzerinde işaretlenecek koordinatlar {"x": x_coords, "y": y_coords}
            title: İlk görüntünün başlığı (varsayılan: "Sample")
            title2: İkinci görüntünün başlığı (isteğe bağlı)
            figsize: Görüntüleme boyutu (varsayılan: (6,6))
            comparison: Karşılaştırma modunu etkinleştirme (varsayılan: False)
        """
        
        if comparison and image2 is not None:
            # İki görüntüyü yan yana göster
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            
            # İlk görüntü
            axs[0].imshow(image, aspect="auto")
            if coor is not None:
                x, y = coor["x"], coor["y"]
                axs[0].scatter(x, y, color="red", s=1, alpha=0.4)
            axs[0].set_title(title)
            axs[0].axis('off')
            
            # İkinci görüntü
            axs[1].imshow(image2, aspect="auto")
            if coor2 is not None:
                x, y = coor2["x"], coor2["y"]
                axs[1].scatter(x, y, color="red", s=1, alpha=0.4)
            axs[1].set_title(title2 if title2 is not None else f"{title} (2)")
            axs[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            # Tek görüntü göster
            plt.figure(figsize=figsize)
            plt.imshow(image, aspect="auto")
            
            if coor is not None:
                x, y = coor["x"], coor["y"]
                plt.scatter(x, y, color="red", s=1, alpha=0.4)
                
            plt.axis('off')
            plt.title(title)
            plt.show()


    def get_visualize(self, id="1", spot=True):
        sample = "S_" + id
        
        if sample in self.train_slide_ids:
            image = np.array(self.train_images[sample])
            spots = self.train_spot_tables[sample]
        elif sample in self.test_slide_ids:
            image = np.array(self.test_images[sample])
            spots = self.test_spot_tables[sample]
        else:
            print("Not found", sample)

        if spot:
            self.visualize(image, coor=spots, title=sample)
        else:
            self.visualize(image, title=sample)
        


#create dataset
histodata = HistologyDataLoader()
histodata.load_train_data()
histodata.load_test_data()




histodata.get_visualize(spot=False)


img_S1 = histodata.train_images['S_1']


def check_image(img):
    """
    Görüntüyü belirtilen hedefe dönüştürür (varsayılan RGB)
    
    Parametreler:
        img: Giriş görüntüsü
        target_format: Hedef format ('RGB' veya 'BGR')
    
    Dönüş:
        İşlenmiş görüntü
    """
    # Veri tipini doğru şekilde dönüştür
    if img.dtype != np.uint8:
        if img.max() <= 1.0:  # [0,1] aralığında normalize edilmiş görüntü
            img = (img * 255).astype(np.uint8)
        else:  # Görüntü zaten uygun aralıkta
            img = img.astype(np.uint8)
    
    return img



def histopathology_tissue_mask(img):
    """
    Histopatolojik görüntüdeki doku alanını maskeleyip kenarlıktaki nokta desenini çıkaran gelişmiş fonksiyon.
    
    Parametre:
        img: İşlenecek görüntü (RGB veya BGR formatında)
        
    Dönüş:
        final_mask: Doku alanını belirten ikili maske
    """
    #1
    img_rgb = check_image(img)
    
    # 2. Çoklu doku tespiti
    # Gri tonlamalı görüntü oluştur ve gürültü azalt
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Doku tespiti için farklı yöntemler uygula
    # a. Kenarlık noktalarını tespit et
    _, dots_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # b. HSV renk uzayında mor doku tespiti - daha geniş renk aralığı
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_purple = np.array([100, 15, 20])  # Genişletilmiş renk aralığı
    upper_purple = np.array([180, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # c. Otsu thresholding ile arka plan/doku ayrımı
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # 3. Maske işlemleri
    # Tüm doku tespitlerini birleştir
    combined_tissue_mask = cv2.bitwise_or(purple_mask, otsu_mask)
    
    # Kenarlık noktalarını genişlet
    kernel_small = np.ones((5, 5), np.uint8)
    dilated_dots = cv2.dilate(dots_mask, kernel_small, iterations=2)
    
    # Doku maskesinden nokta desenini çıkar
    tissue_mask = cv2.bitwise_and(combined_tissue_mask, cv2.bitwise_not(dilated_dots))
    
    # 4. Maske iyileştirme
    # Boşlukları doldur
    kernel_large = np.ones((40, 40), np.uint8)
    filled_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Gürültüyü temizle
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 5. Büyük doku alanlarını seç
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filled_mask)
    
    for contour in contours:
        # Daha küçük alanlar için daha düşük eşik değeri
        if cv2.contourArea(contour) > 5000:  
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    # 6. Son dokunuşlar
    # İçteki küçük boşlukları doldur
    kernel_medium = np.ones((20, 20), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Kenarları yumuşat
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, kernel_small, iterations=1)
    final_mask = cv2.medianBlur(final_mask, 5)  # Kenarları düzgünleştir
    
    return final_mask


def process_histopathology_image(img):
    img_rgb = check_image(img)

    
    # 1. Doku maskesini oluştur
    mask = histopathology_tissue_mask(img)
    
    # 2. Maskeyi görüntüye uygula (sadece doku bölgesini al)
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # 3. CLAHE uygula
    # LAB renk uzayına dönüştür
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # L kanalının tipini kontrol et
    if l.dtype != np.uint8:
        l = l.astype(np.uint8)
    
    # CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    
    # Kanalları birleştir
    enhanced_lab = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return result





def improved_whitespace_filtering(img, filter_strength='moderate', preserve_tissue=True):
    """
    H&E boyalı histopatoloji görüntülerinde beyaz alanları daha hassas filtreleyen gelişmiş fonksiyon
    
    Parametreler:
        img: İşlenecek görüntü
        filter_strength: Filtreleme şiddeti ('light', 'moderate', 'aggressive')
        preserve_tissue: Doku yapısını koruma modunu etkinleştir
        
    Dönüş:
        filtered_img: Filtrelenmiş görüntü
        mask: Doku maskesi
    """
    # Görüntüyü kontrol et
    img_rgb = check_image(img)
    
    # Parlaklık ve HSV kanallarını hesapla
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Filtreleme şiddetine göre parametreleri ayarla
    if filter_strength == 'light':
        brightness_threshold = 220
        sensitivity = 0.6
    elif filter_strength == 'moderate':
        brightness_threshold = 210
        sensitivity = 0.8
    else:  # aggressive
        brightness_threshold = 200
        sensitivity = 1.0
    
    # 1. Parlaklık bazlı temel beyaz alan maskesi
    _, brightness_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Doygunluk kanalı kullanarak doku tespiti (düşük doygunluk = beyaz alanlar)
    s_channel = hsv[:,:,1]
    _, saturation_mask = cv2.threshold(s_channel, 30 * sensitivity, 255, cv2.THRESH_BINARY)
    
    # 3. H&E boyalı dokularda mor bölgeleri koru
    lower_purple = np.array([120, 20, 20])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # 4. Eosin boyalı bölgeleri (pembe) koru
    lower_pink = np.array([150, 20, 100])
    upper_pink = np.array([180, 150, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Tüm doku maskelerini birleştir
    tissue_mask = cv2.bitwise_or(cv2.bitwise_or(brightness_mask, saturation_mask), 
                                 cv2.bitwise_or(purple_mask, pink_mask))
    
    # Doku koruma seçeneği etkinse
    if preserve_tissue:
        # Zayıf dokuları da koru (daha düşük eşikle tekrar tespit et)
        _, weak_tissue_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Renk bilgisini kullanarak zayıf dokuları filtrele
        texture_enhanced = cv2.addWeighted(gray, 0.5, hsv[:,:,1], 0.5, 0)
        _, texture_mask = cv2.threshold(texture_enhanced, 30, 255, cv2.THRESH_BINARY)
        
        # İnce doku yapılarını koru
        kernel = np.ones((3,3), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_DILATE, kernel)
        tissue_mask = cv2.bitwise_or(tissue_mask, cv2.bitwise_and(weak_tissue_mask, texture_mask))
    
    # Son maske işlemleri
    kernel = np.ones((3,3), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Maskeyi orijinal görüntüye uygula
    filtered_img = cv2.bitwise_and(img_rgb, img_rgb, mask=tissue_mask)
    
    return filtered_img, tissue_mask



def histopathology_tissue_mask(img):
    img_rgb = check_image(img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenarlık noktalarını tespit et
    _, dots_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    
    # HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Hematoksilin boyalı bölgeler (mor-mavi)
    lower_purple = np.array([90, 30, 20])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Eosin boyalı bölgeler (pembe-kırmızı)
    lower_pink = np.array([150, 20, 100])
    upper_pink = np.array([180, 150, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Otsu eşikleme
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    # Tüm doku maskelerini birleştir
    combined_tissue_mask = cv2.bitwise_or(purple_mask, pink_mask)
    
    # Devam eden işlemler...
    kernel_small = np.ones((5, 5), np.uint8)
    dilated_dots = cv2.dilate(dots_mask, kernel_small, iterations=2)
    tissue_mask = cv2.bitwise_and(combined_tissue_mask, cv2.bitwise_not(dilated_dots))
    
    # Boşlukları doldur - daha küçük kernel boyutu
    kernel_large = np.ones((120, 120), np.uint8)  # 180 yerine daha küçük
    filled_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # Gürültüyü temizle
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Doku alanlarını seç - eşik değerini düşürdük
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filled_mask)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Daha fazla doku yakalamak için 800 yerine 500
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    return final_mask

    

def process_histopathology_image(img):
    """
    Histopatoloji görüntüsünü işleyip beyaz alanları kaldırarak dokuları vurgular
    """
    img_rgb = check_image(img)
    
    # Doku maskesini oluştur
    mask = histopathology_tissue_mask(img_rgb)
    
    # Maskeyi görüntüye uygula
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # CLAHE uygula (mevcut kodunuz)
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    if l.dtype != np.uint8:
        l = l.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # Gelişmiş beyaz alan filtreleme
    #result, whitespace_mask = advanced_whitespace_filtering(result)
    result, _ = improved_whitespace_filtering(result, filter_strength='light')
    
    return result



image = histodata.train_images['S_1']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))


image = histodata.train_images['S_2']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))


image = histodata.train_images['S_3']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))


image = histodata.train_images['S_4']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))


image = histodata.train_images['S_5']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))


image = histodata.train_images['S_6']
histodata.visualize(image, image2=process_histopathology_image(image),
                    comparison=True, figsize=(12,6))




