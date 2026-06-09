import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Chemin vers les donnÃ©es
base_path = '/kaggle/input/state-farm-distracted-driver-detection'
train_path = os.path.join(base_path, 'imgs/train')

# VÃ©rifier que le chemin existe
if os.path.exists(train_path):
    print("âœ… Dossier train trouvÃ©!")
    
    # Lister toutes les classes (dossiers c0, c1, c2, etc.)
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"ğŸ“� Classes trouvÃ©es: {classes}")
    
    # Compter le nombre d'images par classe
    for class_name in classes:
        class_path = os.path.join(train_path, class_name)
        images = os.listdir(class_path)
        print(f"   {class_name}: {len(images)} images")
        
else:
    print("â�Œ Dossier train non trouvÃ©")



# Mapping des classes
class_mapping = {
    'c0': 'Conduite SÃ©curitaire',
    'c1': 'SMS Main Droite', 
    'c2': 'TÃ©lÃ©phone Main Droite',
    'c3': 'SMS Main Gauche',
    'c4': 'TÃ©lÃ©phone Main Gauche',
    'c5': 'RÃ©gler Radio',
    'c6': 'Boire',
    'c7': 'Atteindre ArriÃ¨re',
    'c8': 'Coiffure/Maquillage',
    'c9': 'Parler Passager'
}

# Visualisation de la distribution
class_counts = [2489, 2267, 2317, 2346, 2326, 2312, 2325, 2002, 1911, 2129]
class_names = [class_mapping[f'c{i}'] for i in range(10)]

plt.figure(figsize=(14, 6))
bars = plt.bar(class_names, class_counts, color=plt.cm.Set3(np.linspace(0, 1, 10)))
plt.title('Distribution des Comportements des Conducteurs', fontsize=16, fontweight='bold')
plt.ylabel('Nombre d\'Images')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("ğŸ“Š RÃ©sumÃ© du dataset:")
print(f"Total images: {sum(class_counts)}")
print(f"Nombre de classes: {len(class_counts)}")


def display_sample_images(n_per_class=2):
    """Affiche des exemples d'images pour chaque classe"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        images = os.listdir(class_path)
        
        # Prendre n_per_class images alÃ©atoires
        selected_images = np.random.choice(images, n_per_class, replace=False)
        
        # Charger et afficher la premiÃ¨re image
        img_path = os.path.join(class_path, selected_images[0])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f'{class_mapping[class_name]}\n({class_name})', 
                         fontweight='bold', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('Exemples de Comportements - State Farm Dataset', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

print("ğŸ–¼ï¸� Affichage des exemples d'images...")
display_sample_images()


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class VideoFrameExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidÃ©o: {video_path}")
        
        # Informations sur la vidÃ©o
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        print(f"ğŸ�¥ Informations vidÃ©o:")
        print(f"   ğŸ“� RÃ©solution: {self.width}x{self.height}")
        print(f"   âš¡ FPS: {self.fps:.2f}")
        print(f"   ğŸ“Š Total frames: {self.total_frames}")
        print(f"   â�±ï¸� DurÃ©e: {self.duration:.2f}s")
    
    def extract_all_frames(self, output_dir=None, skip_frames=1):
        """Extrait toutes les frames de la vidÃ©o"""
        frames = []
        frame_numbers = []
        
        print(f"ğŸ“¥ Extraction des frames (skip={skip_frames})...")
        
        for frame_num in tqdm(range(0, self.total_frames, skip_frames)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_numbers.append(frame_num)
                
                # Sauvegarde si output_dir spÃ©cifiÃ©
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    frame_filename = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
        
        self.cap.release()
        print(f"âœ… {len(frames)} frames extraites avec succÃ¨s!")
        return frames, frame_numbers
    
    def extract_frames_by_time(self, start_time=0, end_time=None, output_dir=None):
        """Extrait les frames entre start_time et end_time (en secondes)"""
        if end_time is None:
            end_time = self.duration
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        print(f"â�° Extraction des frames de {start_time}s Ã  {end_time}s...")
        return self.extract_all_frames(output_dir, skip_frames=1)
    
    def extract_sample_frames(self, n_samples=10):
        """Extrait n Ã©chantillons rÃ©partis sur toute la vidÃ©o"""
        sample_indices = np.linspace(0, self.total_frames-1, n_samples, dtype=int)
        frames = []
        
        print(f"ğŸ�¯ Extraction de {n_samples} Ã©chantillons...")
        
        for idx in tqdm(sample_indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame_rgb))
        
        self.cap.release()
        return frames
    
    def get_frame_at_time(self, time_seconds):
        """RÃ©cupÃ¨re une frame spÃ©cifique Ã  un temps donnÃ©"""
        frame_number = int(time_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, frame_number
        else:
            return None, frame_number
    
    def preview_video(self, n_frames=8):
        """AperÃ§u de la vidÃ©o avec des frames rÃ©parties"""
        frames = self.extract_sample_frames(n_frames)
        
        # Affichage
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (frame_num, frame) in enumerate(frames):
            if idx < len(axes):
                axes[idx].imshow(frame)
                time_seconds = frame_num / self.fps
                axes[idx].set_title(f'Frame {frame_num}\nTemps: {time_seconds:.1f}s')
                axes[idx].axis('off')
        
        plt.suptitle(f'AperÃ§u de la VidÃ©o: {os.path.basename(self.video_path)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return frames


!pip install yt-dlp opencv-python matplotlib tqdm numpy --quiet


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class VideoFrameAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        # Informations vidÃ©o
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        print(f"ğŸ�¥ VIDÃ‰O ANALYSÃ‰E: {os.path.basename(video_path)}")
        print(f"   ğŸ“� RÃ©solution: {self.width}x{self.height}")
        print(f"   âš¡ FPS: {self.fps:.2f}")
        print(f"   ğŸ“Š Total frames: {self.total_frames}")
        print(f"   â�±ï¸� DurÃ©e: {self.duration:.2f}s")
    
    def extract_frames(self, n_frames=12, output_dir=None):
        """Extrait n frames rÃ©parties sur la vidÃ©o - VERSION CORRIGÃ‰E"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Conversion explicite en float pour set()
        frame_indices = np.linspace(0, self.total_frames-1, n_frames, dtype=float)
        frames_data = []
        
        print(f"ğŸ“¥ Extraction de {n_frames} frames...")
        
        for idx in tqdm(frame_indices):
            # Conversion explicite en float
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = self.cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = idx / self.fps
                
                frame_info = {
                    'frame_number': int(idx),
                    'timestamp': timestamp,
                    'image': frame_rgb,
                    'image_bgr': frame
                }
                frames_data.append(frame_info)
                
                # Sauvegarder si output_dir spÃ©cifiÃ©
                if output_dir:
                    filename = f"frame_{int(idx):06d}_{timestamp:.1f}s.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
        
        self.cap.release()
        print(f"âœ… {len(frames_data)} frames extraites!")
        return frames_data
    
    def extract_frames_interval(self, interval_seconds=5, output_dir=None):
        """Extrait des frames Ã  intervalle rÃ©gulier - VERSION CORRIGÃ‰E"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        frame_interval = int(interval_seconds * self.fps)
        frames_data = []
        
        print(f"â�±ï¸� Extraction avec intervalle de {interval_seconds}s...")
        
        for frame_num in tqdm(range(0, self.total_frames, frame_interval)):
            # Conversion explicite en float
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
            ret, frame = self.cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_num / self.fps
                
                frame_info = {
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'image': frame_rgb,
                    'image_bgr': frame
                }
                frames_data.append(frame_info)
                
                if output_dir:
                    filename = f"interval_{frame_num:06d}_{timestamp:.1f}s.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
        
        self.cap.release()
        print(f"âœ… {len(frames_data)} frames extraites!")
        return frames_data
    
    def extract_sequential_frames(self, start_frame=0, num_frames=12, output_dir=None):
        """Extrait des frames sÃ©quentielles - mÃ©thode plus simple"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        frames_data = []
        
        print(f"ğŸ�¬ Extraction de {num_frames} frames sÃ©quentielles...")
        
        for frame_num in tqdm(range(start_frame, start_frame + num_frames)):
            if frame_num >= self.total_frames:
                break
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
            ret, frame = self.cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_num / self.fps
                
                frame_info = {
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'image': frame_rgb,
                    'image_bgr': frame
                }
                frames_data.append(frame_info)
                
                if output_dir:
                    filename = f"seq_{frame_num:06d}_{timestamp:.1f}s.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
        
        self.cap.release()
        print(f"âœ… {len(frames_data)} frames sÃ©quentielles extraites!")
        return frames_data
    
    def preview_frames(self, frames_data, title="Frames Extraites"):
        """AperÃ§u visuel des frames"""
        if not frames_data:
            print("â�Œ Aucune frame Ã  afficher")
            return
            
        n_frames = len(frames_data)
        cols = min(4, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        
        if rows == 1:
            if cols == 1:
                axes = [axes]
            else:
                axes = axes
        else:
            axes = axes.flatten()
        
        for i, frame_info in enumerate(frames_data):
            if i < len(axes):
                axes[i].imshow(frame_info['image'])
                axes[i].set_title(
                    f"Frame {frame_info['frame_number']}\n"
                    f"{frame_info['timestamp']:.1f}s", 
                    fontweight='bold', 
                    fontsize=10
                )
                axes[i].axis('off')
        
        # Cacher les axes vides
        for i in range(len(frames_data), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"{title}\n{os.path.basename(self.video_path)}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def analyze_video_content(self, frames_data):
        """Analyse le contenu de la vidÃ©o"""
        print("\nğŸ“Š ANALYSE DU CONTENU VIDÃ‰O")
        print("=" * 40)
        
        if not frames_data:
            print("â�Œ Aucune frame Ã  analyser")
            return
        
        timestamps = [frame['timestamp'] for frame in frames_data]
        
        print(f"ğŸ“ˆ Statistiques:")
        print(f"   Frames analysÃ©es: {len(frames_data)}")
        print(f"   Plage temporelle: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
        
        if len(timestamps) > 1:
            print(f"   Intervalle moyen: {np.mean(np.diff(timestamps)):.1f}s")
        
        # Analyse des couleurs des premiÃ¨res frames
        print(f"ğŸ�¨ Analyse des couleurs (3 premiÃ¨res frames):")
        for i, frame_info in enumerate(frames_data[:3]):
            frame = frame_info['image']
            avg_color = np.mean(frame, axis=(0, 1)).astype(int)
            print(f"   Frame {frame_info['frame_number']}: "
                  f"RGB({avg_color[0]:3d}, {avg_color[1]:3d}, {avg_color[2]:3d})")


if video_path and os.path.exists(video_path):
    print("ğŸš€ LANCEMENT DE L'ANALYSE")
    print("=" * 50)
    
    # Initialiser l'analyseur
    analyzer = VideoFrameAnalyzer(video_path)
    
    # 1. Extraire des Ã©chantillons rÃ©partis
    print("\n1ï¸�âƒ£ EXTRACTION DE FRAMES RÃ‰PARTIES")
    sample_frames = analyzer.extract_frames(
        n_frames=12,
        output_dir='/kaggle/working/sample_frames'
    )
    
    # 2. AperÃ§u visuel
    print("\n2ï¸�âƒ£ APERÃ‡U VISUEL")
    analyzer.preview_frames(sample_frames, "Frames Ã‰chantillons")
    
    # 3. Analyse du contenu
    print("\n3ï¸�âƒ£ ANALYSE DU CONTENU")
    analyzer.analyze_video_content(sample_frames)
    
    # 4. Extraction avec intervalle
    print("\n4ï¸�âƒ£ EXTRACTION AVEC INTERVALLE")
    interval_frames = analyzer.extract_frames_interval(
        interval_seconds=10,
        output_dir='/kaggle/working/interval_frames'
    )
    
    print("\nğŸ�‰ ANALYSE TERMINÃ‰E AVEC SUCCÃˆS!")
    print(f"ğŸ“� Frames Ã©chantillons: /kaggle/working/sample_frames/")
    print(f"ğŸ“� Frames intervalle: /kaggle/working/interval_frames/")
    
else:
    print("\nğŸ“‹ INSTRUCTIONS POUR UPLOADER VOTRE VIDÃ‰O:")
    print("1. Allez dans l'onglet 'Data' (icÃ´ne ğŸ“� Ã  droite)")
    print("2. Cliquez sur '+ Add data'")
    print("3. Choisissez 'Upload'")
    print("4. Glissez-dÃ©posez votre fichier: 19796441-hd_1920_1080_60fps.mp4")
    print("5. Attendez que l'upload se termine")
    print("6. RÃ©exÃ©cutez cette cellule")


!pip install ultralytics supervision --quiet


import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from tqdm import tqdm


import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from tqdm import tqdm

class CarDetector:
    def __init__(self, model_size='m'):
        """
        model_size: 'n' (rapide), 's', 'm' (Ã©quilibrÃ©), 'l' (prÃ©cis)
        """
        print(f"ğŸš— Chargement du modÃ¨le YOLOv8{model_size.upper()}...")
        
        # Charger le modÃ¨le YOLO prÃ©-entraÃ®nÃ©
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # On garde seulement la classe 'voiture' (ID 2 dans COCO)
        self.car_class_id = 2
        
        print("âœ… ModÃ¨le chargÃ©! PrÃªt Ã  dÃ©tecter les voitures.")
    
    def detect_cars(self, image_path, conf_threshold=0.3):
        """
        DÃ©tecte les voitures dans une image
        image_path: chemin vers l'image ou image numpy
        conf_threshold: Seuil de confiance (0.3 = 30%)
        """
        # Si c'est un chemin de fichier, utiliser directement
        if isinstance(image_path, str) and os.path.exists(image_path):
            # Utiliser le chemin directement pour YOLO
            results = self.model(image_path, conf=conf_threshold, verbose=False)
        else:
            # Si c'est un tableau numpy, le convertir
            if isinstance(image_path, np.ndarray):
                # YOLO accepte les images numpy en BGR
                if len(image_path.shape) == 3 and image_path.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_path
                results = self.model(image_bgr, conf=conf_threshold, verbose=False)
            else:
                raise ValueError("Type d'image non supportÃ©")
        
        # Liste pour stocker les voitures dÃ©tectÃ©es
        cars = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # VÃ©rifier si c'est une voiture
                    if class_id == self.car_class_id:
                        # RÃ©cupÃ©rer les coordonnÃ©es de la boÃ®te
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        car_info = {
                            'bbox': [x1, y1, x2, y2],  # Position
                            'confidence': confidence,    # Niveau de confiance
                            'area': (x2 - x1) * (y2 - y1)  # Surface
                        }
                        cars.append(car_info)
        
        return cars
    
    def draw_detections(self, image, cars):
        """Dessine les voitures dÃ©tectÃ©es sur l'image"""
        # Copier l'image pour ne pas modifier l'originale
        if len(image.shape) == 3:
            image_with_boxes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        else:
            image_with_boxes = image.copy()
        
        # Dessiner chaque voiture dÃ©tectÃ©e
        for car in cars:
            x1, y1, x2, y2 = car['bbox']
            confidence = car['confidence']
            
            # Rectangle vert autour de la voiture
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ã‰tiquette avec le pourcentage de confiance
            label = f"Car: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Fond blanc pour le texte
            cv2.rectangle(image_with_boxes, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (255, 255, 255), -1)
            
            # Texte noir
            cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Statistiques en haut Ã  gauche
        stats_text = f"Voitures: {len(cars)}"
        cv2.putText(image_with_boxes, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Reconvertir en RGB pour l'affichage
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        return image_with_boxes_rgb

print("âœ… Classe CarDetector corrigÃ©e avec succÃ¨s!")


    def detect_cars(self, image, conf_threshold=0.3):
        """
        DÃ©tecte les voitures dans une image
        conf_threshold: Seuil de confiance (0.3 = 30%)
        """
        # Convertir l'image en BGR (format OpenCV)
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Faire la dÃ©tection avec YOLO
        results = self.model(image_bgr, conf=conf_threshold, verbose=False)
        
        # Liste pour stocker les voitures dÃ©tectÃ©es
        cars = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # VÃ©rifier si c'est une voiture
                    if class_id == self.car_class_id:
                        # RÃ©cupÃ©rer les coordonnÃ©es de la boÃ®te
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        car_info = {
                            'bbox': [x1, y1, x2, y2],  # Position
                            'confidence': confidence,    # Niveau de confiance
                            'area': (x2 - x1) * (y2 - y1)  # Surface
                        }
                        cars.append(car_info)
        
        return cars


    def draw_detections(self, image, cars):
        """Dessine les voitures dÃ©tectÃ©es sur l'image"""
        # Copier l'image pour ne pas modifier l'originale
        if len(image.shape) == 3:
            image_with_boxes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        else:
            image_with_boxes = image.copy()
        
        # Dessiner chaque voiture dÃ©tectÃ©e
        for car in cars:
            x1, y1, x2, y2 = car['bbox']
            confidence = car['confidence']
            
            # Rectangle vert autour de la voiture
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ã‰tiquette avec le pourcentage de confiance
            label = f"Car: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Fond blanc pour le texte
            cv2.rectangle(image_with_boxes, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (255, 255, 255), -1)
            
            # Texte noir
            cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Statistiques en haut Ã  gauche
        stats_text = f"Voitures: {len(cars)}"
        cv2.putText(image_with_boxes, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Reconvertir en RGB pour l'affichage
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        return image_with_boxes_rgb


image_path="/kaggle/input/caar/keras/default/1/CAR.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convertir en uint8 normal
image_rgb = image_rgb.astype(np.uint8)

# Corriger l'encodage pour YOLO
image_for_yolo = image_rgb.copy()



import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

print("ğŸ”„ CHARGEMENT DU MODÃˆLE YOLO...")
model = YOLO('yolov8m.pt')
print("âœ… ModÃ¨le YOLO chargÃ©!")

def load_and_verify_image(image_path):
    """Charge et vÃ©rifie l'image manuellement"""
    print(f"ğŸ“� Chargement de: {image_path}")
    
    if not os.path.exists(image_path):
        print("â�Œ Fichier non trouvÃ©")
        return None
    
    # Charger avec OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("â�Œ Ã‰chec du chargement OpenCV")
        return None
    
    print(f"âœ… Image chargÃ©e: {image.shape[1]}x{image.shape[0]}")
    return image

def detect_cars_manual(image):
    try:
        img = image.astype(np.uint8)

        img = cv2.resize(img, (1280, 720))

        results = model(img, conf=0.25, verbose=False)

        cars = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Classe 2 = voiture
                    if class_id == 2:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cars.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'area': (x2 - x1) * (y2 - y1)
                        })

        return cars

    except Exception as e:
        print("â�Œ Erreur de dÃ©tection:", e)
        return []

def draw_detections_manual(image, cars):
    """Dessine les dÃ©tections sur l'image"""
    image_with_boxes = image.copy()
    
    for car in cars:
        x1, y1, x2, y2 = car['bbox']
        confidence = car['confidence']
        
        # Rectangle vert
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Ã‰tiquette
        label = f"Car: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Fond pour le texte
        cv2.rectangle(image_with_boxes, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     (0, 255, 0), -1)
        
        # Texte blanc
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Compteur
    stats_text = f"Voitures: {len(cars)}"
    cv2.putText(image_with_boxes, stats_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return image_with_boxes


import os

root = "/kaggle/input"
for path, dirs, files in os.walk(root):
    print(path, files)



def test_car_jpeg_direct():
    """Test direct avec CAR.jpeg"""
    print("ğŸ§ª TEST DIRECT AVEC CAR.JPEG")
    print("=" * 40)
    
    image_path = "/kaggle/input/caar/keras/default/1/CAR.jpg"
    
    # 1. Charger l'image
    image_bgr = load_and_verify_image(image_path)
    if image_bgr is None:
        return
    
    # 2. Convertir en RGB pour l'affichage
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. DÃ©tection des voitures
    print("ğŸ”� DÃ©tection en cours...")
    cars = detect_cars_manual(image_bgr)  # Utiliser l'image BGR pour YOLO
    
    # 4. Affichage des rÃ©sultats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Image originale
    ax1.imshow(image_rgb)
    ax1.set_title('IMAGE ORIGINALE', fontweight='bold', fontsize=16, pad=20)
    ax1.axis('off')
    
    # Image avec dÃ©tections
    result_image = draw_detections_manual(image_rgb, cars)
    ax2.imshow(result_image)
    ax2.set_title(f'DÃ‰TECTION - {len(cars)} VOITURE(S)', 
                 fontweight='bold', fontsize=16, pad=20)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 5. RÃ©sultats
    print(f"\nğŸ“Š RÃ‰SULTATS:")
    print(f"   ğŸš— Voitures dÃ©tectÃ©es: {len(cars)}")
    
    if cars:
        for i, car in enumerate(cars):
            x1, y1, x2, y2 = car['bbox']
            print(f"   ğŸ“� Voiture {i+1}:")
            print(f"      â€¢ Confiance: {car['confidence']:.3f}")
            print(f"      â€¢ Position: [{x1}, {y1}, {x2}, {y2}]")
            print(f"      â€¢ Surface: {car['area']} pixels")
    else:
        print("   â�Œ Aucune voiture dÃ©tectÃ©e")

# Test direct
test_car_jpeg_direct()

