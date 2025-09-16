import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def load_dataset(data_path, img_size=(256, 256)):

    images_dir = os.path.join(data_path, 'images')
    masks_dir = os.path.join(data_path, 'masks')
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    images = []
    masks = []
    
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Loading data"):
        # Cargar imagen
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img)
        
        # Cargar máscara
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = (mask > 128).astype(np.uint8)  # Binarizar
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def split_dataset(images, masks, test_size=0.2, val_size=0.2, random_state=42):
    """
    Divide el dataset en train, validation y test por imagen
    """
    # Primera división: train + temp vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )
    
    # Segunda división: train vs validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def extract_pixels_with_balance(images, masks, max_samples_per_class=10000):
    """
    Extrae píxeles balanceados de imágenes y máscaras
    """
    all_pixels = []
    all_labels = []
    
    for img, mask in zip(images, masks):
        pixels = img.reshape(-1, 3)
        labels = mask.reshape(-1)
        
        all_pixels.append(pixels)
        all_labels.append(labels)
    
    all_pixels = np.vstack(all_pixels)
    all_labels = np.hstack(all_labels)
    
    # Balancear clases
    lesion_idx = np.where(all_labels == 1)[0]
    non_lesion_idx = np.where(all_labels == 0)[0]
    
    # Submuestreo si es necesario
    if len(lesion_idx) > max_samples_per_class:
        lesion_idx = np.random.choice(lesion_idx, max_samples_per_class, replace=False)
    if len(non_lesion_idx) > max_samples_per_class:
        non_lesion_idx = np.random.choice(non_lesion_idx, max_samples_per_class, replace=False)
    
    balanced_idx = np.concatenate([lesion_idx, non_lesion_idx])
    np.random.shuffle(balanced_idx)
    
    return all_pixels[balanced_idx], all_labels[balanced_idx]

def plot_histograms(X_train, y_train):
    """
    Grafica histogramas de canales RGB para lesión vs no-lesión
    """
    lesion_pixels = X_train[y_train == 1]
    non_lesion_pixels = X_train[y_train == 0]
    
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        axes[i].hist(lesion_pixels[:, i], bins=50, alpha=0.7, color=colors[i], 
                    label='Lesión', density=True)
        axes[i].hist(non_lesion_pixels[:, i], bins=50, alpha=0.7, color='gray', 
                    label='No-lesión', density=True)
        axes[i].set_title(f'Canal {channels[i]}')
        axes[i].set_xlabel('Intensidad')
        axes[i].set_ylabel('Densidad')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histogramas_rgb.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estadísticas
    print("Estadísticas para lesión:")
    print(f"  R: μ={np.mean(lesion_pixels[:, 0]):.2f}, σ={np.std(lesion_pixels[:, 0]):.2f}")
    print(f"  G: μ={np.mean(lesion_pixels[:, 1]):.2f}, σ={np.std(lesion_pixels[:, 1]):.2f}")
    print(f"  B: μ={np.mean(lesion_pixels[:, 2]):.2f}, σ={np.std(lesion_pixels[:, 2]):.2f}")
    
    print("\nEstadísticas para no-lesión:")
    print(f"  R: μ={np.mean(non_lesion_pixels[:, 0]):.2f}, σ={np.std(non_lesion_pixels[:, 0]):.2f}")
    print(f"  G: μ={np.mean(non_lesion_pixels[:, 1]):.2f}, σ={np.std(non_lesion_pixels[:, 1]):.2f}")
    print(f"  B: μ={np.mean(non_lesion_pixels[:, 2]):.2f}, σ={np.std(non_lesion_pixels[:, 2]):.2f}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def mask_to_uint8(mask):
    """
    Convierte una máscara a uint8 [0,255].
    Acepta máscaras booleanas, 0/1, binarias en float o ya en uint8.
    """
    if mask.dtype == np.uint8:
        return mask
    # si es booleano
    if mask.dtype == bool:
        return (mask.astype(np.uint8) * 255)
    # si tiene valores entre 0 y 1 (float)
    if mask.max() <= 1.0:
        return (mask * 255).astype(np.uint8)
    # si tiene otros rangos, normalizamos a 0-255
    m = mask.astype(np.float32)
    m -= m.min()
    if m.max() > 0:
        m = m / m.max()
    return (m * 255).astype(np.uint8)

def save_masks_as_images(pred_masks, true_masks, output_dir, prefix="img"):
    """
    Guarda pares (pred, true) como PNG en output_dir.
    pred_masks y true_masks: listas o iterables de arrays 2D (H,W) con valores 0/1 o 0-255.
    """
    ensure_dir(output_dir)
    for idx, (pred, true) in enumerate(zip(pred_masks, true_masks)):
        pred_arr = mask_to_uint8(np.array(pred))
        true_arr = mask_to_uint8(np.array(true))

        # Asegurar que son 2D
        if pred_arr.ndim == 3 and pred_arr.shape[2] == 3:
            im_pred = Image.fromarray(pred_arr)
        else:
            im_pred = Image.fromarray(pred_arr, mode="L")

        if true_arr.ndim == 3 and true_arr.shape[2] == 3:
            im_true = Image.fromarray(true_arr)
        else:
            im_true = Image.fromarray(true_arr, mode="L")

        im_pred.save(os.path.join(output_dir, f"{prefix}_pred_{idx}.png"))
        im_true.save(os.path.join(output_dir, f"{prefix}_true_{idx}.png"))
