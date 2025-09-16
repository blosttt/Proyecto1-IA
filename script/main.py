import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, split_dataset, extract_pixels_with_balance, plot_histograms, save_masks_as_images
from classifiers import BayesianClassifier, BayesianPCAClassifier, evaluate_classifier, plot_roc_curves, apply_kmeans_to_images
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import os

def main():
    # Configuración
    DATA_PATH = 'data'  # Ruta al dataset
    RANDOM_STATE = 42
    MAX_SAMPLES = 10000
    
    print("Cargando dataset...")
    images, masks = load_dataset(DATA_PATH)
    print(f"Dataset cargado: {len(images)} imágenes, shape: {images[0].shape}")
    
    # Dividir dataset
    print("\nDividiendo dataset...")
    (X_train_img, y_train_img), (X_val_img, y_val_img), (X_test_img, y_test_img) = split_dataset(
        images, masks, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_train_img)} imágenes")
    print(f"Validation: {len(X_val_img)} imágenes")
    print(f"Test: {len(X_test_img)} imágenes")
    
    # Extraer píxeles balanceados para entrenamiento
    print("\nExtrayendo píxeles balanceados...")
    X_train, y_train = extract_pixels_with_balance(X_train_img, y_train_img, MAX_SAMPLES)
    X_val, y_val = extract_pixels_with_balance(X_val_img, y_val_img, MAX_SAMPLES)
    
    print(f"Píxeles de entrenamiento: {X_train.shape}")
    print(f"Píxeles de validación: {X_val.shape}")
    
    # Visualización de histogramas
    print("\nGenerando histogramas...")
    plot_histograms(X_train, y_train)
    
    # Normalización
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 1. Clasificador Bayesiano RGB
    print("\n" + "="*50)
    print("1. ENTRENANDO CLASIFICADOR BAYESIANO RGB")
    print("="*50)
    
    bayesian_rgb = BayesianClassifier()
    bayesian_rgb.fit(X_train_scaled, y_train)
    threshold_rgb = bayesian_rgb.set_threshold_youden(X_val_scaled, y_val)
    print(f"Umbral óptimo (Youden): {threshold_rgb:.4f}")
    
    # 2. Clasificador Bayesiano + PCA
    print("\n" + "="*50)
    print("2. ENTRENANDO CLASIFICADOR BAYESIANO + PCA")
    print("="*50)
    
    bayesian_pca = BayesianPCAClassifier(n_components=0.95)
    bayesian_pca.fit(X_train_scaled, y_train)
    threshold_pca = bayesian_pca.set_threshold_youden(X_val_scaled, y_val)
    print(f"Umbral óptimo (Youden): {threshold_pca:.4f}")
    
    # Preparar datos de test
    X_test_flat = np.vstack([img.reshape(-1, 3) for img in X_test_img])
    y_test_flat = np.hstack([mask.reshape(-1) for mask in y_test_img])
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Evaluar clasificadores
    results = {}
    
    results['Bayesian RGB'] = evaluate_classifier(bayesian_rgb, X_test_scaled, y_test_flat, "Bayesian RGB")
    results['Bayesian PCA'] = evaluate_classifier(bayesian_pca, X_test_scaled, y_test_flat, "Bayesian PCA")
    
    # 3. K-Means
    print("\n" + "="*50)
    print("3. APLICANDO K-MEANS")
    print("="*50)
    
    kmeans_masks = apply_kmeans_to_images(X_test_img)
    
    # Evaluar K-Means
    y_pred_kmeans = np.hstack([mask.reshape(-1) for mask in kmeans_masks])
    
    # Asegurar que las etiquetas estén alineadas (K-Means puede invertir las etiquetas)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_flat, y_pred_kmeans)
    if cm[0, 0] + cm[1, 1] < cm[0, 1] + cm[1, 0]:
        y_pred_kmeans = 1 - y_pred_kmeans  # Invertir etiquetas
    
    accuracy_kmeans = accuracy_score(y_test_flat, y_pred_kmeans)
    precision_kmeans = precision_score(y_test_flat, y_pred_kmeans, zero_division=0)
    recall_kmeans = recall_score(y_test_flat, y_pred_kmeans, zero_division=0)
    specificity_kmeans = recall_score(1 - y_test_flat, 1 - y_pred_kmeans, zero_division=0)
    f1_kmeans = f1_score(y_test_flat, y_pred_kmeans, zero_division=0)
    
    print("\nK-Means Results:")
    print(f"Accuracy: {accuracy_kmeans:.4f}")
    print(f"Precision: {precision_kmeans:.4f}")
    print(f"Recall (Sensitivity): {recall_kmeans:.4f}")
    print(f"Specificity: {specificity_kmeans:.4f}")
    print(f"F1-Score: {f1_kmeans:.4f}")
    
    results['K-Means'] = {
        'accuracy': accuracy_kmeans,
        'precision': precision_kmeans,
        'recall': recall_kmeans,
        'specificity': specificity_kmeans,
        'f1': f1_kmeans,
        'y_pred': y_pred_kmeans
    }
    
    # Curvas ROC
    print("\nGenerando curvas ROC...")
    plot_roc_curves({k: v for k, v in results.items() if k != 'K-Means'}, y_test_flat)
    
    # Métricas a nivel de imagen (Jaccard)
    print("\n" + "="*50)
    print("MÉTRICAS A NIVEL DE IMAGEN (JACCARD)")
    print("="*50)
    
    from sklearn.metrics import jaccard_score
    
    def calculate_jaccard_per_image(true_masks, pred_masks):
        jaccard_scores = []
        for true_mask, pred_mask in zip(true_masks, pred_masks):
            jaccard = jaccard_score(true_mask.reshape(-1), pred_mask.reshape(-1), average='binary')
            jaccard_scores.append(jaccard)
        return np.mean(jaccard_scores)
    
    # Para Bayesian RGB
    bayesian_preds = bayesian_rgb.predict(X_test_scaled)
    bayesian_masks = [pred.reshape(mask.shape) for pred, mask in zip(
        np.split(bayesian_preds, len(X_test_img)), y_test_img
    )]
    jaccard_bayesian = calculate_jaccard_per_image(y_test_img, bayesian_masks)
    
    # Para Bayesian PCA
    bayesian_pca_preds = bayesian_pca.predict(X_test_scaled)
    bayesian_pca_masks = [pred.reshape(mask.shape) for pred, mask in zip(
        np.split(bayesian_pca_preds, len(X_test_img)), y_test_img
    )]
    jaccard_bayesian_pca = calculate_jaccard_per_image(y_test_img, bayesian_pca_masks)
    
    # Para K-Means
    jaccard_kmeans = calculate_jaccard_per_image(y_test_img, kmeans_masks)
    
    print(f"Jaccard Bayesian RGB: {jaccard_bayesian:.4f}")
    print(f"Jaccard Bayesian PCA: {jaccard_bayesian_pca:.4f}")
    print(f"Jaccard K-Means: {jaccard_kmeans:.4f}")
    
    # Guardar resultados
    print("\n" + "="*50)
    print("RESUMEN FINAL")
    print("="*50)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall: {res['recall']:.4f}")
        print(f"  Specificity: {res['specificity']:.4f}")
        print(f"  F1-Score: {res['f1']:.4f}")
    
    print(f"\nJaccard Index:")
    print(f"  Bayesian RGB: {jaccard_bayesian:.4f}")
    print(f"  Bayesian PCA: {jaccard_bayesian_pca:.4f}")
    print(f"  K-Means: {jaccard_kmeans:.4f}")

    print("\nGuardando imágenes procesadas en 'processed/' ...")
    save_masks_as_images(bayesian_masks, y_test_img, output_dir="processed/bayesian_rgb", prefix="img")
    save_masks_as_images(bayesian_pca_masks, y_test_img, output_dir="processed/bayesian_pca", prefix="img")
    save_masks_as_images(kmeans_masks, y_test_img, output_dir="processed/kmeans", prefix="img")
    print("Guardado finalizado. Revise la carpeta 'processed/'.")

    def _to_uint8_image(img):
        """Convierte image HxWx3 o HxW (0-1 o 0-255) a uint8 0-255 HxWx3."""
        arr = np.array(img)
        if arr.dtype == np.uint8:
            pass
        else:
            # normalizar si float
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        # si es grayscale convertir a 3 canales
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[2] == 4:  # eliminar alfa si existe
            arr = arr[..., :3]
        return arr

    def _binarize_mask(mask, thresh=0.5):
        """Convierte máscara a 0/1 numpy uint8."""
        m = np.array(mask)
        if m.dtype == np.bool_:
            return m.astype(np.uint8)
        if m.dtype == np.uint8 and m.max() > 1:
            return (m > 127).astype(np.uint8)
        if m.max() <= 1.0:
            return (m >= thresh).astype(np.uint8)
        return (m > 0).astype(np.uint8)

    def overlay_mask_rgb(image_rgb, mask_bin, color=(255,0,0), alpha=0.5):
        """
        Superpone máscara binaria (0/1) sobre imagen RGB uint8.
        color: tupla 0-255
        """
        img = _to_uint8_image(image_rgb).copy()
        mask = _binarize_mask(mask_bin).astype(bool)
        if mask.shape != img.shape[:2]:
            raise ValueError(f"Shape mismatch: image {img.shape[:2]} vs mask {mask.shape}")
        # aplicar overlay
        col_arr = np.zeros_like(img, dtype=np.uint8)
        col_arr[...,0] = color[0]
        col_arr[...,1] = color[1]
        col_arr[...,2] = color[2]
        img_masked = img.copy()
        img_masked[mask] = (img[mask] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
        return img_masked

    def make_comparison_panel(image, gt_mask, pred_mask):
        """
        Devuelve una imagen PIL con 3 columnas: [original | GT overlay | Pred overlay]
        Adicional: debajo de cada overlay, una mini máscara binaria (opcional).
        """
        # normalizar / validar
        img_rgb = _to_uint8_image(image)
        gt_b = _binarize_mask(gt_mask)
        pred_b = _binarize_mask(pred_mask)

        H, W = img_rgb.shape[:2]
        # overlays
        gt_overlay = overlay_mask_rgb(img_rgb, gt_b, color=(0,255,0), alpha=0.5)    # verde GT
        pred_overlay = overlay_mask_rgb(img_rgb, pred_b, color=(255,0,0), alpha=0.5) # rojo pred

        # preparar máscaras visuales (convertir a 3 canales 0/255)
        gt_vis = (gt_b * 255).astype(np.uint8)
        pred_vis = (pred_b * 255).astype(np.uint8)
        gt_vis_rgb = np.stack([gt_vis]*3, axis=-1)
        pred_vis_rgb = np.stack([pred_vis]*3, axis=-1)

        # construir panel: dos filas (principal + masks reducidas)
        # fila principal: original | gt_overlay | pred_overlay  (cada W x H)
        top = np.concatenate([img_rgb, gt_overlay, pred_overlay], axis=1)

        # fila inferior: mask originals (redimensionadas en altura, p.e. 1/4)
        mask_h = max( int(H * 0.25), 20 )
        # resize masks using PIL to keep aspect ratio / smoothing
        top_pil = Image.fromarray(top)
        gt_mask_pil = Image.fromarray(gt_vis_rgb).resize((W, mask_h), Image.NEAREST)
        pred_mask_pil = Image.fromarray(pred_vis_rgb).resize((W, mask_h), Image.NEAREST)
        original_mask_pil = Image.fromarray(np.stack([(_binarize_mask(np.zeros((H,W))) * 255)]*3, axis=-1)).resize((W, mask_h), Image.NEAREST)
        # compose bottom: blank | gt_mask | pred_mask
        bottom = Image.new("RGB", (W*3, mask_h))
        bottom.paste(original_mask_pil, (0,0))
        bottom.paste(gt_mask_pil, (W,0))
        bottom.paste(pred_mask_pil, (W*2,0))

        # concatenar verticalmente
        top_pil = top_pil.convert("RGB")
        full_h = H + mask_h
        comp = Image.new("RGB", (W*3, full_h))
        comp.paste(top_pil, (0,0))
        comp.paste(bottom, (0,H))
        return comp

    def save_all_comparisons(images, gt_masks, pred_masks, out_dir="comparisons/images", prefix="comp"):
        """
        Guarda una imagen compuesta por muestra en out_dir:
        out_dir/prefix_0001.png, ...
        Requiere len(images)==len(gt_masks)==len(pred_masks)
        """
        os.makedirs(out_dir, exist_ok=True)
        n = len(images)
        if not (len(gt_masks)==n and len(pred_masks)==n):
            raise ValueError("Las listas deben tener la misma longitud")
        for i, (img, gt, pred) in enumerate(zip(images, gt_masks, pred_masks)):
            try:
                comp = make_comparison_panel(img, gt, pred)
            except Exception as e:
                print(f"[WARN] fallo en sample {i}: {e}")
                continue
            fname = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
            comp.save(fname, format="PNG")
        print(f"Guardadas {n} comparaciones en: {out_dir}")

    # comparar Bayes RGB (pred) contra GT
    save_all_comparisons(X_test_img, y_test_img, bayesian_masks, out_dir="comparisons/bayesian_images", prefix="bayes")

    # comparar Bayes PCA
    save_all_comparisons(X_test_img, y_test_img, bayesian_pca_masks, out_dir="comparisons/bayesian_pca_images", prefix="bayes_pca")

    # comparar KMeans
    save_all_comparisons(X_test_img, y_test_img, kmeans_masks, out_dir="comparisons/kmeans_images", prefix="kmeans")


if __name__ == "__main__":
    main()
    