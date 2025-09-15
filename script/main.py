import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, split_dataset, extract_pixels_with_balance, plot_histograms
from classifiers import BayesianClassifier, BayesianPCAClassifier, evaluate_classifier, plot_roc_curves, apply_kmeans_to_images
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

if __name__ == "__main__":
    main()