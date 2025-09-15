import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class BayesianClassifier:
    def __init__(self):
        self.prior_lesion = None
        self.prior_non_lesion = None
        self.mean_lesion = None
        self.mean_non_lesion = None
        self.cov_lesion = None
        self.cov_non_lesion = None
        self.threshold = None
        
    def fit(self, X, y):
        # Priors
        self.prior_lesion = np.mean(y == 1)
        self.prior_non_lesion = np.mean(y == 0)
        
        # Medias y covarianzas
        self.mean_lesion = np.mean(X[y == 1], axis=0)
        self.mean_non_lesion = np.mean(X[y == 0], axis=0)
        self.cov_lesion = np.cov(X[y == 1].T)
        self.cov_non_lesion = np.cov(X[y == 0].T)
        
        # Añadir pequeña constante para estabilidad numérica
        self.cov_lesion += np.eye(3) * 1e-6
        self.cov_non_lesion += np.eye(3) * 1e-6
        
    def likelihood_ratio(self, X):
        # Distribuciones normales multivariadas
        pdf_lesion = multivariate_normal.pdf(X, mean=self.mean_lesion, cov=self.cov_lesion, allow_singular=True)
        pdf_non_lesion = multivariate_normal.pdf(X, mean=self.mean_non_lesion, cov=self.cov_non_lesion, allow_singular=True)
        
        return pdf_lesion / pdf_non_lesion
    
    def predict_proba(self, X):
        lr = self.likelihood_ratio(X)
        proba_lesion = lr / (1 + lr)
        return proba_lesion
    
    def set_threshold_youden(self, X_val, y_val):
        lr_val = self.likelihood_ratio(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, lr_val)
        youden_idx = np.argmax(tpr - fpr)
        self.threshold = thresholds[youden_idx]
        return self.threshold
    
    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold_youden() first.")
        
        lr = self.likelihood_ratio(X)
        return (lr > self.threshold).astype(int)

class BayesianPCAClassifier:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.bayesian = BayesianClassifier()
        self.n_components = n_components
        
    def fit(self, X, y):
        # Aplicar PCA
        X_pca = self.pca.fit_transform(X)
        print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} componentes (varianza explicada: {np.sum(self.pca.explained_variance_ratio_):.3f})")
        
        # Entrenar clasificador bayesiano en espacio PCA
        self.bayesian.fit(X_pca, y)
        
    def set_threshold_youden(self, X_val, y_val):
        X_val_pca = self.pca.transform(X_val)
        return self.bayesian.set_threshold_youden(X_val_pca, y_val)
    
    def predict_proba(self, X):
        X_pca = self.pca.transform(X)
        return self.bayesian.predict_proba(X_pca)
    
    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.bayesian.predict(X_pca)

def evaluate_classifier(classifier, X_test, y_test, name="Classifier"):
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    specificity = recall_score(1 - y_test, 1 - y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def plot_roc_curves(results_dict, y_true):
    plt.figure(figsize=(10, 8))
    
    for name, results in results_dict.items():
        if 'y_proba' in results:
            fpr, tpr, _ = roc_curve(y_true, results['y_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('curvas_roc.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_kmeans_to_images(images, n_clusters=2):
    """
    Aplica K-Means a cada imagen individualmente
    """
    segmented_masks = []
    
    for img in images:
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reconstruir máscara
        segmented_mask = labels.reshape(img.shape[:2])
        segmented_masks.append(segmented_mask)
    
    return np.array(segmented_masks)