

import numpy as np
import matplotlib.pyplot as plt


# 1. FONCTION DE DISTANCE EUCLIDIENNE
def distance_euclidienne(point, centre):
    """Calcule la distance euclidienne entre un point et un centre."""
    return np.sqrt(np.sum((point - centre) ** 2))

# 2. INITIALISATION DES CENTRES (aléatoire)

def initialiser_centres(X, k):
    """Choisit k points au hasard dans X comme centres initiaux."""
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices].copy()

# 3. ASSIGNATION DES POINTS AUX CLUSTERS

def assigner_clusters(X, centres):
    """Assigne chaque point au centre le plus proche."""
    labels = []
    for point in X:
        distances = [distance_euclidienne(point, centre) for centre in centres]
        labels.append(np.argmin(distances))
    return np.array(labels)


# 4. MISE À JOUR DES CENTRES

def mettre_a_jour_centres(X, labels, k):
    """Recalcule chaque centre comme la moyenne des points assignés."""
    nouveaux_centres = np.zeros((k, X.shape[1]))
    for i in range(k):
        points_du_cluster = X[labels == i]
        if len(points_du_cluster) > 0:
            nouveaux_centres[i] = points_du_cluster.mean(axis=0)
        else:
            # Si un cluster est vide, on le réinitialise au hasard
            nouveaux_centres[i] = X[np.random.choice(len(X))]
    return nouveaux_centres


# 5. ALGORITHME PRINCIPAL : NUÉES DYNAMIQUES

def nuees_dynamiques(X, k, max_iter=100, tolerance=1e-4):
    """
    Algorithme des Nuées Dynamiques (K-Means) from scratch.
    
    Paramètres :
        X : array (n_samples, n_features) -> les données
        k : int -> nombre de clusters
        max_iter : int -> nombre maximum d'itérations
        tolerance : float -> seuil de convergence
    
    Retourne :
        centres : array (k, n_features)
        labels : array (n_samples,)
    """
    # 1. Etape d'Initialisation
    centres = initialiser_centres(X, k)
    
    for iteration in range(max_iter):
        # 2. Assignation
        labels = assigner_clusters(X, centres)
        
        # 3. Mise à jour
        anciens_centres = centres.copy()
        centres = mettre_a_jour_centres(X, labels, k)
        
        # 4. Test de convergence
        deplacement = np.sum(np.sqrt(np.sum((centres - anciens_centres) ** 2, axis=1)))
        print(f"Itération {iteration + 1} - Déplacement total des centres : {deplacement:.6f}")
        
        if deplacement < tolerance:
            print(f"✅ Convergence atteinte à l'itération {iteration + 1}")
            break
    
    return centres, labels

# 6. GÉNÉRATION DE DONNÉES DE TEST (avec numpy uniquement)

def generer_donnees(n_points=300, k=3):
    """Génère k groupes de points autour de centres aléatoires."""
    np.random.seed(42)
    X = []
    for i in range(k):
        centre = np.random.rand(2) * 10
        points = np.random.randn(n_points // k, 2) + centre
        X.append(points)
    return np.vstack(X)


# 7. PROGRAMME PRINCIPAL

if __name__ == "__main__":
    # Génération des données
    X = generer_donnees(n_points=300, k=3)
    print(f"Données générées : {X.shape[0]} points en {X.shape[1]} dimensions\n")
    
    # Lancement de l'algorithme
    k = 3
    centres, labels = nuees_dynamiques(X, k, max_iter=100, tolerance=1e-4)
    
    print("\n--- Résultats ---")
    print(f"Centres finaux :\n{centres}")
    print(f"Nombre de points par cluster : {np.bincount(labels)}")
    
     
    #Visualisation
    plt.figure(figsize=(10, 6))
    couleurs = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(k):
        points_cluster = X[labels == i]
        plt.scatter(points_cluster[:, 0], points_cluster[:, 1],
                    c=couleurs[i % len(couleurs)], label=f'Cluster {i}', alpha=0.6)
    
    plt.scatter(centres[:, 0], centres[:, 1],
                c='black', marker='X', s=300, edgecolors='yellow',
                linewidths=2, label='Centres')
    
    plt.title("Nuées Dynamiques (K-Means) - Implémentation from scratch")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()