"""
MODULO: Recommender System con Matrix Factorization (AUTO-INSTALLANTE)
Versione stabile per sistemi di raccomandazione con SGD e regolarizzazione.
"""

# =====================================================
# SEZIONE 1: AUTO-INSTALLAZIONE LIBRERIE
# =====================================================
import subprocess
import sys
import importlib.util

def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"📦 Installazione di '{package_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ '{package_name}' installato.")
    else:
        print(f"✅ '{package_name}' già presente.")

# Dipendenze necessarie
install_and_import("numpy")
install_and_import("pandas")
install_and_import("scikit-learn", "sklearn")
install_and_import("tqdm")

# =====================================================
# SEZIONE 2: IMPORT LIBRERIE
# =====================================================
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
from tqdm import tqdm
import json

# =====================================================
# SEZIONE 3: CONFIGURAZIONE LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# SEZIONE 4: CLASSE PRINCIPALE
# =====================================================
class MatrixFactorization:
    """
    Implementazione di Matrix Factorization con Stochastic Gradient Descent (SGD).
    Scopo: predire rating utente-item e generare raccomandazioni personalizzate.
    
    Come funziona:
    - Decomponiamo la matrice utenti-item in due matrici più piccole: P (utenti x fattori) e Q (item x fattori).
    - Il rating predetto è il prodotto scalare tra il vettore dell'utente e quello dell'item.
    - Addestriamo con SGD minimizzando l'errore quadratico + regolarizzazione.
    """
    
    def __init__(
        self,
        n_factors: int = 50,          # Numero di fattori latenti (più fattori = più capacità, ma più overfitting)
        learning_rate: float = 0.01,   # Velocità di apprendimento
        reg_param: float = 0.02,       # Forza della regolarizzazione (previene overfitting)
        n_iterations: int = 100        # Numero di passaggi su tutto il dataset
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_iterations = n_iterations
        self.P = None  # Matrice utenti (n_utenti x n_fattori)
        self.Q = None  # Matrice item (n_item x n_fattori)
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0
        logger.info(f"MatrixFactorization avviato con {n_factors} fattori")

    def fit(self, data: pd.DataFrame) -> None:
        """
        Addestra il modello sui rating.
        data: DataFrame con colonne ['user_id', 'item_id', 'rating']
        """
        logger.info("Inizio addestramento...")
        
        # Estrai colonne
        user_ids = data['user_id'].values
        item_ids = data['item_id'].values
        ratings = data['rating'].values
        
        # Crea mappe ID -> indice (perché gli ID potrebbero non essere sequenziali)
        self.user_map = {uid: i for i, uid in enumerate(np.unique(user_ids))}
        self.item_map = {iid: i for i, iid in enumerate(np.unique(item_ids))}
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        
        # Converti in indici
        user_idx = np.array([self.user_map[uid] for uid in user_ids])
        item_idx = np.array([self.item_map[iid] for iid in item_ids])
        
        # Media globale dei rating
        self.global_mean = np.mean(ratings)
        
        # Inizializza matrici con valori casuali piccoli
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        # Training con SGD
        for iteration in range(self.n_iterations):
            total_error = 0
            n_samples = len(user_idx)
            
            # Shuffle dei dati per evitare che il modello impari l'ordine
            indices = np.random.permutation(n_samples)
            user_idx_shuffled = user_idx[indices]
            item_idx_shuffled = item_idx[indices]
            ratings_shuffled = ratings[indices]
            
            for i in range(n_samples):
                u = user_idx_shuffled[i]
                it = item_idx_shuffled[i]
                rating = ratings_shuffled[i]
                
                # Predizione: media globale + bias utente + bias item + prodotto scalare
                pred = self.global_mean + self.user_bias[u] + self.item_bias[it] + np.dot(self.P[u], self.Q[it])
                
                # Errore
                error = rating - pred
                total_error += error ** 2
                
                # AGGIORNAMENTO GRADIENTI (SGD)
                # Bias utente
                self.user_bias[u] += self.learning_rate * (error - self.reg_param * self.user_bias[u])
                # Bias item
                self.item_bias[it] += self.learning_rate * (error - self.reg_param * self.item_bias[it])
                
                # Vettori latenti
                p_u = self.P[u]
                q_i = self.Q[it]
                self.P[u] += self.learning_rate * (error * q_i - self.reg_param * p_u)
                self.Q[it] += self.learning_rate * (error * p_u - self.reg_param * q_i)
            
            # Log RMSE ogni 10 iterazioni
            if (iteration + 1) % 10 == 0:
                rmse = np.sqrt(total_error / n_samples)
                logger.info(f"Iterazione {iteration+1}/{self.n_iterations} - RMSE: {rmse:.4f}")

    def predict(self, user_id: int, item_id: int) -> float:
        """Predice il rating per una coppia utente-item."""
        try:
            u = self.user_map[user_id]
            it = self.item_map[item_id]
            pred = self.global_mean + self.user_bias[u] + self.item_bias[it] + np.dot(self.P[u], self.Q[it])
            return np.clip(pred, 1.0, 5.0)  # Clip nel range dei rating
        except KeyError:
            return self.global_mean  # Se utente/item non noto, restituisce la media

    def predict_for_user(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Genera le top-N raccomandazioni per un utente."""
        u = self.user_map.get(user_id)
        if u is None:
            logger.warning(f"Utente {user_id} non trovato")
            return []
        
        predictions = []
        for item_id, it in self.item_map.items():
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Trova item simili basandosi sui fattori latenti (coseno similarità)."""
        if item_id not in self.item_map:
            return []
        
        it = self.item_map[item_id]
        item_vector = self.Q[it]
        
        similarities = []
        for other_item_id, other_it in self.item_map.items():
            if other_item_id != item_id:
                other_vector = self.Q[other_it]
                similarity = np.dot(item_vector, other_vector) / (
                    np.linalg.norm(item_vector) * np.linalg.norm(other_vector) + 1e-8
                )
                similarities.append((other_item_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

    def save_model(self, path: str) -> None:
        """Salva il modello in JSON."""
        model_data = {
            'P': self.P.tolist(),
            'Q': self.Q.tolist(),
            'user_bias': self.user_bias.tolist(),
            'item_bias': self.item_bias.tolist(),
            'global_mean': self.global_mean,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'n_factors': self.n_factors
        }
        with open(path, 'w') as f:
            json.dump(model_data, f)
        logger.info(f"Modello salvato in {path}")

    def load_model(self, path: str) -> None:
        """Carica il modello da JSON."""
        with open(path, 'r') as f:
            model_data = json.load(f)
        self.P = np.array(model_data['P'])
        self.Q = np.array(model_data['Q'])
        self.user_bias = np.array(model_data['user_bias'])
        self.item_bias = np.array(model_data['item_bias'])
        self.global_mean = model_data['global_mean']
        self.user_map = model_data['user_map']
        self.item_map = model_data['item_map']
        self.n_factors = model_data['n_factors']
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        logger.info(f"Modello caricato da {path}")

# =====================================================
# SEZIONE 5: FUNZIONE MAIN (DEMO)
# =====================================================
def main():
    logger.info("=== Demo Recommender System ===")
    
    # Genera dati sintetici: 100 utenti, 50 item, 1000 rating casuali
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.normal(3, 1, n_ratings).clip(1, 5)
    
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Split train/test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Addestra modello
    model = MatrixFactorization(n_factors=20, learning_rate=0.01, reg_param=0.02, n_iterations=30)
    model.fit(train_data)
    
    # Valuta su test
    predictions = []
    actual = []
    for _, row in test_data.iterrows():
        pred = model.predict(row['user_id'], row['item_id'])
        predictions.append(pred)
        actual.append(row['rating'])
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    logger.info(f"RMSE su test: {rmse:.4f}")
    
    # Genera raccomandazioni per un utente
    recs = model.predict_for_user(1, n_recommendations=5)
    logger.info(f"Raccomandazioni per utente 1: {recs}")
    
    # Trova item simili
    similar = model.get_similar_items(10, n_similar=5)
    logger.info(f"Item simili a 10: {similar}")
    
    print("\n✅ Programma completato con successo!")

if __name__ == "__main__":
    main()