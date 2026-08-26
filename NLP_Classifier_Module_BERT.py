"""
MODULO: NLP Classifier (CORRETTO E AUTO-INSTALLANTE)
Versione stabile per classificazione di testi clinici con BERT.
Questo programma è progettato per studenti e per chi vuole testare subito un modello NLP.
"""

# =====================================================
# SEZIONE 1: AUTO-INSTALLAZIONE LIBRERIE (FATTO AUTOMATICAMENTE)
# =====================================================
# Spiegazione: Questa funzione controlla se le librerie necessarie sono presenti.
# Se mancano, le installa con "pip" senza che l'utente debba fare nulla.
import subprocess
import sys
import importlib.util
import os

def install_and_import(package_name, import_name=None):
    """
    Tenta di importare una libreria; se non esiste, la installa automaticamente.
    
    Args:
        package_name: Nome del pacchetto su PyPI (es. 'torch')
        import_name: Nome con cui importarlo in Python (es. 'torch'), se diverso.
    """
    if import_name is None:
        import_name = package_name
    
    # Verifica se il modulo è già disponibile
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"📦 Libreria '{package_name}' non trovata. Installazione in corso...")
        # Esegue il comando: pip install <pacchetto>
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ '{package_name}' installato con successo.")
    else:
        print(f"✅ Libreria '{package_name}' già presente.")

# Installazione di TUTTE le librerie necessarie per questo script
# Nota: 'torch' deve essere installato prima di 'transformers' per evitare conflitti.
install_and_import("torch")
install_and_import("transformers")
install_and_import("scikit-learn", "sklearn")  # Su PyPi si chiama scikit-learn, ma si importa come sklearn
install_and_import("pandas")
install_and_import("numpy")
install_and_import("tqdm")

# =====================================================
# SEZIONE 2: IMPORT DELLE LIBRERIE (ORA TUTTE DISPONIBILI)
# =====================================================
# Spiegazione: Dopo l'installazione automatica, possiamo importare tutto serenamente.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ATTENZIONE: QUI ABBIAMO RISOLTO L'ERRORE!
# Invece di importare AdamW da 'transformers' (che causava errore),
# lo importiamo direttamente da 'torch.optim', che è la fonte ufficiale e stabile.
from torch.optim import AdamW  

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# =====================================================
# SEZIONE 3: CONFIGURAZIONE DEL LOG (PER VISUALIZZARE AVANZAMENTO)
# =====================================================
# Spiegazione: Il logging ci mostra a schermo cosa sta facendo il programma.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# SEZIONE 4: CLASSE PER IL DATASET (GESTISCE I DATI)
# =====================================================
class ClinicalTextDataset(Dataset):
    """
    Dataset personalizzato per testi clinici.
    Gestisce tokenizzazione e padding on-the-fly.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Inizializza il dataset con testi e etichette.
        
        Args:
            texts: Lista di testi clinici
            labels: Lista di etichette numeriche
            tokenizer: Tokenizer HuggingFace
            max_length: Lunghezza massima dei token
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Dataset inizializzato con {len(texts)} campioni")

    def __len__(self) -> int:
        """Restituisce il numero di campioni nel dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Recupera un campione tokenizzato.
        
        Returns:
            Dict con input_ids, attention_mask e labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenizzazione: trasforma il testo in numeri (ID) che il modello capisce.
        # 'truncation=True' taglia il testo se supera 'max_length'.
        # 'padding='max_length'' aggiunge zeri per rendere tutti i testi della stessa lunghezza.
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'  # 'pt' sta per PyTorch, restituisce tensori
        )
        
        # Restituisce i tensori appiattendoli (flatten) e l'etichetta
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =====================================================
# SEZIONE 5: CLASSE PRINCIPALE DEL CLASSIFICATORE
# =====================================================
class NLPClassifier:
    """
    Classificatore NLP per testi clinici basato su transformer (BERT).
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = None
    ):
        """
        Inizializza il classificatore.
        
        Args:
            model_name: Nome del modello HuggingFace (es. 'bert-base-uncased').
            num_labels: Numero di classi target (es. 2 per positivo/negativo).
            device: 'cuda' per GPU, 'cpu' per processore. Se None, lo sceglie automaticamente.
        """
        # Sceglie la GPU se disponibile, altrimenti la CPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilizzo dispositivo: {self.device}")
        
        # Carica il tokenizer (trasforma testo in token) e il modello pre-addestrato
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        # Sposta il modello sul dispositivo scelto (GPU o CPU)
        self.model.to(self.device)
        logger.info(f"Modello {model_name} caricato con {num_labels} classi")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str] = None,
        val_labels: List[int] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> Dict[str, float]:
        """
        Addestra il modello sui dati forniti.
        
        Args:
            train_texts: Testi di training
            train_labels: Etichette di training
            val_texts: Testi di validazione (opzionale)
            val_labels: Etichette di validazione (opzionale)
            epochs: Numero di epoche (passate su tutto il dataset)
            batch_size: Quanti campioni processare alla volta
            learning_rate: Velocità di apprendimento
            
        Returns:
            Dict con metriche di training
        """
        # Crea il dataset e il DataLoader (che mischia e organizza i dati in batch)
        train_dataset = ClinicalTextDataset(
            train_texts,
            train_labels,
            self.tokenizer
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Mischia i dati per evitare che il modello impari l'ordine
            num_workers=0  # '0' significa che carica i dati nel processo principale (semplice)
        )
        
        # Inizializza l'ottimizzatore AdamW (risolto l'errore qui!)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler: riduce gradualmente il learning rate per affinare il modello
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=epochs * len(train_loader) // 10
        )
        
        # Imposta il modello in modalità "training" (attiva dropout e batch norm)
        self.model.train()
        best_val_accuracy = 0.0
        
        # Ciclo principale di addestramento
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Sposta i dati sul dispositivo (GPU/CPU)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass: il modello calcola le predizioni
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss  # Estrae il valore dell'errore (loss)
                
                # Backward pass: calcola i gradienti (derivate) per aggiornare i pesi
                loss.backward()
                # 'Gradient clipping' impedisce che i gradienti diventino troppo grandi (esplosione)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()  # Aggiorna i pesi del modello
                scheduler.step()  # Aggiorna il learning rate
                optimizer.zero_grad()  # Resetta i gradienti per il prossimo batch
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Validazione (se fornita)
            if val_texts and val_labels:
                val_metrics = self.evaluate(val_texts, val_labels)
                logger.info(f"Validation Metrics: {val_metrics}")
                
                # Salva il modello se ha ottenuto l'accuratezza migliore finora
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    self.save_model("best_model.pt")
        
        return {'train_loss': avg_loss, 'best_val_accuracy': best_val_accuracy}

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Valuta il modello su dati di test (non visti durante il training).
        
        Args:
            texts: Testi da valutare
            labels: Etichette reali
            
        Returns:
            Dict con accuracy, f1 score e report dettagliato
        """
        self.model.eval()  # Modalità "eval" (disabilita dropout per essere deterministico)
        predictions = []
        true_labels = []
        
        eval_dataset = ClinicalTextDataset(texts, labels, self.tokenizer)
        eval_loader = DataLoader(eval_dataset, batch_size=32)
        
        # 'torch.no_grad()' disabilita il calcolo dei gradienti (risparmia memoria e velocizza)
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)  # Prende la classe con punteggio più alto
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].numpy())
        
        # Calcola metriche standard di Machine Learning
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }

    def predict(self, texts: List[str]) -> List[int]:
        """
        Genera predizioni per nuovi testi (es. in produzione).
        
        Args:
            texts: Lista di testi da classificare
            
        Returns:
            Lista di predizioni (0 o 1)
        """
        self.model.eval()
        predictions = []
        
        dummy_labels = [0] * len(texts)
        dataset = ClinicalTextDataset(texts, dummy_labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions

    def save_model(self, path: str) -> None:
        """Salva il modello e il tokenizer su disco per riutilizzarli dopo."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Modello salvato in {path}")

    def load_model(self, path: str) -> None:
        """Carica modello e tokenizer da disco."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Modello caricato da {path}")

# =====================================================
# SEZIONE 6: FUNZIONE MAIN PER TESTARE IL PROGRAMMA
# =====================================================
def main():
    """
    Funzione di test/demo.
    Se esegui questo script, parte una demo con dati fittizi per mostrare che funziona.
    """
    logger.info("=== Demo NLP Classifier (Versione Auto-Installante e Corretta) ===")
    
    # Dati di esempio (simulano note cliniche)
    train_texts = [
        "Paziente presenta febbre alta e tosse secca persistente da 3 giorni",
        "Dopo intervento chirurgico, il paziente mostra segni di infezione",
        "Esami del sangue mostrano valori normali, dimissione prevista domani",
        "Paziente con diabete di tipo 2 non controllato, necessita insulina"
    ]
    train_labels = [1, 1, 0, 1]  # 1 = richiede attenzione, 0 = stabile
    
    test_texts = [
        "Paziente in condizioni stabili, pressione normale",
        "Temperatura elevata 39.5°C, richiesti ulteriori esami"
    ]
    test_labels = [0, 1]
    
    # Inizializza il classificatore
    # Usiamo 'bert-base-uncased' che è piccolo e veloce per la demo
    classifier = NLPClassifier(model_name="bert-base-uncased", num_labels=2)
    
    # Addestra (solo 2 epoche per velocizzare la dimostrazione)
    classifier.train(
        train_texts,
        train_labels,
        epochs=2,
        batch_size=2
    )
    
    # Valuta sulle frasi di test
    metrics = classifier.evaluate(test_texts, test_labels)
    logger.info(f"Metriche sul test set: {metrics}")
    
    # Prova a predire una nuova frase
    new_texts = [
        "Paziente dimesso, follow-up in 2 settimane"
    ]
    predictions = classifier.predict(new_texts)
    logger.info(f"Predizione per '{new_texts[0]}' -> Classe: {predictions[0]} (1=attenzione, 0=stabile)")

    print("\n✅ Programma completato con successo! Tutto funziona.")

if __name__ == "__main__":
    main()