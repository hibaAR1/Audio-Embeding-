"""
ADAPTIVE AUDIO FUSION
UrbanSound8K
Extraction des features + évaluation des pipelines de fusion

Configuration utilisée :
- PANNs CNN14 : embedding de 2048 dimensions
- AST : embedding de 768 dimensions
- Projection PANNs : 2048 -> 128
- Projection AST : 768 -> 128
- Fusion : 128 + 128 = 256
- UrbanSound8K : 10 classes, 10 folds

Installation :
pip install panns_inference transformers soundfile librosa numpy scikit-learn torch pandas

Extraction :
python urban_2048_fusion.py --extract \
    --urbansound_path "C:\\Users\\21268\\Downloads\\UrbanSound8K" \
    --features_dir "./urban_features_2048"

Évaluation :
python urban_2048_fusion.py \
    --features_dir "./urban_features_2048"
"""

import os
import csv
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # PANNs fonctionne à 32 kHz
    SR = 32000

    # AST fonctionne à 16 kHz
    AST_SR = 16000

    # UrbanSound8K contient des fichiers de maximum 4 secondes
    N_SAMPLES = SR * 4

    # Dimensions réellement utilisées
    PANNS_DIM = 2048
    PANNS_SCORES_DIM = 527
    SSL_DIM = 768

    # Dimensions après projection
    PROJ_DIM = 128
    HIDDEN_DIM = 256

    # UrbanSound8K
    N_CLASSES = 10
    N_FOLDS = 10

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CLASS_NAMES = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]


# ============================================================
# UTILITAIRES
# ============================================================

def set_seed(seed=42):
    """
    Fixe les graines aléatoires pour rendre les expériences
    plus reproductibles.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_audio(audio_path):
    """
    Charge un fichier audio, le convertit en mono,
    le rééchantillonne à 32 kHz et le complète à 4 secondes.
    """

    import librosa

    audio, sr = librosa.load(
        audio_path,
        sr=Config.SR,
        mono=True
    )

    target_length = Config.N_SAMPLES

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    audio = audio.astype(np.float32)

    return audio


# ============================================================
# EXTRACTION DES FEATURES
# ============================================================

def extract_features(urbansound_path, output_dir):
    """
    Extrait :

    PANNs :
        scores AudioSet : 527 dimensions
        embedding CNN14 : 2048 dimensions

    AST :
        embedding : 768 dimensions
    """

    from panns_inference import AudioTagging
    import librosa
    from transformers import ASTFeatureExtractor, ASTModel

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        urbansound_path,
        "metadata",
        "UrbanSound8K.csv"
    )

    audio_dir = os.path.join(
        urbansound_path,
        "audio"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {csv_path}"
        )

    # --------------------------------------------------------
    # Lecture du fichier metadata
    # --------------------------------------------------------

    samples = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            samples.append(
                {
                    "filename": row["slice_file_name"],
                    "fold": int(row["fold"]),
                    "class_id": int(row["classID"]),
                    "class_name": row["class"],
                }
            )

    print(f"Nombre total d'échantillons : {len(samples)}")
    print(
        f"Nombre de classes : "
        f"{len(set(sample['class_name'] for sample in samples))}"
    )
    print(f"Device utilisé : {Config.DEVICE}")

    # --------------------------------------------------------
    # Chargement de PANNs
    # --------------------------------------------------------

    print("\nChargement de PANNs CNN14...")

    panns_model = AudioTagging(
        checkpoint_path=None,
        device=str(Config.DEVICE)
    )

    panns_model.model.eval()

    # --------------------------------------------------------
    # Chargement de AST
    # --------------------------------------------------------

    print("Chargement de AST...")

    ast_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    ast_model = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    ).to(Config.DEVICE)

    ast_model.eval()

    # --------------------------------------------------------
    # Listes de sauvegarde
    # --------------------------------------------------------

    panns_features = []
    ast_features = []
    labels = []
    folds = []

    # --------------------------------------------------------
    # Extraction fichier par fichier
    # --------------------------------------------------------

    for i, sample in enumerate(samples):

        audio_path = os.path.join(
            audio_dir,
            f"fold{sample['fold']}",
            sample["filename"]
        )

        if not os.path.exists(audio_path):
            print(f"[SKIP] Fichier introuvable : {audio_path}")
            continue

        try:
            # ------------------------------------------------
            # Préparation audio à 32 kHz pour PANNs
            # ------------------------------------------------

            audio_32k = load_and_prepare_audio(audio_path)

            audio_tensor = torch.FloatTensor(
                audio_32k
            ).unsqueeze(0).to(Config.DEVICE)

            # ------------------------------------------------
            # PANNs
            # ------------------------------------------------
            #
            # Ordre des sorties PANNs :
            # 1. panns_scores     : 527 dimensions
            # 2. panns_embedding : 2048 dimensions
            #

            with torch.no_grad():
                panns_scores, panns_embedding = (
                    panns_model.model(audio_tensor)
                )

            # Vérification des dimensions PANNs
            if i == 0:
                print(
                    "\nVérification des sorties PANNs :"
                )
                print(
                    f"Scores PANNs : "
                    f"{tuple(panns_scores.shape)}"
                )
                print(
                    f"Embedding PANNs : "
                    f"{tuple(panns_embedding.shape)}"
                )

            if panns_scores.shape[-1] != Config.PANNS_SCORES_DIM:
                raise ValueError(
                    "Dimension incorrecte pour les scores PANNs : "
                    f"{panns_scores.shape[-1]} au lieu de "
                    f"{Config.PANNS_SCORES_DIM}"
                )

            if panns_embedding.shape[-1] != Config.PANNS_DIM:
                raise ValueError(
                    "Dimension incorrecte pour l'embedding PANNs : "
                    f"{panns_embedding.shape[-1]} au lieu de "
                    f"{Config.PANNS_DIM}"
                )

            # On sauvegarde l'embedding 2048,
            # pas les scores 527
            panns_features.append(
                panns_embedding[0].detach().cpu().numpy()
            )

            # ------------------------------------------------
            # Préparation audio à 16 kHz pour AST
            # ------------------------------------------------

            audio_16k = librosa.resample(
                audio_32k,
                orig_sr=Config.SR,
                target_sr=Config.AST_SR
            )

            ast_inputs = ast_extractor(
                audio_16k,
                sampling_rate=Config.AST_SR,
                return_tensors="pt"
            )

            ast_inputs = {
                key: value.to(Config.DEVICE)
                for key, value in ast_inputs.items()
            }

            # ------------------------------------------------
            # AST
            # ------------------------------------------------

            with torch.no_grad():
                ast_output = ast_model(**ast_inputs)

                ast_embedding = (
                    ast_output.last_hidden_state.mean(dim=1)
                )

            if i == 0:
                print(
                    f"Embedding AST : "
                    f"{tuple(ast_embedding.shape)}"
                )

            if ast_embedding.shape[-1] != Config.SSL_DIM:
                raise ValueError(
                    "Dimension incorrecte pour AST : "
                    f"{ast_embedding.shape[-1]} au lieu de "
                    f"{Config.SSL_DIM}"
                )

            ast_features.append(
                ast_embedding[0].detach().cpu().numpy()
            )

            labels.append(sample["class_id"])
            folds.append(sample["fold"])

            if (i + 1) % 100 == 0:
                print(
                    f"{i + 1}/{len(samples)} fichiers extraits..."
                )

        except Exception as error:
            print(
                f"[ERREUR] {audio_path}\n"
                f"          {error}"
            )
            continue

    # --------------------------------------------------------
    # Conversion en tableaux NumPy
    # --------------------------------------------------------

    panns_arr = np.asarray(
        panns_features,
        dtype=np.float32
    )

    ast_arr = np.asarray(
        ast_features,
        dtype=np.float32
    )

    labels_arr = np.asarray(
        labels,
        dtype=np.int64
    )

    folds_arr = np.asarray(
        folds,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Vérifications finales
    # --------------------------------------------------------

    if panns_arr.ndim != 2:
        raise ValueError(
            f"PANNs doit être un tableau 2D, reçu : "
            f"{panns_arr.shape}"
        )

    if ast_arr.ndim != 2:
        raise ValueError(
            f"AST doit être un tableau 2D, reçu : "
            f"{ast_arr.shape}"
        )

    if panns_arr.shape[1] != Config.PANNS_DIM:
        raise ValueError(
            f"PANNs final : {panns_arr.shape[1]} dimensions, "
            f"attendu : {Config.PANNS_DIM}"
        )

    if ast_arr.shape[1] != Config.SSL_DIM:
        raise ValueError(
            f"AST final : {ast_arr.shape[1]} dimensions, "
            f"attendu : {Config.SSL_DIM}"
        )

    if not (
        len(panns_arr)
        == len(ast_arr)
        == len(labels_arr)
        == len(folds_arr)
    ):
        raise ValueError(
            "Le nombre d'échantillons ne correspond pas entre "
            "PANNs, AST, labels et folds."
        )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    np.save(
        os.path.join(output_dir, "panns.npy"),
        panns_arr
    )

    np.save(
        os.path.join(output_dir, "ssl.npy"),
        ast_arr
    )

    np.save(
        os.path.join(output_dir, "labels.npy"),
        labels_arr
    )

    np.save(
        os.path.join(output_dir, "folds.npy"),
        folds_arr
    )

    print("\nFeatures sauvegardées avec succès.")
    print(f"Dossier : {output_dir}")
    print(f"PANNs : {panns_arr.shape}")
    print(f"AST   : {ast_arr.shape}")
    print(f"Labels: {labels_arr.shape}")
    print(f"Folds : {folds_arr.shape}")

    print("\nConfiguration confirmée :")
    print("PANNs utilisé : 2048 dimensions")
    print("AST utilisé   : 768 dimensions")


# ============================================================
# CHARGEMENT DES FEATURES
# ============================================================

def load_features(features_dir):
    """
    Charge les features sauvegardées et vérifie leurs dimensions.
    """

    panns_path = os.path.join(
        features_dir,
        "panns.npy"
    )

    ast_path = os.path.join(
        features_dir,
        "ssl.npy"
    )

    labels_path = os.path.join(
        features_dir,
        "labels.npy"
    )

    folds_path = os.path.join(
        features_dir,
        "folds.npy"
    )

    required_files = [
        panns_path,
        ast_path,
        labels_path,
        folds_path,
    ]

    if not all(os.path.exists(path) for path in required_files):
        print(
            "ERREUR : features introuvables. "
            "Lance d'abord l'extraction."
        )
        return None

    panns = np.load(panns_path)
    ast = np.load(ast_path)
    labels = np.load(labels_path)
    folds = np.load(folds_path)

    print("\nVérification des fichiers chargés :")
    print(f"PANNs : {panns.shape}")
    print(f"AST   : {ast.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Folds : {folds.shape}")

    if panns.ndim != 2:
        raise ValueError(
            f"PANNs doit être 2D, reçu : {panns.shape}"
        )

    if ast.ndim != 2:
        raise ValueError(
            f"AST doit être 2D, reçu : {ast.shape}"
        )

    if panns.shape[1] != Config.PANNS_DIM:
        raise ValueError(
            f"Le fichier PANNs contient {panns.shape[1]} dimensions, "
            f"mais le modèle attend {Config.PANNS_DIM}."
        )

    if ast.shape[1] != Config.SSL_DIM:
        raise ValueError(
            f"Le fichier AST contient {ast.shape[1]} dimensions, "
            f"mais le modèle attend {Config.SSL_DIM}."
        )

    if not (
        len(panns)
        == len(ast)
        == len(labels)
        == len(folds)
    ):
        raise ValueError(
            "Les fichiers n'ont pas le même nombre d'échantillons."
        )

    return panns, ast, labels, folds


# ============================================================
# PIPELINE END-TO-END
# ============================================================

class EndToEndPipeline(nn.Module):

    def __init__(
        self,
        proj_dim=Config.PROJ_DIM,
        hidden_dim=Config.HIDDEN_DIM
    ):
        super().__init__()

        self.bn_panns = nn.BatchNorm1d(
            Config.PANNS_DIM
        )

        self.bn_ast = nn.BatchNorm1d(
            Config.SSL_DIM
        )

        self.proj_panns = nn.Linear(
            Config.PANNS_DIM,
            proj_dim
        )

        self.proj_ast = nn.Linear(
            Config.SSL_DIM,
            proj_dim
        )

        fusion_dim = proj_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(
            hidden_dim // 2,
            Config.N_CLASSES
        )

    def forward_with_features(self, panns, ast):
        panns_projected = F.relu(
            self.proj_panns(
                self.bn_panns(panns)
            )
        )

        ast_projected = F.relu(
            self.proj_ast(
                self.bn_ast(ast)
            )
        )

        fused = torch.cat(
            [
                panns_projected,
                ast_projected,
            ],
            dim=1
        )

        features = self.mlp(fused)
        logits = self.classifier(features)

        return logits, features

    def forward(self, panns, ast):
        logits, _ = self.forward_with_features(
            panns,
            ast
        )
        return logits

    def fisher_loss(self, features, labels):
        n_classes = Config.N_CLASSES
        feature_dim = features.shape[1]

        global_mean = features.mean(dim=0)

        class_means = torch.zeros(
            n_classes,
            feature_dim,
            device=features.device
        )

        class_counts = torch.zeros(
            n_classes,
            device=features.device
        )

        for class_id in range(n_classes):

            mask = labels == class_id

            if mask.sum() > 0:
                class_means[class_id] = (
                    features[mask].mean(dim=0)
                )

                class_counts[class_id] = (
                    mask.float().sum()
                )

        active_classes = class_counts > 0

        inter_var = (
            (
                class_means[active_classes]
                - global_mean.unsqueeze(0)
            ) ** 2
        ).sum()

        intra_var = torch.tensor(
            0.0,
            device=features.device
        )

        for class_id in range(n_classes):

            mask = labels == class_id

            if mask.sum() > 0:
                difference = (
                    features[mask]
                    - class_means[class_id].unsqueeze(0)
                )

                intra_var = intra_var + (
                    difference ** 2
                ).sum()

        inter_var = torch.clamp(
            inter_var,
            min=1e-8
        )

        intra_var = torch.clamp(
            intra_var,
            min=1e-8
        )

        return intra_var / inter_var

    def train_pipeline(
        self,
        X_panns,
        X_ast,
        labels,
        epochs=120,
        learning_rate=5e-4
    ):
        X_panns_tensor = torch.FloatTensor(
            X_panns
        ).to(Config.DEVICE)

        X_ast_tensor = torch.FloatTensor(
            X_ast
        ).to(Config.DEVICE)

        labels_tensor = torch.LongTensor(
            labels
        ).to(Config.DEVICE)

        self.to(Config.DEVICE)
        self.train()

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=30,
            T_mult=2
        )

        cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

        for epoch in range(epochs):

            optimizer.zero_grad()

            logits, features = self.forward_with_features(
                X_panns_tensor,
                X_ast_tensor
            )

            classification_loss = cross_entropy(
                logits,
                labels_tensor
            )

            fisher_weight = min(
                0.5,
                0.01 + epoch * 0.005
            )

            fisher_loss_value = self.fisher_loss(
                features,
                labels_tensor
            )

            loss = (
                classification_loss
                + fisher_weight * fisher_loss_value
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 30 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss={loss.item():.4f} | "
                    f"CE={classification_loss.item():.4f} | "
                    f"Fisher={fisher_loss_value.item():.4f}"
                )

    def predict(self, X_panns, X_ast):

        self.eval()

        X_panns_tensor = torch.FloatTensor(
            X_panns
        ).to(Config.DEVICE)

        X_ast_tensor = torch.FloatTensor(
            X_ast
        ).to(Config.DEVICE)

        with torch.no_grad():
            logits = self(
                X_panns_tensor,
                X_ast_tensor
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

        return predictions.cpu().numpy()


# ============================================================
# PIPELINE SÉPARÉ AVEC LDA ET SVM
# ============================================================

class SeparatedPipeline:

    def __init__(
        self,
        proj_dim=Config.PROJ_DIM,
        hidden_dim=Config.HIDDEN_DIM
    ):
        self.fusion = self._build_fusion(
            proj_dim,
            hidden_dim
        )

        self.scaler = StandardScaler()

        self.lda = LinearDiscriminantAnalysis(
            n_components=Config.N_CLASSES - 1
        )

    def _build_fusion(self, proj_dim, hidden_dim):

        class FusionNetwork(nn.Module):

            def __init__(self):
                super().__init__()

                self.bn_panns = nn.BatchNorm1d(
                    Config.PANNS_DIM
                )

                self.bn_ast = nn.BatchNorm1d(
                    Config.SSL_DIM
                )

                self.proj_panns = nn.Linear(
                    Config.PANNS_DIM,
                    proj_dim
                )

                self.proj_ast = nn.Linear(
                    Config.SSL_DIM,
                    proj_dim
                )

                self.head = nn.Sequential(
                    nn.Linear(
                        proj_dim * 2,
                        hidden_dim
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(
                        hidden_dim,
                        Config.N_CLASSES
                    )
                )

            def forward(self, panns, ast):

                panns_projected = F.relu(
                    self.proj_panns(
                        self.bn_panns(panns)
                    )
                )

                ast_projected = F.relu(
                    self.proj_ast(
                        self.bn_ast(ast)
                    )
                )

                fused = torch.cat(
                    [
                        panns_projected,
                        ast_projected,
                    ],
                    dim=1
                )

                return self.head(fused)

            def extract(self, panns, ast):

                panns_projected = F.relu(
                    self.proj_panns(
                        self.bn_panns(panns)
                    )
                )

                ast_projected = F.relu(
                    self.proj_ast(
                        self.bn_ast(ast)
                    )
                )

                return torch.cat(
                    [
                        panns_projected,
                        ast_projected,
                    ],
                    dim=1
                )

        return FusionNetwork()

    def train_pipeline(
        self,
        X_panns,
        X_ast,
        labels,
        epochs=100
    ):
        print("\n[1] Entraînement de la fusion...")

        X_panns_tensor = torch.FloatTensor(
            X_panns
        ).to(Config.DEVICE)

        X_ast_tensor = torch.FloatTensor(
            X_ast
        ).to(Config.DEVICE)

        labels_tensor = torch.LongTensor(
            labels
        ).to(Config.DEVICE)

        self.fusion.to(Config.DEVICE)
        self.fusion.train()

        optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=5e-4,
            weight_decay=1e-3
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

        cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

        for epoch in range(epochs):

            optimizer.zero_grad()

            logits = self.fusion(
                X_panns_tensor,
                X_ast_tensor
            )

            loss = cross_entropy(
                logits,
                labels_tensor
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss={loss.item():.4f}"
                )

        self.fusion.eval()

        with torch.no_grad():
            fused_features = self.fusion.extract(
                X_panns_tensor,
                X_ast_tensor
            ).cpu().numpy()

        print("[2] Normalisation avant LDA...")

        self.scaler.fit(fused_features)

        scaled_features = self.scaler.transform(
            fused_features
        )

        print("[3] Entraînement LDA...")

        self.lda.fit(
            scaled_features,
            labels
        )

    def extract_features(self, X_panns, X_ast):

        self.fusion.eval()

        X_panns_tensor = torch.FloatTensor(
            X_panns
        ).to(Config.DEVICE)

        X_ast_tensor = torch.FloatTensor(
            X_ast
        ).to(Config.DEVICE)

        with torch.no_grad():
            fused_features = self.fusion.extract(
                X_panns_tensor,
                X_ast_tensor
            ).cpu().numpy()

        return self.scaler.transform(
            fused_features
        )


# ============================================================
# BASELINE KNN
# ============================================================

class BaselineKNN:

    def __init__(
        self,
        n_components=128,
        n_neighbors=5
    ):
        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=n_components
        )

        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance"
        )

    def fit(self, X_panns, X_ast, labels):

        X = np.concatenate(
            [
                X_panns,
                X_ast,
            ],
            axis=1
        )

        X = self.scaler.fit_transform(X)
        X = self.pca.fit_transform(X)

        self.knn.fit(X, labels)

    def predict(self, X_panns, X_ast):

        X = np.concatenate(
            [
                X_panns,
                X_ast,
            ],
            axis=1
        )

        X = self.scaler.transform(X)
        X = self.pca.transform(X)

        return self.knn.predict(X)


# ============================================================
# BASELINE SVM
# ============================================================

class BaselineSVM:

    def __init__(self):

        self.scaler = StandardScaler()

        self.svm = SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )

    def fit(self, X_panns, X_ast, labels):

        X = np.concatenate(
            [
                X_panns,
                X_ast,
            ],
            axis=1
        )

        X = self.scaler.fit_transform(X)

        self.svm.fit(X, labels)

    def predict(self, X_panns, X_ast):

        X = np.concatenate(
            [
                X_panns,
                X_ast,
            ],
            axis=1
        )

        X = self.scaler.transform(X)

        return self.svm.predict(X)


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate_all(panns, ast, labels, folds):

    results = {
        "baseline_knn": [],
        "baseline_svm": [],
        "end_to_end": [],
        "separated_svm": [],
    }

    for fold in range(1, Config.N_FOLDS + 1):

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold}/{Config.N_FOLDS}"
        )
        print("=" * 70)

        train_mask = folds != fold
        test_mask = folds == fold

        X_panns_train = panns[train_mask]
        X_ast_train = ast[train_mask]
        y_train = labels[train_mask]

        X_panns_test = panns[test_mask]
        X_ast_test = ast[test_mask]
        y_test = labels[test_mask]

        print(
            f"Train : {len(y_train)}"
        )
        print(
            f"Test  : {len(y_test)}"
        )

        # ----------------------------------------------------
        # Baseline kNN
        # ----------------------------------------------------

        print("\n[A] BASELINE KNN")

        knn = BaselineKNN(
            n_components=128,
            n_neighbors=5
        )

        knn.fit(
            X_panns_train,
            X_ast_train,
            y_train
        )

        predictions = knn.predict(
            X_panns_test,
            X_ast_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f">>> BASELINE KNN : "
            f"{accuracy * 100:.2f}%"
        )

        results["baseline_knn"].append(
            accuracy
        )

        # ----------------------------------------------------
        # Baseline SVM
        # ----------------------------------------------------

        print("\n[B] BASELINE SVM")

        svm_baseline = BaselineSVM()

        svm_baseline.fit(
            X_panns_train,
            X_ast_train,
            y_train
        )

        predictions = svm_baseline.predict(
            X_panns_test,
            X_ast_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f">>> BASELINE SVM : "
            f"{accuracy * 100:.2f}%"
        )

        results["baseline_svm"].append(
            accuracy
        )

        # ----------------------------------------------------
        # End-to-end
        # ----------------------------------------------------

        print("\n[C] END-TO-END")

        end_to_end = EndToEndPipeline()

        end_to_end.train_pipeline(
            X_panns_train,
            X_ast_train,
            y_train,
            epochs=120,
            learning_rate=5e-4
        )

        predictions = end_to_end.predict(
            X_panns_test,
            X_ast_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f">>> END-TO-END : "
            f"{accuracy * 100:.2f}%"
        )

        results["end_to_end"].append(
            accuracy
        )

        # ----------------------------------------------------
        # Séparé + LDA + SVM
        # ----------------------------------------------------

        print("\n[D] SÉPARÉ + LDA + SVM")

        separated = SeparatedPipeline()

        separated.train_pipeline(
            X_panns_train,
            X_ast_train,
            y_train,
            epochs=100
        )

        X_lda_train = separated.extract_features(
            X_panns_train,
            X_ast_train
        )

        X_lda_test = separated.extract_features(
            X_panns_test,
            X_ast_test
        )

        svm = SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )

        svm.fit(
            X_lda_train,
            y_train
        )

        predictions = svm.predict(
            X_lda_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f">>> SÉPARÉ + SVM : "
            f"{accuracy * 100:.2f}%"
        )

        results["separated_svm"].append(
            accuracy
        )

    return results


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def print_final_results(results):

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX 10-FOLD")
    print("=" * 70)

    for name, accuracies in results.items():

        mean_accuracy = np.mean(accuracies) * 100
        std_accuracy = np.std(accuracies) * 100

        print(
            f"{name:<20s} : "
            f"{mean_accuracy:.2f}% "
            f"+/- {std_accuracy:.2f}%"
        )

    print("\nDétail par fold :")

    n_folds = len(
        next(iter(results.values()))
    )

    for fold_index in range(n_folds):

        values = []

        for name in results:

            value = (
                results[name][fold_index] * 100
            )

            values.append(
                f"{name}={value:.2f}%"
            )

        print(
            f"Fold {fold_index + 1} : "
            + " | ".join(values)
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urbansound_path",
        type=str,
        default=r"C:\Users\21268\Downloads\UrbanSound8K",
        help="Chemin vers UrbanSound8K"
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default="./urban_features_2048",
        help="Dossier des features"
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extraire les features"
    )

    args = parser.parse_args()

    set_seed(42)

    print("=" * 70)
    print("ADAPTIVE AUDIO FUSION")
    print("UrbanSound8K")
    print("=" * 70)

    print(f"Device : {Config.DEVICE}")
    print(f"PANNs  : {Config.PANNS_DIM} dimensions")
    print(f"AST    : {Config.SSL_DIM} dimensions")
    print(f"Projet : {Config.PROJ_DIM} dimensions")

    if args.extract:

        print("\nEXTRACTION DES FEATURES")
        print("Cette étape peut prendre du temps.")

        extract_features(
            args.urbansound_path,
            args.features_dir
        )

        return

    data = load_features(
        args.features_dir
    )

    if data is None:
        print(
            "\nLance d'abord l'extraction avec "
            "--extract."
        )
        return

    panns, ast, labels, folds = data

    print("\nDimensions confirmées :")
    print(f"PANNs : {panns.shape}")
    print(f"AST   : {ast.shape}")
    print(
        f"Classes : {len(np.unique(labels))}"
    )
    print(
        f"Folds : {len(np.unique(folds))}"
    )

    results = evaluate_all(
        panns,
        ast,
        labels,
        folds
    )

    print_final_results(
        results
    )


if __name__ == "__main__":
    main()
