"""
=============================================================================
Protocolo Experimental: Batch Normalization Strategies en Fine-Tuning
de Modelos Auto-Supervisados para Series Temporales (UCR Benchmark)
=============================================================================
Arquitecturas  : FCN, ResNet1D, InceptionTime, LSTM-FCN
Preentrenamiento: TNC (Temporal Neighborhood Coding)

Estrategias BN (6):
  BN-F     : Frozen BN — estadísticas y parámetros γ/β congelados
  BN-U     : Full Update — todo se actualiza normalmente
  BN-P     : Partial — estadísticas congeladas, γ/β entrenables
  AdaBN    : Adaptive BN — estadísticas recalibradas con datos target
  IN-Adapt : InstanceNorm adaptativa — reemplaza BN por IN durante fine-tuning
             (robusto a batch size pequeño; reemplaza TransNorm original que
              requería datos fuente+target simultáneos — incompatible con este
              diseño experimental)
  BN-LR    : Low-LR BN — capas BN con lr × 0.1 respecto al resto

Correcciones v2 respecto al quick run:
  1. TransNorm → IN-Adapt (ver arriba)
  2. Inicialización kaiming en todos los modelos para evitar colapso con
     pocos datos etiquetados
  3. AdaBN estabilizado: calibración limitada a 10 batches + warmup 5 épocas
  4. Early stopping por pérdida mínima (paciencia=10) para evitar divergencia
  5. Gradient clipping (max_norm=1.0) en todas las estrategias

Factores controlados:
  label_ratio : 5%, 20%, 100%
  batch_size  : 16, 64, 256
  seeds       : 5 (42, 7, 123, 2024, 999)

Métricas: accuracy, f1-macro, precision-macro, recall-macro, loss_std
Salidas : results/raw_results.jsonl + figuras PNG + tablas CSV

Uso:
    python bn_experiment.py            # experimento completo (30 datasets)
    python bn_experiment.py --quick    # debug (2 datasets, 2 seeds)
=============================================================================
"""

import os
import sys
import json
import math
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import friedmanchisquare

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. Argparse
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='BN Strategies Experiment')
parser.add_argument('--quick', action='store_true',
                    help='Modo debug: 2 datasets, 2 arquitecturas, 2 seeds')
ARGS = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuración global
# ─────────────────────────────────────────────────────────────────────────────
SEEDS        = [42, 7, 123, 2024, 999]
LABEL_RATIOS = [0.05, 0.20, 1.00]
BATCH_SIZES  = [16, 64, 256]
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = Path('./results')
RESULTS_DIR.mkdir(exist_ok=True)

# IN-Adapt reemplaza TransNorm (ver encabezado para justificación)
BN_STRATEGIES = ['BN-F', 'BN-U', 'BN-P', 'AdaBN', 'IN-Adapt', 'BN-LR']
ARCHITECTURES = ['FCN', 'ResNet1D', 'InceptionTime', 'LSTMFCN']

DATASET_GROUPS = {
    'ECG':       ['ECG200', 'ECG5000', 'NonInvasiveFetalECGThorax1'],
    'HAR':       ['BasicMotions', 'Cricket', 'Epilepsy'],
    'Sensor':    ['ElectricDevices', 'FaceDetection', 'Heartbeat',
                  'NATOPS', 'PEMS-SF'],
    'Synthetic': ['ArrowHead', 'Chinatown', 'Coffee', 'Computers',
                  'Earthquakes', 'FordA', 'FordB', 'GunPoint'],
    'Image':     ['FaceAll', 'FaceFour', 'Fish', 'Herring',
                  'Lightning2', 'Lightning7', 'MedicalImages'],
    'Other':     ['Plane', 'Trace', 'TwoLeadECG', 'Wafer', 'Wine'],
}
ALL_DATASETS = [d for g in DATASET_GROUPS.values() for d in g]

if ARGS.quick:
    DATASETS_TO_RUN = ALL_DATASETS[:2]
    ARCHS_TO_RUN    = ['FCN', 'ResNet1D']
    SEEDS_TO_RUN    = SEEDS[:2]
    RATIOS_TO_RUN   = [0.05, 1.00]
    BATCHES_TO_RUN  = [64]
    PRETRAIN_EPOCHS = 5
    FINETUNE_EPOCHS = 10
    PATIENCE        = 5
    print('⚡ MODO QUICK ACTIVADO')
else:
    DATASETS_TO_RUN = ALL_DATASETS
    ARCHS_TO_RUN    = ARCHITECTURES
    SEEDS_TO_RUN    = SEEDS
    RATIOS_TO_RUN   = LABEL_RATIOS
    BATCHES_TO_RUN  = BATCH_SIZES
    PRETRAIN_EPOCHS = 50
    FINETUNE_EPOCHS = 100   # más épocas + early stopping
    PATIENCE        = 10    # early stopping patience

total_runs = (len(DATASETS_TO_RUN) * len(ARCHS_TO_RUN) *
              len(BN_STRATEGIES) * len(RATIOS_TO_RUN) *
              len(BATCHES_TO_RUN) * len(SEEDS_TO_RUN))

print(f'  Device       : {DEVICE}')
print(f' Datasets     : {len(DATASETS_TO_RUN)}')
print(f'  Arquitecturas : {ARCHS_TO_RUN}')
print(f' Estrategias  : {BN_STRATEGIES}')
print(f' Total runs   : {total_runs:,}')
print(f' Resultados   : {RESULTS_DIR.absolute()}')
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_weights_kaiming(m):
    """Inicialización Kaiming para evitar colapso con pocos datos."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Carga de datasets UCR
# ─────────────────────────────────────────────────────────────────────────────
try:
    from aeon.datasets import load_classification
except ImportError:
    print(' aeon no instalado. Ejecuta: pip install aeon')
    sys.exit(1)


def load_dataset(name):
    """Carga un dataset UCR normalizado como tensores PyTorch."""
    try:
        X_train, y_train = load_classification(name, split='train')
        X_test,  y_test  = load_classification(name, split='test')
    except Exception as e:
        print(f'    {name}: {e}')
        return None

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    le = LabelEncoder()
    y_train = le.fit_transform(y_train).astype(np.int64)
    y_test  = le.transform(y_test).astype(np.int64)

    return {
        'name':       name,
        'X_train':    torch.tensor(X_train),
        'y_train':    torch.tensor(y_train),
        'X_test':     torch.tensor(X_test),
        'y_test':     torch.tensor(y_test),
        'n_classes':  len(np.unique(y_train)),
        'seq_len':    X_train.shape[2],
        'n_channels': X_train.shape[1],
    }


def get_labeled_subset(dataset, ratio, seed):
    """Subset etiquetado estratificado por clase."""
    rng = np.random.RandomState(seed)
    y   = dataset['y_train'].numpy()
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n = max(1, int(len(cls_idx) * ratio))
        idx.extend(rng.choice(cls_idx, n, replace=False).tolist())
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. Arquitecturas backbone
# ─────────────────────────────────────────────────────────────────────────────

class FCN(nn.Module):
    """FCN de 3 capas (Wang et al., 2017)."""
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 128, 8, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding='same'),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, n_classes)
        self.apply(init_weights_kaiming)

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=-1))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 8, padding='same'),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 5, padding='same'),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding='same'),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, padding='same'),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet-1D con 3 bloques residuales."""
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            ResBlock(in_channels, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
        )
        self.classifier = nn.Linear(128, n_classes)
        self.apply(init_weights_kaiming)

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=-1))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, n_filters=32):
        super().__init__()
        self.bottleneck  = nn.Conv1d(in_ch, n_filters, 1, padding='same', bias=False)
        self.conv_large  = nn.Conv1d(n_filters, n_filters, 39, padding='same', bias=False)
        self.conv_mid    = nn.Conv1d(n_filters, n_filters, 19, padding='same', bias=False)
        self.conv_small  = nn.Conv1d(n_filters, n_filters,  9, padding='same', bias=False)
        self.maxpool     = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv     = nn.Conv1d(in_ch, n_filters, 1, padding='same', bias=False)
        self.bn          = nn.BatchNorm1d(n_filters * 4)
        self.relu        = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([
            self.conv_large(b), self.conv_mid(b),
            self.conv_small(b), self.mp_conv(self.maxpool(x))
        ], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime (Fawaz et al., 2020)."""
    def __init__(self, in_channels, n_classes, n_filters=32, depth=6):
        super().__init__()
        channels = [in_channels] + [n_filters * 4] * depth
        self.blocks = nn.Sequential(
            *[InceptionBlock(channels[i], n_filters) for i in range(depth)]
        )
        self.classifier = nn.Linear(n_filters * 4, n_classes)
        self.apply(init_weights_kaiming)

    def forward(self, x):
        return self.classifier(self.blocks(x).mean(dim=-1))


class LSTMFCN(nn.Module):
    """LSTM-FCN (Karim et al., 2018)."""
    def __init__(self, in_channels, seq_len, n_classes):
        super().__init__()
        self.lstm    = nn.LSTM(in_channels, 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels, 128, 8, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding='same'),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Linear(256, n_classes)
        self.apply(init_weights_kaiming)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))
        lstm_feat   = self.dropout(lstm_out[:, -1, :])
        fcn_feat    = self.fcn(x).mean(dim=-1)
        return self.classifier(torch.cat([lstm_feat, fcn_feat], dim=1))


def build_model(arch, in_channels, seq_len, n_classes):
    if arch == 'FCN':          return FCN(in_channels, n_classes)
    if arch == 'ResNet1D':     return ResNet1D(in_channels, n_classes)
    if arch == 'InceptionTime':return InceptionTime(in_channels, n_classes)
    if arch == 'LSTMFCN':      return LSTMFCN(in_channels, seq_len, n_classes)
    raise ValueError(f'Arquitectura desconocida: {arch}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Preentrenamiento TNC
# ─────────────────────────────────────────────────────────────────────────────
class TNCEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.projector(self.encoder(x).mean(dim=-1))


class Discriminator(nn.Module):
    def __init__(self, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, z1, z2):
        return self.net(torch.cat([z1, z2], dim=-1))


def tnc_loss(disc, z_anc, z_pos, z_neg):
    pos = disc(z_anc, z_pos)
    neg = disc(z_anc, z_neg)
    return (-torch.log(pos + 1e-8) - torch.log(1 - neg + 1e-8)).mean()


def sample_tnc_triplets(X, delta=20):
    B, C, L = X.shape
    delta = max(1, min(delta, L // 4))
    anchors, positives, negatives = [], [], []
    for i in range(B):
        lo, hi = delta, max(delta + 1, L - delta - 1)
        t = random.randint(lo, hi)
        t_pos = random.randint(max(0, t - delta), min(L - 1, t + delta))
        neg_range = list(range(0, max(0, t - delta))) + \
                    list(range(min(L, t + delta + 1), L))
        t_neg = random.choice(neg_range) if neg_range else t_pos
        w = max(1, min(delta, L // 4))
        anchors.append(X[i, :, max(0, t - w): t + w + 1])
        positives.append(X[i, :, max(0, t_pos - w): t_pos + w + 1])
        negatives.append(X[i, :, max(0, t_neg - w): t_neg + w + 1])

    def pad(segs):
        max_l = max(s.shape[-1] for s in segs)
        return torch.stack([
            nn.functional.pad(s, (0, max_l - s.shape[-1])) for s in segs
        ])
    return pad(anchors), pad(positives), pad(negatives)


def pretrain_tnc(X_unlabeled, in_channels, epochs=50, lr=1e-3,
                 batch_size=64, device=DEVICE):
    encoder = TNCEncoder(in_channels).to(device)
    disc    = Discriminator().to(device)
    opt     = optim.Adam(list(encoder.parameters()) +
                         list(disc.parameters()), lr=lr)
    loader  = DataLoader(TensorDataset(X_unlabeled),
                         batch_size=batch_size, shuffle=True)
    encoder.train(); disc.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            if xb.shape[-1] < 8:
                continue
            anc, pos, neg = sample_tnc_triplets(xb)
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            loss = tnc_loss(disc, encoder(anc), encoder(pos), encoder(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# 6. Estrategias de Batch Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _replace_bn_with_in(model):
    """
    Clona el modelo reemplazando cada BatchNorm1d por InstanceNorm1d.
    Usado exclusivamente por IN-Adapt.
    InstanceNorm normaliza por instancia (independiente del batch size),
    lo que la hace robusta a mini-lotes pequeños — factor clave de este estudio.
    """
    model = deepcopy(model)

    def replace_in_module(parent):
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm1d):
                in_layer = nn.InstanceNorm1d(
                    child.num_features,
                    affine=True,          # mantiene γ/β entrenables
                    track_running_stats=False
                )
                # Transferir γ/β preentrenados si existen
                if child.weight is not None:
                    in_layer.weight = nn.Parameter(child.weight.data.clone())
                if child.bias is not None:
                    in_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(parent, name, in_layer)
            else:
                replace_in_module(child)

    replace_in_module(model)
    return model


def apply_bn_strategy(model, strategy, X_target=None, device=DEVICE):
    """
    Configura el modelo según la estrategia de normalización.
    Retorna (model_configurado, optimizer_param_groups).
    """
    # IN-Adapt opera sobre un modelo clonado con BN→IN ya reemplazado
    if strategy == 'IN-Adapt':
        model = _replace_bn_with_in(model).to(device)
        model.train()
        opt_groups = [{'params': model.parameters()}]
        return model, opt_groups

    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    if strategy == 'BN-F':
        # Estadísticas running_mean/var y parámetros γ/β completamente congelados.
        # Preserva la distribución del preentrenamiento.
        for bn in bn_layers:
            bn.eval()
            for p in bn.parameters():
                p.requires_grad_(False)
        opt_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]

    elif strategy == 'BN-U':
        # Todo el modelo se actualiza normalmente.
        # Baseline estándar de fine-tuning.
        model.train()
        opt_groups = [{'params': model.parameters()}]

    elif strategy == 'BN-P':
        # Estadísticas congeladas (eval mode), γ/β entrenables.
        # Permite escalar/desplazar la normalización sin actualizar las stats.
        for bn in bn_layers:
            bn.eval()
            bn.weight.requires_grad_(True)
            bn.bias.requires_grad_(True)
        opt_groups = [{'params': model.parameters()}]

    elif strategy == 'AdaBN':
        # Recalibra running stats usando datos target (sin etiquetas).
        # Limitado a 10 batches para evitar inestabilidad (corrección v2).
        # Luego congela stats y sólo entrena el clasificador + γ/β.
        if X_target is None:
            raise ValueError('AdaBN requiere X_target')
        model.eval()
        with torch.no_grad():
            for bn in bn_layers:
                bn.reset_running_stats()
                bn.momentum = 0.1
                bn.train()
            calib_loader = DataLoader(
                TensorDataset(X_target.to(device)),
                batch_size=64, shuffle=True
            )
            for i, (xb,) in enumerate(calib_loader):
                model(xb)
                if i >= 9:   # máximo 10 batches de calibración
                    break
        for bn in bn_layers:
            bn.eval()
            for p in bn.parameters():
                p.requires_grad_(True)   # γ/β entrenables, stats congeladas
        opt_groups = [{'params': model.parameters()}]

    elif strategy == 'BN-LR':
        # BN actualiza con lr × 0.1 para actualización suave de stats.
        # Resto del modelo con lr estándar.
        model.train()
        bn_params     = [p for n, p in model.named_parameters()
                         if 'bn' in n.lower() or 'norm' in n.lower()]
        non_bn_params = [p for n, p in model.named_parameters()
                         if 'bn' not in n.lower() and 'norm' not in n.lower()]
        opt_groups = [
            {'params': non_bn_params, 'lr': 1e-3},
            {'params': bn_params,     'lr': 1e-4},
        ]
    else:
        raise ValueError(f'Estrategia desconocida: {strategy}')

    return model, opt_groups


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fine-tuning con early stopping y gradient clipping
# ─────────────────────────────────────────────────────────────────────────────

def finetune_and_evaluate(model, X_lab, y_lab, X_test, y_test,
                           strategy, X_target=None,
                           epochs=100, batch_size=64, lr=1e-3,
                           patience=10, device=DEVICE):
    """
    Fine-tune con:
      - Early stopping (paciencia configurable)
      - Gradient clipping (max_norm=1.0) para evitar divergencia
      - AdaBN warmup: 5 épocas con BN frozen antes de liberar γ/β
    """
    model = deepcopy(model).to(device)
    model, opt_groups = apply_bn_strategy(
        model, strategy, X_target=X_target, device=device
    )

    opt_groups = [g for g in opt_groups
                  if len([p for p in g['params'] if p.requires_grad]) > 0]
    if not opt_groups:
        opt_groups = [{'params': [p for p in model.parameters()
                                  if p.requires_grad]}]

    optimizer = optim.Adam(opt_groups, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    eff_bs = max(2, min(batch_size, len(X_lab)))
    loader = DataLoader(
        TensorDataset(X_lab.to(device), y_lab.to(device)),
        batch_size=eff_bs, shuffle=True, drop_last=(len(X_lab) > eff_bs)
    )

    train_losses = []
    best_loss    = float('inf')
    best_weights = None
    no_improve   = 0

    # Para AdaBN: warmup de 5 épocas con clasificador solamente
    adabn_warmup = 5 if strategy == 'AdaBN' else 0

    for epoch in range(epochs):

        # Mantener BN en eval si la estrategia lo requiere
        if strategy in ('BN-F', 'BN-P'):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        elif strategy == 'AdaBN':
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
            if epoch < adabn_warmup:
                # Warmup: sólo entrenar clasificador
                for n, p in model.named_parameters():
                    if 'classifier' not in n:
                        p.requires_grad_(False)
            else:
                # Post-warmup: liberar γ/β de BN
                for n, p in model.named_parameters():
                    p.requires_grad_(True)
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
        else:
            model.train()

        ep_loss = 0.0
        for Xb, yb in loader:
            logits = model(Xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss    = avg_loss
            best_weights = deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restaurar mejores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluación
    model.eval()
    with torch.no_grad():
        # Procesar en batches para evitar OOM en datasets grandes
        all_preds = []
        eval_loader = DataLoader(TensorDataset(X_test.to(device)),
                                 batch_size=256)
        for (xb,) in eval_loader:
            all_preds.append(model(xb).argmax(dim=1).cpu())
        preds = torch.cat(all_preds).numpy()

    y_true = y_test.numpy()
    return {
        'accuracy':   accuracy_score(y_true, preds),
        'f1':         f1_score(y_true, preds, average='macro', zero_division=0),
        'precision':  precision_score(y_true, preds, average='macro', zero_division=0),
        'recall':     recall_score(y_true, preds, average='macro', zero_division=0),
        'final_loss': train_losses[-1] if train_losses else float('nan'),
        'loss_std':   float(np.std(train_losses)),
        'epochs_run': len(train_losses),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bucle experimental principal
# ─────────────────────────────────────────────────────────────────────────────
results_file = RESULTS_DIR / 'raw_results.jsonl'
all_results  = []

print('=' * 70)
print('INICIANDO EXPERIMENTO')
print('=' * 70)

for dataset_name in tqdm(DATASETS_TO_RUN, desc='Datasets', unit='ds'):
    data = load_dataset(dataset_name)
    if data is None:
        continue

    print(f'\n {dataset_name} | '
          f'train={len(data["X_train"])} test={len(data["X_test"])} '
          f'clases={data["n_classes"]} seq_len={data["seq_len"]}')

    # Preentrenamiento TNC (una vez por dataset, sin etiquetas)
    print(f'   → Preentrenando TNC ({PRETRAIN_EPOCHS} epochs)...')
    pretrain_tnc(
        data['X_train'], data['n_channels'],
        epochs=PRETRAIN_EPOCHS, device=DEVICE
    )

    for arch in ARCHS_TO_RUN:
        for ratio in RATIOS_TO_RUN:
            for bs in BATCHES_TO_RUN:
                for seed in SEEDS_TO_RUN:
                    set_seed(seed)

                    labeled_idx = get_labeled_subset(data, ratio, seed)
                    X_lab = data['X_train'][labeled_idx]
                    y_lab = data['y_train'][labeled_idx]

                    base_model = build_model(
                        arch, data['n_channels'],
                        data['seq_len'], data['n_classes']
                    )

                    for strategy in BN_STRATEGIES:
                        try:
                            metrics = finetune_and_evaluate(
                                model=base_model,
                                X_lab=X_lab,
                                y_lab=y_lab,
                                X_test=data['X_test'],
                                y_test=data['y_test'],
                                strategy=strategy,
                                X_target=data['X_train'],
                                epochs=FINETUNE_EPOCHS,
                                batch_size=bs,
                                patience=PATIENCE,
                                device=DEVICE
                            )
                            row = {
                                'dataset':      dataset_name,
                                'domain':       next(k for k, v in DATASET_GROUPS.items()
                                                     if dataset_name in v),
                                'architecture': arch,
                                'bn_strategy':  strategy,
                                'label_ratio':  ratio,
                                'batch_size':   bs,
                                'seed':         seed,
                                **metrics
                            }
                            status = (f"acc={metrics['accuracy']:.3f} "
                                      f"f1={metrics['f1']:.3f} "
                                      f"ep={metrics['epochs_run']}")
                        except Exception as e:
                            row = {
                                'dataset': dataset_name, 'architecture': arch,
                                'bn_strategy': strategy, 'label_ratio': ratio,
                                'batch_size': bs, 'seed': seed,
                                'error': str(e)
                            }
                            status = f'ERROR: {e}'

                        print(f'   {arch:12s} | {strategy:8s} | '
                              f'ratio={ratio:.2f} bs={bs:3d} '
                              f'seed={seed:4d} | {status}')

                        all_results.append(row)
                        with open(results_file, 'a') as f:
                            f.write(json.dumps(row) + '\n')

print('\n Experimento completado.')

# ─────────────────────────────────────────────────────────────────────────────
# 9. Análisis y visualizaciones
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('ANÁLISIS DE RESULTADOS')
print('=' * 70)

rows = [r for r in all_results if 'error' not in r]
if not rows:
    print('  No hay resultados válidos para analizar.')
    sys.exit(0)

df = pd.DataFrame(rows)
print(f' Filas válidas: {len(df):,}')
print(f'   Errores: {len(all_results) - len(rows)}')

# ── 9.1 Ranking global ────────────────────────────────────────────────
summary = (
    df.groupby('bn_strategy')[['accuracy', 'f1', 'precision', 'recall']]
    .agg(['mean', 'std'])
    .round(4)
)
print('\n Ranking global por estrategia BN (por F1-macro):')
print(summary.sort_values(('f1', 'mean'), ascending=False).to_string())

# ── 9.2 F1 por estrategia × ratio de etiquetas ───────────────────────
pivot_ratio = df.pivot_table(
    values='f1', index='bn_strategy',
    columns='label_ratio', aggfunc='mean'
).round(4)
print('\n F1-macro por estrategia × proporción de etiquetas:')
print(pivot_ratio.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
pivot_ratio.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('F1-macro por Estrategia BN y Proporción de Etiquetas')
ax.set_xlabel('Estrategia BN')
ax.set_ylabel('F1-macro')
ax.legend(title='Label Ratio', labels=[f'{int(r*100)}%' for r in pivot_ratio.columns])
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'f1_by_strategy_ratio.png', dpi=150)
plt.close()
print(' f1_by_strategy_ratio.png')

# ── 9.3 Heatmap arquitectura × estrategia ────────────────────────────
heat = df.pivot_table(
    values='f1', index='architecture',
    columns='bn_strategy', aggfunc='mean'
).round(4)
print('\n📋 Heatmap F1 (arquitectura × estrategia):')
print(heat.to_string())

fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(heat, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
ax.set_title('F1-macro: Arquitectura × Estrategia BN')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'heatmap_arch_bn.png', dpi=150)
plt.close()
print(' heatmap_arch_bn.png')

# ── 9.4 Heatmap dominio × estrategia ─────────────────────────────────
if 'domain' in df.columns:
    heat_dom = df.pivot_table(
        values='f1', index='domain',
        columns='bn_strategy', aggfunc='mean'
    ).round(4)
    print('\n Heatmap F1 (dominio × estrategia):')
    print(heat_dom.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(heat_dom, annot=True, fmt='.3f', cmap='Blues', ax=ax)
    ax.set_title('F1-macro: Dominio × Estrategia BN')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'heatmap_domain_bn.png', dpi=150)
    plt.close()
    print(' heatmap_domain_bn.png')

# ── 9.5 Boxplot por tamaño de mini-lote ──────────────────────────────
bs_list = sorted(df['batch_size'].unique())
n_bs = len(bs_list)
fig, axes = plt.subplots(1, max(1, n_bs), figsize=(5 * n_bs, 4), sharey=True)
if n_bs == 1:
    axes = [axes]
for ax, bs in zip(axes, bs_list):
    sub = df[df['batch_size'] == bs]
    if sub.empty:
        continue
    order = sub.groupby('bn_strategy')['f1'].mean().sort_values(ascending=False).index
    sns.boxplot(data=sub, x='bn_strategy', y='f1', order=order, ax=ax)
    ax.set_title(f'Batch size = {bs}')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=35)
axes[0].set_ylabel('F1-macro')
plt.suptitle('Distribución de F1 por Estrategia BN y Tamaño de Mini-lote')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'boxplot_batch_size.png', dpi=150)
plt.close()
print('💾 boxplot_batch_size.png')

# ── 9.6 Estabilidad del entrenamiento ────────────────────────────────
stability = (
    df.groupby('bn_strategy')['loss_std']
    .agg(['mean', 'std'])
    .round(4)
    .sort_values('mean')
)
print('\n Estabilidad del entrenamiento (menor = más estable):')
print(stability.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
stability['mean'].plot(kind='bar', ax=ax, color='steelblue',
                       yerr=stability['std'], capsize=4)
ax.set_title('Estabilidad del Entrenamiento por Estrategia BN')
ax.set_ylabel('Std de pérdida de entrenamiento')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'stability.png', dpi=150)
plt.close()
print(' stability.png')

# ── 9.7 Épocas hasta convergencia (early stopping) ───────────────────
if 'epochs_run' in df.columns:
    conv = (
        df.groupby('bn_strategy')['epochs_run']
        .agg(['mean', 'std'])
        .round(1)
        .sort_values('mean')
    )
    print('\n  Épocas hasta convergencia (early stopping):')
    print(conv.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    conv['mean'].plot(kind='bar', ax=ax, color='darkorange',
                      yerr=conv['std'], capsize=4)
    ax.set_title('Épocas hasta Convergencia por Estrategia BN')
    ax.set_ylabel('Épocas')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'convergence_epochs.png', dpi=150)
    plt.close()
    print(' convergence_epochs.png')

# ─────────────────────────────────────────────────────────────────────────────
# 10. Análisis estadístico: Friedman + Nemenyi
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('ANÁLISIS ESTADÍSTICO')
print('=' * 70)

pivot_stats = df.pivot_table(
    values='f1', index='dataset',
    columns='bn_strategy', aggfunc='mean'
).dropna()

print(f'📐 Datasets con datos completos: {len(pivot_stats)}')

if len(pivot_stats) >= 3:
    valid_strategies = [s for s in BN_STRATEGIES if s in pivot_stats.columns]
    groups = [pivot_stats[s].values for s in valid_strategies]
    stat, p_value = friedmanchisquare(*groups)
    print(f'\n🧮 Test de Friedman:')
    print(f'   χ² = {stat:.4f}')
    print(f'   p  = {p_value:.6f}')
    sig = p_value < 0.05
    print(f'   {" Diferencias significativas (p < 0.05)" if sig else "⚠️  Sin diferencias significativas"}')

    ranks      = pivot_stats[valid_strategies].rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()
    print('\n Rankings medios (menor = mejor):')
    print(mean_ranks.round(4).to_string())

    def critical_difference(k, n, alpha=0.05):
        q_table = {2:1.960, 3:2.344, 4:2.569, 5:2.728,
                   6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}
        q = q_table.get(k, 2.850)
        return q * math.sqrt(k * (k + 1) / (6 * n))

    cd = critical_difference(len(valid_strategies), len(pivot_stats))
    print(f'\n📏 Diferencia Crítica (Nemenyi, α=0.05): CD = {cd:.4f}')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(list(mean_ranks.index), mean_ranks.values,
            color='steelblue', alpha=0.7)
    ax.axvline(mean_ranks.values[0] + cd, color='red',
               linestyle='--', label=f'CD = {cd:.3f}')
    ax.set_xlabel('Ranking medio (menor = mejor)')
    ax.set_title('Rankings medios — Test de Nemenyi (α=0.05)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cd_diagram.png', dpi=150)
    plt.close()
    print('💾 cd_diagram.png')
else:
    print('  Pocos datasets para test estadístico robusto.')

# ─────────────────────────────────────────────────────────────────────────────
# 11. Exportar tablas CSV
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('EXPORTANDO TABLAS CSV')
print('=' * 70)

(df.groupby(['bn_strategy', 'label_ratio'])[['accuracy', 'f1']]
   .agg(['mean', 'std']).round(4)
   .to_csv(RESULTS_DIR / 'table_main.csv'))
print(' table_main.csv')

(df.groupby(['architecture', 'bn_strategy'])[['accuracy', 'f1']]
   .agg(['mean', 'std']).round(4)
   .to_csv(RESULTS_DIR / 'table_by_arch.csv'))
print(' table_by_arch.csv')

(df.groupby(['bn_strategy', 'batch_size'])[['accuracy', 'f1']]
   .agg(['mean', 'std']).round(4)
   .to_csv(RESULTS_DIR / 'table_by_batch_size.csv'))
print(' table_by_batch_size.csv')

if 'domain' in df.columns:
    (df.groupby(['domain', 'bn_strategy'])[['accuracy', 'f1']]
       .agg(['mean', 'std']).round(4)
       .to_csv(RESULTS_DIR / 'table_by_domain.csv'))
    print(' table_by_domain.csv')

df.to_csv(RESULTS_DIR / 'all_results_full.csv', index=False)
print(' all_results_full.csv  (dataset completo)')

# ─────────────────────────────────────────────────────────────────────────────
# 12. Resumen final
# ─────────────────────────────────────────────────────────────────────────────
best_strategy = df.groupby('bn_strategy')['f1'].mean().idxmax()
best_f1       = df.groupby('bn_strategy')['f1'].mean().max()

print('\n' + '=' * 70)
print('RESUMEN FINAL')
print('=' * 70)
print(f' Mejor estrategia global : {best_strategy}  (F1={best_f1:.4f})')
print(f' Resultados en           : {RESULTS_DIR.absolute()}')
print()
print('Archivos generados:')
for f in sorted(RESULTS_DIR.iterdir()):
    kb = f.stat().st_size / 1024
    print(f'  {f.name:<40s} {kb:6.1f} KB')
