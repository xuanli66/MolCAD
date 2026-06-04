
import os, random, numpy as np, torch
import math
import argparse
import sys
import atexit
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.unified_encoder import UnifiedMoleculeEncoder
from models.smiles_encoder import SmilesCNNEncoder, CharSmilesTokenizer, get_smiles_list
from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        for s in self._streams:
            isatty = getattr(s, "isatty", None)
            if callable(isatty) and isatty():
                return True
        return False

# =========================================================
DEFAULT_SEED = 40
DEFAULT_SPLIT_SEED = 40
DEFAULT_DATASET_NAME = "SIDER"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50

parser = argparse.ArgumentParser(description="MVFFG training (random split with reproducibility)")
parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="training seed (init/shuffle/dropout)")
parser.add_argument("--split_seed", type=int, default=None, help="random scaffold split seed (default: use --seed)")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--p_feat", type=float, default=0.2, help="feature dropout prob before fusion")
parser.add_argument("--lambda_align", type=float, default=0.2, help="InfoNCE alignment loss weight")
parser.add_argument("--lam_distill", type=float, default=0.3, help="cross-modal distillation max weight")
parser.add_argument("--initial_lr", type=float, default=1e-4, help="AdamW initial learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-3, help="AdamW weight decay")

args = parser.parse_args()

SEED = int(args.seed)

if args.split_seed is None:
    SPLIT_SEED = SEED
else:
    SPLIT_SEED = int(args.split_seed)

dataset_name = str(args.dataset_name)
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)

p_feat = float(args.p_feat)
is_dropout = True

ALIGN_DIM = 128
tau = 0.07
lambda_align = float(args.lambda_align)

lam_distill = float(args.lam_distill)
distill_dim = 256
distill_type = "cos"
distill_warmup = 5
distill_ramp = 30

lambda_kd = 0.1
T_kd = 3.0

aux_warmup = 10
aux_ramp   = 20

modality_dropout_p = 0.1
modality_dropout_mode = "one"

INITIAL_LR = float(args.initial_lr)
WEIGHT_DECAY = float(args.weight_decay)
ETA_MIN = 1e-6

def seed_everything(seed=SEED, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic, warn_only=False)

seed_everything(SEED)

_final_macro_auc_for_logname = None
_orig_stdout, _orig_stderr = sys.stdout, sys.stderr

train_log_dir = ROOT / "logs" / "train_logs"
train_log_dir.mkdir(parents=True, exist_ok=True)
_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
_tmp_log_path = train_log_dir / f"{dataset_name}_seed{SEED}_split{SPLIT_SEED}_bs{BATCH_SIZE}_{_run_id}.log"
_log_fh = open(_tmp_log_path, "w", encoding="utf-8", buffering=1)

sys.stdout = _TeeIO(_orig_stdout, _log_fh)
sys.stderr = _TeeIO(_orig_stderr, _log_fh)

def _finalize_log_file():
    global _log_fh

    try:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        try:
            if _log_fh is not None:
                _log_fh.flush()
                _log_fh.close()
        finally:
            _log_fh = None
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr

    if _final_macro_auc_for_logname is None:
        return str(_tmp_log_path)

    final_log_path = train_log_dir / (
        f"{dataset_name}_seed{SEED}_split{SPLIT_SEED}_bs{BATCH_SIZE}_AUC{_final_macro_auc_for_logname:.4f}.log"
    )

    if final_log_path.exists():
        stem = final_log_path.stem
        suffix = final_log_path.suffix
        k = 1
        while True:
            cand = final_log_path.with_name(f"{stem}_{k}{suffix}")
            if not cand.exists():
                final_log_path = cand
                break
            k += 1

    try:
        _tmp_log_path.rename(final_log_path)
        return str(final_log_path)
    except Exception:
        return str(_tmp_log_path)


def _close_log_on_exit():

    global _log_fh
    try:
        if _log_fh is not None:
            try:
                _log_fh.flush()
                _log_fh.close()
            finally:
                _log_fh = None
    except Exception:
        pass


atexit.register(_close_log_on_exit)

class MultiModalDataset(Dataset):
    def __init__(self, base_dataset, indices, smiles_list, tokenizer, max_len=256):
        self.base = base_dataset
        self.indices = list(indices)
        self.smiles_list = smiles_list
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gidx = self.indices[i]
        data = self.base[gidx]
        data.idx = torch.tensor(gidx, dtype=torch.long)

        s = self.smiles_list[gidx] if self.smiles_list is not None else None
        ids, mask = self.tok.encode(s, max_len=self.max_len)
        return data, torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def collate_mm(batch):
    data_list, ids_list, mask_list = zip(*batch)
    pyg_batch = Batch.from_data_list(list(data_list))
    ids = torch.stack(ids_list, dim=0)
    mask = torch.stack(mask_list, dim=0)
    return pyg_batch, ids, mask

def random_split(dataset, seed=42, frac_train=0.8, frac_val=0.1):
    num_samples = len(dataset)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g).tolist()
    train_split = int(frac_train * num_samples)
    val_split = int((frac_train + frac_val) * num_samples)
    train_idx = perm[:train_split]
    val_idx = perm[train_split:val_split]
    test_idx = perm[val_split:]
    return train_idx, val_idx, test_idx

def pahelix_random_scaffold_split(smiles_list, seed=None, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    try:
        from utils.splitters import RandomScaffoldSplitter
    except Exception as e:
        raise ImportError("Cannot import pahelix.utils.splitters.RandomScaffoldSplitter. "
                          "Please ensure pahelix is installed and importable.") from e

    ph_dataset = np.array(
    [{"smiles": smiles_list[i], "idx": i} for i in range(len(smiles_list))],
    dtype=object)

    splitter = RandomScaffoldSplitter()
    train_set, valid_set, test_set = splitter.split(
        ph_dataset,
        frac_train=frac_train,
        frac_valid=frac_val,
        frac_test=frac_test,
        seed=seed,
    )

    train_idx = [x["idx"] for x in train_set]
    val_idx   = [x["idx"] for x in valid_set]
    test_idx  = [x["idx"] for x in test_set]
    return train_idx, val_idx, test_idx

def distill_loss_fn(teacher_z, student_z, kind="cos"):

    if kind == "cos":
        t = F.normalize(teacher_z, dim=1)
        s = F.normalize(student_z, dim=1)
        return 1.0 - (t * s).sum(dim=1).mean()
    elif kind == "mse":

        t = F.normalize(teacher_z, dim=1)
        s = F.normalize(student_z, dim=1)
        return F.mse_loss(s, t)
    else:
        raise ValueError(f"Unknown distill kind: {kind}")

def distill_lambda_schedule_cos(epoch, max_lam=0.2, warmup=5, ramp=20):
    if epoch <= warmup:
        return 0.0
    t = (epoch - warmup) / float(max(1, ramp))
    t = max(0.0, min(1.0, t))
    return max_lam * 0.5 * (1 - math.cos(math.pi * t))

def apply_modality_dropout(graph_feat, fp_feat, smiles_feat, p=0.1, mode="one"):

    if p <= 0:
        return graph_feat, fp_feat, smiles_feat

    if mode == "one":

        if torch.rand(1).item() < p:
            k = torch.randint(0, 3, (1,)).item()
            if k == 0:
                graph_feat = torch.zeros_like(graph_feat)
            elif k == 1:
                fp_feat = torch.zeros_like(fp_feat)
            else:
                smiles_feat = torch.zeros_like(smiles_feat)
    elif mode == "independent":
        if torch.rand(1).item() < p:
            graph_feat = torch.zeros_like(graph_feat)
        if torch.rand(1).item() < p:
            fp_feat = torch.zeros_like(fp_feat)
        if torch.rand(1).item() < p:
            smiles_feat = torch.zeros_like(smiles_feat)

    return graph_feat, fp_feat, smiles_feat

def linear_ramp(epoch, max_w, warmup=10, ramp=20):
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(max_w)
    t = (epoch - warmup) / float(ramp)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return float(max_w) * t

@torch.no_grad()
def compute_pos_weight_from_indices(base_dataset, train_indices, num_tasks, eps=1.0, clamp_max=50.0):

    pos = torch.zeros(num_tasks, dtype=torch.float64)
    neg = torch.zeros(num_tasks, dtype=torch.float64)

    for gidx in train_indices:
        y = base_dataset[gidx].y
        if y.dim() == 0:
            y = y.view(1)
        if y.dim() == 1:
            y = y.view(1, -1)
        y = y.squeeze(0).double()

        m = ~torch.isnan(y)
        if m.any():
            yv = y[m]
            pos[m] += (yv > 0).sum().item()
            neg[m] += (yv <= 0).sum().item()

    pos_weight = (neg + eps) / (pos + eps)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=clamp_max).float()
    return pos_weight

# =========================================================
dataset = MoleculeNet(root="../data", name=dataset_name)

smiles_list = get_smiles_list(dataset)
tokenizer = CharSmilesTokenizer(smiles_list)
print("[SMILES] vocab_size =", tokenizer.vocab_size)

if hasattr(dataset, "num_tasks") and dataset.num_tasks is not None:
    output_dim = dataset.num_tasks
else:
    y0 = dataset[0].y
    output_dim = y0.shape[-1] if y0.dim() > 0 else 1

split_dir = ROOT / "data" / "splits"
split_dir.mkdir(parents=True, exist_ok=True)


scaffold_datasets = {"BBBP", "BACE"}
dataset_key = str(dataset_name).upper()

if dataset_key in scaffold_datasets:
    split_strategy = "pahelix_random_scaffold"
    split_path = split_dir / f"{dataset_name}_split_{split_strategy}_seed{SPLIT_SEED}.npz"
else:
    split_strategy = "random"
    split_path = split_dir / f"{dataset_name}_split_{split_strategy}_seed{SPLIT_SEED}.npz"
split_path = str(split_path)

if os.path.exists(split_path):
    arr = np.load(split_path)
    train_idx, val_idx, test_idx = arr["train"].tolist(), arr["val"].tolist(), arr["test"].tolist()
else:
    if dataset_key in scaffold_datasets:
        train_idx, val_idx, test_idx = pahelix_random_scaffold_split(
            smiles_list,
            seed=SPLIT_SEED,
            frac_train=0.8,
            frac_val=0.1,
            frac_test=0.1,
        )
        np.savez(split_path, train=np.array(train_idx), val=np.array(val_idx), test=np.array(test_idx))
        print(f"Created pahelix random scaffold split for {dataset_name} and saved to {split_path}")
    else:
        train_idx, val_idx, test_idx = random_split(dataset, seed=SPLIT_SEED, frac_train=0.8, frac_val=0.1)
        np.savez(split_path, train=np.array(train_idx), val=np.array(val_idx), test=np.array(test_idx))
        print(f"Created random split for {dataset_name} and saved to {split_path}")

train_dataset = MultiModalDataset(dataset, train_idx, smiles_list, tokenizer, max_len=256)
val_dataset = MultiModalDataset(dataset, val_idx, smiles_list, tokenizer, max_len=256)
test_dataset = MultiModalDataset(dataset, test_idx, smiles_list, tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_mm)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mm)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mm)

# =========================================================
fp_path = ROOT / "data" / dataset_name / "features" / "ecfp.npy"
fp_path = str(fp_path)

fp_np = np.load(fp_path)
fp_tensor = torch.from_numpy(fp_np).float()

FP_DIM = fp_tensor.shape[1]
print(f"[Fingerprint] Loaded: {fp_path} | shape={tuple(fp_tensor.shape)} | FP_DIM={FP_DIM}")

fp_proj = nn.Sequential(
    nn.Linear(FP_DIM, 256),
    nn.ReLU(inplace=True),
    nn.LayerNorm(256),
    nn.Dropout(0.2)
).to(device)

smiles_encoder = SmilesCNNEncoder(
    vocab_size=tokenizer.vocab_size,
    emb_dim=128,
    out_dim=256,
    channels=128,
    kernels=(3, 5, 7),
    dropout=0.2,
    pad_id=tokenizer.pad_id
).to(device)

# =========================================================
model = UnifiedMoleculeEncoder(graph_hidden_dim=128).to(device)

print(f"[Architecture] Graph feature dimension: {model.output_dim} (no projection)")

tmp_batch, tmp_ids, tmp_mask = next(iter(train_loader))
tmp_batch = tmp_batch.to(device)

ea = tmp_batch.edge_attr
print("[CHECK] edge_attr dtype:", None if ea is None else ea.dtype)
print("[CHECK] edge_attr shape:", None if ea is None else tuple(ea.shape))
if ea is not None:
    print("[CHECK] edge_attr min/max:", ea.min().item(), ea.max().item())
    print("[CHECK] first row:", ea[0])

with torch.no_grad():
    tmp_g = model(tmp_batch) if True else model(tmp_batch, None)
graph_dim = tmp_g.shape[1]
print("[Init] inferred graph_dim =", graph_dim)

# 三路归一化（简单、有效）
graph_norm = nn.LayerNorm(graph_dim).to(device)
fp_norm = nn.LayerNorm(256).to(device)
smiles_norm = nn.LayerNorm(256).to(device)

fusion_in_dim = graph_dim + 256 + 256
final_fusion_layer = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(fusion_in_dim, 256),
    nn.ReLU(inplace=True),
    nn.LayerNorm(256),
    nn.Dropout(0.4),
    nn.Linear(256, output_dim)
).to(device)

# ===== Gated fusion  =====
fusion_gate = nn.Sequential(
    nn.Linear(fusion_in_dim, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 3)
).to(device)

print(f"[Fusion] graph({graph_dim}) + fp(256) + smiles(256) = {fusion_in_dim}")

# =========================================================
fp_align_head = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, ALIGN_DIM)
).to(device)

sm_align_head = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, ALIGN_DIM)
).to(device)

ce = nn.CrossEntropyLoss()

# =========================================================
teacher_in_dim = 256 + 256

teacher_proj = nn.Sequential(
    nn.Linear(teacher_in_dim, distill_dim),
    nn.ReLU(inplace=True),
    nn.LayerNorm(distill_dim),
).to(device)

student_proj = nn.Sequential(
    nn.Linear(graph_dim, 128),
    nn.ReLU(inplace=True),
    nn.LayerNorm(128),
).to(device)
# =========================================================
optimizer = torch.optim.AdamW(
    [p for p in (list(model.parameters()) +
                 list(fp_proj.parameters()) +
                 list(smiles_encoder.parameters()) +
                 list(final_fusion_layer.parameters()) +
                 list(fusion_gate.parameters()) +
                 list(student_proj.parameters()) +
                 list(fp_align_head.parameters()) +
                 list(sm_align_head.parameters()) +
                 list(graph_norm.parameters()) +
                 list(fp_norm.parameters()) +
                 list(smiles_norm.parameters()))
     if p.requires_grad],
    lr=INITIAL_LR,
    weight_decay=WEIGHT_DECAY
)

warmup_epochs = 10
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=ETA_MIN)

pos_weight = compute_pos_weight_from_indices(
    base_dataset=dataset,
    train_indices=train_idx,
    num_tasks=output_dim,
    eps=1.0,
    clamp_max=20.0
).to(device)


bce_none = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

# =========================================================
best_val_auc = -1.0
save_dir = ROOT / "ckpt" / "runs"
save_dir.mkdir(parents=True, exist_ok=True)
best_model_path = save_dir / f"{dataset_name}_best_seed{SEED}_split{SPLIT_SEED}.pth"
best_model_path = str(best_model_path)

train_auc_log, val_auc_log = [], []


def one_pass(loader, train=False, epoch=0):
    modules = [model, fp_proj, smiles_encoder,
               fp_align_head, sm_align_head,
               teacher_proj, student_proj,
               graph_norm, fp_norm, smiles_norm,
               final_fusion_layer]
    for m in modules:
        m.train() if train else m.eval()

    sum_loss = 0.0
    total_distill = 0
    n_batches = 0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(train):
        for i, (data, ids, mask) in enumerate(loader):
            data = data.to(device)
            ids = ids.to(device)
            mask = mask.to(device)

            graph_feat = model(data)
            graph_feat = graph_norm(graph_feat)

            idx_cpu = data.idx.view(-1).long().cpu()
            fp_batch = fp_tensor[idx_cpu].to(device)
            fp_feat = fp_proj(fp_batch)
            fp_feat = fp_norm(fp_feat)

            smiles_feat = smiles_encoder(ids, mask)
            smiles_feat = smiles_norm(smiles_feat)

            if train:
                graph_feat, fp_feat, smiles_feat = apply_modality_dropout(
                    graph_feat, fp_feat, smiles_feat,
                    p=modality_dropout_p,
                    mode=modality_dropout_mode
                )

            if is_dropout:
                graph_feat = F.dropout(graph_feat, p=p_feat, training=train)
                fp_feat = F.dropout(fp_feat, p=p_feat, training=train)
                smiles_feat = F.dropout(smiles_feat, p=p_feat, training=train)

            gate_in = torch.cat([graph_feat, fp_feat, smiles_feat], dim=1)
            w = torch.softmax(fusion_gate(gate_in), dim=1)
            w_g = w[:, 0:1]
            w_fp = w[:, 1:2]
            w_sm = w[:, 2:3]

            fusion_in = torch.cat([w_g * graph_feat, w_fp * fp_feat, w_sm * smiles_feat], dim=1)
            out = final_fusion_layer(fusion_in)

            kd_loss = 0.0
            if train and lambda_kd > 0:

                was_training = final_fusion_layer.training
                final_fusion_layer.eval()
                with torch.no_grad():
                    zero_g = torch.zeros_like(graph_feat)
                    teacher_in = torch.cat([zero_g, fp_feat, smiles_feat], dim=1)
                    t_logits = final_fusion_layer(teacher_in)
                    t_prob = torch.sigmoid(t_logits / T_kd)
                if was_training:
                    final_fusion_layer.train()

                kd_loss = F.binary_cross_entropy_with_logits(out / T_kd, t_prob) * (T_kd * T_kd)

            target = data.y.float()
            if target.dim() == 1:
                target = target.view(-1, output_dim)
            elif target.shape[-1] != output_dim:
                target = target.view(-1, output_dim)

            target_use = torch.nan_to_num(target, nan=0.0)
            loss_raw = bce_none(out, target_use)
            mask_y = ~torch.isnan(target)
            task_loss = loss_raw[mask_y].mean()

            B = fp_feat.size(0)
            labels = torch.arange(B, device=device)

            p_fp = F.normalize(fp_align_head(fp_feat), dim=1)
            p_sm = F.normalize(sm_align_head(smiles_feat), dim=1)
            logits_align = (p_fp @ p_sm.t()) / tau

            loss_fp2sm = ce(logits_align, labels)
            loss_sm2fp = ce(logits_align.t(), labels)
            align_loss = 0.5 * (loss_fp2sm + loss_sm2fp)

            if train and lam_distill > 0:
                t_fp = fp_align_head(fp_feat)
                t_sm = sm_align_head(smiles_feat)

                teacher_z = 0.5 * (t_fp + t_sm)
                teacher_z = teacher_z.detach()
                student_z = student_proj(graph_feat)
                distill_loss = distill_loss_fn(teacher_z, student_z, kind=distill_type)

                total_distill += float(distill_loss.detach().cpu().item())

            else:
                distill_loss = 0.0
            n_batches += 1

            lam_align_now = linear_ramp(epoch, lambda_align, warmup=aux_warmup, ramp=aux_ramp)
            lam_kd_now    = linear_ramp(epoch, lambda_kd,    warmup=aux_warmup, ramp=aux_ramp)
            lam_distill_now = distill_lambda_schedule_cos(epoch, max_lam=lam_distill, warmup=distill_warmup, ramp=distill_ramp)

            batch_loss = task_loss \
                + (lam_align_now * align_loss if train else 0.0) \
                + (lam_kd_now * kd_loss if train else 0.0) \
                + (lam_distill_now * distill_loss if train else 0.0)

            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            sum_loss += float(batch_loss.item() if torch.is_tensor(batch_loss) else batch_loss)

            all_labels.append(target.detach().cpu())
            all_probs.append(torch.sigmoid(out).detach().cpu())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    auc_list = []
    for k in range(output_dim):
        yk = all_labels[:, k]
        pk = all_probs[:, k]
        m = ~np.isnan(yk)
        if m.sum() == 0:
            continue
        yv = yk[m]
        pv = pk[m]
        if len(np.unique(yv)) == 2:
            auc_list.append(roc_auc_score(yv, pv))
    if len(auc_list):
        macro_auc = float(np.mean(auc_list))
    else:
        raise RuntimeError("No valid AUC could be computed.")

    avg_loss = sum_loss / max(1, len(loader))
    avg_distill = total_distill / max(1, n_batches)
    return avg_loss, macro_auc, avg_distill


for epoch in range(1, EPOCHS + 1):
    # warmup
    if epoch <= warmup_epochs:
        for pg in optimizer.param_groups:
            pg["lr"] = INITIAL_LR * (epoch / warmup_epochs)

    train_loss, train_auc, train_distill = one_pass(train_loader, train=True, epoch=epoch)
    val_loss, val_auc, _ = one_pass(val_loader, train=False, epoch=epoch)

    if epoch > warmup_epochs:
        scheduler.step()

    train_auc_log.append(train_auc)
    val_auc_log.append(val_auc)

    print(f"Epoch {epoch:03d} | "
          f"Train Loss: {train_loss:.4f} | Train Macro AUC: {train_auc:.4f} | Distill: {train_distill:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save({
            "model": model.state_dict(),
            "fusion_gate": fusion_gate.state_dict(),
            "fp_proj": fp_proj.state_dict(),
            "smiles_encoder": smiles_encoder.state_dict(),
            "fp_align_head": fp_align_head.state_dict(),
            "sm_align_head": sm_align_head.state_dict(),
            "student_proj": student_proj.state_dict(),
            "final_fusion_layer": final_fusion_layer.state_dict(),
            "val_auc": float(val_auc),
        }, best_model_path)
        print(f"   Saved BEST (Val Macro AUC = {val_auc:.4f}) -> {best_model_path}")

ckpt = torch.load(best_model_path, map_location=device)

model.load_state_dict(ckpt["model"])
fusion_gate.load_state_dict(ckpt["fusion_gate"])
final_fusion_layer.load_state_dict(ckpt["final_fusion_layer"])
fp_proj.load_state_dict(ckpt["fp_proj"])
smiles_encoder.load_state_dict(ckpt["smiles_encoder"])

test_loss, test_auc, _ = one_pass(test_loader, train=False, epoch=0)
print(f"\n=== Final Test (Last Epoch Weights) ===")
print(f"Loss: {test_loss:.4f} | Macro ROC-AUC: {test_auc:.4f}")

_final_macro_auc_for_logname = float(test_auc)

plt.plot(range(1, len(train_auc_log) + 1), train_auc_log, marker="o", label="Train AUC")
plt.plot(range(1, len(val_auc_log) + 1), val_auc_log, marker="s", label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("Macro ROC-AUC")
plt.title(f"{dataset_name} - AUC Curves (seed={SEED}) [random split only]")
plt.grid(True)
plt.legend()
plot_dir = ROOT / "logs" / "plot_logs"
plot_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_dir / f"{dataset_name}_auc_curves_seed{SEED}_random.png", dpi=300, bbox_inches="tight")
plt.show()

final_log_path = _finalize_log_file()
print(f"\n[Train Log Saved] {final_log_path}")
