
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
from typing import Tuple
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

DEFAULT_SEED = 40
DEFAULT_SPLIT_SEED = 40
DEFAULT_DATASET_NAME = "ESOL"  # regression: 'ESOL'/'FREESOLV'/'LIPO'
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 50

parser = argparse.ArgumentParser(description="MVFFG training (random split with reproducibility)")
parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="training seed (init/shuffle/dropout)")
parser.add_argument("--split_seed", type=int, default=None, help="random split seed (default: use --seed)")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--p_feat", type=float, default=0.3, help="feature dropout prob before fusion")
parser.add_argument("--lambda_align", type=float, default=0.2, help="InfoNCE alignment loss weight")
parser.add_argument("--lam_distill", type=float, default=0.3, help="cross-modal distillation max weight")
parser.add_argument("--initial_lr", type=float, default=1e-4, help="AdamW initial learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="AdamW weight decay")
parser.add_argument("--single_modality", type=str, default="all", choices=["all", "graph", "fp", "seq"],
                    help="modality ablation mode: all|graph|fp|seq")
args = parser.parse_args()
SINGLE_MODALITY = str(args.single_modality).strip().lower()

SEED = int(args.seed)

if args.split_seed is None:
    SPLIT_SEED = SEED
else:
    SPLIT_SEED = int(args.split_seed)

dataset_input_name = str(args.dataset_name)
DATASET_KEY_UPPER = dataset_input_name.strip().upper()
_dataset_alias = {
    "DELANEY": "ESOL",
    "FREESOLV": "FreeSolv",
    "LIPO": "Lipo",
    "LIPOPHILICITY": "Lipo",
}
dataset_name = _dataset_alias.get(DATASET_KEY_UPPER, dataset_input_name)
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)

p_feat = float(args.p_feat)
is_dropout = True

ALIGN_DIM = 128
tau = 0.07
lambda_align = float(args.lambda_align)
if SINGLE_MODALITY != "all":
    lambda_align = 0.0

lam_distill = float(args.lam_distill) if SINGLE_MODALITY == "all" else 0.0
distill_dim = 256
distill_type = "cos"
distill_warmup = 5
distill_ramp = 30

lambda_kd = 0.2
T_kd = 3.0

aux_warmup = 10
aux_ramp   = 30

modality_dropout_p = 0.0
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

_final_test_rmse_for_logname = None
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

    if _final_test_rmse_for_logname is None:
        return str(_tmp_log_path)

    final_log_path = train_log_dir / (
        f"{dataset_name}_seed{SEED}_split{SPLIT_SEED}_bs{BATCH_SIZE}_RMSE{_final_test_rmse_for_logname:.4f}.log"
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

loader_gen = torch.Generator()
loader_gen.manual_seed(SEED)

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

def apply_single_modality_mask(graph_feat, fp_feat, smiles_feat, mode="all"):
    if mode == "graph":
        return graph_feat, torch.zeros_like(fp_feat), torch.zeros_like(smiles_feat)
    if mode == "fp":
        return torch.zeros_like(graph_feat), fp_feat, torch.zeros_like(smiles_feat)
    if mode == "seq":
        return torch.zeros_like(graph_feat), torch.zeros_like(fp_feat), smiles_feat
    return graph_feat, fp_feat, smiles_feat

def linear_ramp(epoch, max_w, warmup=10, ramp=20):
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(max_w)
    t = (epoch - warmup) / float(ramp)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return float(max_w) * t

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

split_strategy = "random"
split_path = split_dir / f"{dataset_name}_split_{split_strategy}_seed{SPLIT_SEED}.npz"
split_path = str(split_path)

if os.path.exists(split_path):
    arr = np.load(split_path)
    train_idx, val_idx, test_idx = arr["train"].tolist(), arr["val"].tolist(), arr["test"].tolist()
    print(f"Loaded random split from {split_path}")
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

feature_dir_candidates = [
    ROOT / "data" / DATASET_KEY_UPPER / "features",
    ROOT / "data" / dataset_name / "features",
    ROOT / "data" / dataset_name.upper() / "features",
    ROOT / "data" / dataset_name.lower() / "features",
]
feature_dir = None
for cand_dir in feature_dir_candidates:
    if cand_dir.exists():
        feature_dir = cand_dir
        break
if feature_dir is None:
    feature_dir = feature_dir_candidates[0]

fp_candidates = [
    feature_dir / "ecfp.npy",
]
fp_path = None
for cand in fp_candidates:
    if cand.exists():
        fp_path = cand
        break
if fp_path is None:
    raise FileNotFoundError(
        f"No fingerprint file found under {feature_dir}. Tried: {[p.name for p in fp_candidates]}"
    )

fp_np = np.load(str(fp_path))
fp_tensor = torch.from_numpy(fp_np).float()

FP_DIM = fp_tensor.shape[1]

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

model = UnifiedMoleculeEncoder(graph_hidden_dim=128).to(device)

graph_projection = nn.Identity().to(device)
print(f"[Architecture] Graph feature dimension: {model.output_dim} (no projection)")

tmp_batch, tmp_ids, tmp_mask = next(iter(train_loader))
tmp_batch = tmp_batch.to(device)

ea = tmp_batch.edge_attr
if ea is not None:
    print("[CHECK] edge_attr min/max:", ea.min().item(), ea.max().item())
    print("[CHECK] first row:", ea[0])

with torch.no_grad():
    tmp_g = model(tmp_batch) if True else model(tmp_batch, None)
graph_dim = tmp_g.shape[1]
print("[Init] inferred graph_dim =", graph_dim)

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

fusion_gate = nn.Sequential(
    nn.Linear(fusion_in_dim, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 3)
).to(device)

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

optimizer = torch.optim.AdamW(
    [p for p in (list(model.parameters()) +
                 list(graph_projection.parameters()) +
                 list(fp_proj.parameters()) +
                 list(smiles_encoder.parameters()) +
                 list(final_fusion_layer.parameters()) +
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

reg_loss_none = nn.MSELoss(reduction="none")

best_val_rmse = float("inf")
save_dir = ROOT / "ckpt" / "runs"
save_dir.mkdir(parents=True, exist_ok=True)
best_model_path = save_dir / f"{dataset_name}_best_seed{SEED}_split{SPLIT_SEED}.pth"
best_model_path = str(best_model_path)

train_rmse_log, val_rmse_log = [], []

def regression_metrics_from_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    assert y_true.shape == y_pred.shape

    rmses, maes, r2s = [], [], []
    T = y_true.shape[1]
    for k in range(T):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        m = ~np.isnan(yt)
        if int(m.sum()) == 0:
            continue
        yt = yt[m]
        yp = yp[m]
        mse = float(np.mean((yp - yt) ** 2))
        rmse = float(np.sqrt(max(mse, 0.0)))
        mae = float(np.mean(np.abs(yp - yt)))
        ss_res = float(np.sum((yp - yt) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

    macro_rmse = float(np.mean(rmses)) if len(rmses) else float("nan")
    macro_mae = float(np.mean(maes)) if len(maes) else float("nan")
    macro_r2 = float(np.nanmean(r2s)) if len(r2s) else float("nan")
    return macro_rmse, macro_mae, macro_r2

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
    all_labels, all_preds = [], []

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

            graph_feat, fp_feat, smiles_feat = apply_single_modality_mask(
                graph_feat, fp_feat, smiles_feat, mode=SINGLE_MODALITY
            )

            if train and SINGLE_MODALITY == "all":
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
                    t_pred = final_fusion_layer(teacher_in)
                if was_training:
                    final_fusion_layer.train()

                kd_loss = reg_loss_none(out, t_pred).mean()

            target = data.y.float()
            if target.dim() == 1:
                target = target.view(-1, output_dim)
            elif target.shape[-1] != output_dim:
                target = target.view(-1, output_dim)

            target_use = torch.nan_to_num(target, nan=0.0)
            loss_raw = reg_loss_none(out, target_use)
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
            all_preds.append(out.detach().cpu())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()

    macro_rmse, macro_mae, macro_r2 = regression_metrics_from_numpy(all_labels, all_preds)

    avg_loss = sum_loss / max(1, len(loader))
    avg_distill = total_distill / max(1, n_batches)
    return avg_loss, macro_rmse, macro_mae, macro_r2, avg_distill

for epoch in range(1, EPOCHS + 1):
    # warmup
    if epoch <= warmup_epochs:
        for pg in optimizer.param_groups:
            pg["lr"] = INITIAL_LR * (epoch / warmup_epochs)

    train_loss, train_rmse, train_mae, train_r2, train_distill = one_pass(train_loader, train=True, epoch=epoch)
    val_loss, val_rmse, val_mae, val_r2, _ = one_pass(val_loader, train=False, epoch=epoch)

    if epoch > warmup_epochs:
        scheduler.step()

    train_rmse_log.append(train_rmse)
    val_rmse_log.append(val_rmse)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f} | Train R2: {train_r2:.4f} | Distill: {train_distill:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save({
            "model": model.state_dict(),
            "fp_proj": fp_proj.state_dict(),
            "smiles_encoder": smiles_encoder.state_dict(),
            "fp_align_head": fp_align_head.state_dict(),
            "sm_align_head": sm_align_head.state_dict(),
            "teacher_proj": teacher_proj.state_dict(),
            "final_fusion_layer": final_fusion_layer.state_dict(),
            "seed": SEED,
            "val_rmse": float(val_rmse),
        }, best_model_path)
        print(f"   Saved BEST (Val RMSE = {val_rmse:.4f}) -> {best_model_path}")

ckpt = torch.load(best_model_path, map_location=device)

model.load_state_dict(ckpt["model"])
final_fusion_layer.load_state_dict(ckpt["final_fusion_layer"])
fp_proj.load_state_dict(ckpt["fp_proj"])
smiles_encoder.load_state_dict(ckpt["smiles_encoder"])

test_loss, test_rmse, test_mae, test_r2, _ = one_pass(test_loader, train=False, epoch=0)
print(f"\n=== Final Test (Last Epoch Weights) ===")
print(f"Loss: {test_loss:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")

_final_test_rmse_for_logname = float(test_rmse)

plt.plot(range(1, len(train_rmse_log) + 1), train_rmse_log, marker="o", label="Train RMSE")
plt.plot(range(1, len(val_rmse_log) + 1), val_rmse_log, marker="s", label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"{dataset_name} - RMSE Curves (seed={SEED})")
plt.grid(True)
plt.legend()
plot_dir = ROOT / "logs" / "plot_logs"
plot_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_dir / f"{dataset_name}_rmse_curves_seed{SEED}.png", dpi=300, bbox_inches="tight")
plt.show()

final_log_path = _finalize_log_file()
print(f"\n[Train Log Saved] {final_log_path}")
