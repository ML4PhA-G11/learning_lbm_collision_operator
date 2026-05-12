"""PyTorch port of run-all-tensorflow.py.

End-to-end pipeline (training-set generation, NN training, LBM
simulation with ML collision step) using PyTorch instead of TensorFlow/
Keras. The numerical setup (sample sizes, hyper-parameters, network
shape, loss) mirrors the TF version so wall-time comparisons are fair.

All artifacts go to ./artifacts-run-all-pytorch/.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from utils import LB_stencil
from utils_pytorch import LBMSymmetryModel, rmsre


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts-run-all-pytorch"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH    = ARTIFACTS_DIR / "example_dataset.npz"
WEIGHTS_PATH    = ARTIFACTS_DIR / "weights.pt"
MODEL_PATH      = ARTIFACTS_DIR / "example_network.pt"
LOSS_PLOT_PATH  = ARTIFACTS_DIR / "training_loss.png"
DECAY_PLOT_PATH = ARTIFACTS_DIR / "velocity_decay.png"
FIELDS_PLOT_DIR = ARTIFACTS_DIR / "velocity_fields"
FIELDS_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# float64 throughout, to match the TF script's K.set_floatx('float64').
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[pt] device: {device}")
if device.type == "cuda":
    print(f"[pt] GPU: {torch.cuda.get_device_name(0)}")


############################################
# Training-data generation helpers (numpy, identical to the TF version)
############################################


def compute_rho_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=0.0, u_abs_max=0.01):
    rho   = np.random.uniform(rho_min, rho_max, size=num_samples)
    u_abs = np.random.uniform(u_abs_min, u_abs_max, size=num_samples)
    theta = np.random.uniform(0, 2 * np.pi, size=num_samples)
    ux = u_abs * np.cos(theta)
    uy = u_abs * np.sin(theta)
    u  = np.array([ux, uy]).transpose()
    return rho, u


def compute_f_rand(num_samples, sigma_min, sigma_max, c):
    Q  = 9
    K0 = 1 / 9.0
    K1 = 1 / 6.0

    f_rand = np.zeros((num_samples, Q))

    if sigma_min == sigma_max:
        sigma = sigma_min * np.ones(num_samples)
    else:
        sigma = np.random.uniform(sigma_min, sigma_max, size=num_samples)

    for i in range(num_samples):
        f_rand[i, :] = np.random.normal(0, sigma[i], size=(1, Q))
        rho_hat = np.sum(f_rand[i, :])
        ux_hat  = np.sum(f_rand[i, :] * c[:, 0])
        uy_hat  = np.sum(f_rand[i, :] * c[:, 1])
        f_rand[i, :] = (
            f_rand[i, :] - K0 * rho_hat - K1 * ux_hat * c[:, 0] - K1 * uy_hat * c[:, 1]
        )
    return f_rand


def compute_f_pre_f_post(f_eq, f_neq, tau_min=1, tau_max=1):
    tau    = np.random.uniform(tau_min, tau_max, size=f_eq.shape[0])
    f_pre  = f_eq + f_neq
    f_post = f_pre + 1 / tau[:, None] * (f_eq - f_pre)
    return tau, f_pre, f_post


def delete_negative_samples(n_samples, f_eq, f_pre, f_post):
    i_neg_f_eq   = np.where(np.sum(f_eq   < 0, axis=1) > 0)[0]
    i_neg_f_pre  = np.where(np.sum(f_pre  < 0, axis=1) > 0)[0]
    i_neg_f_post = np.where(np.sum(f_post < 0, axis=1) > 0)[0]
    i_neg_f = np.concatenate((i_neg_f_pre, i_neg_f_post, i_neg_f_eq))
    f_eq   = np.delete(np.copy(f_eq),   i_neg_f, 0)
    f_pre  = np.delete(np.copy(f_pre),  i_neg_f, 0)
    f_post = np.delete(np.copy(f_post), i_neg_f, 0)
    return f_eq, f_pre, f_post


def load_data(fname):
    data = np.load(fname, allow_pickle=True)
    return data["f_eq"], data["f_pre"], data["f_post"]


def sol(t, L, F0, nu):
    return F0 * np.exp(-2 * nu * t / (L / (2 * np.pi)) ** 2)


#########################################################
# 1. Create training data
#########################################################
n_samples  = 100_000
u_abs_min  = 1e-15
u_abs_max  = 0.01
sigma_min  = 1e-15
sigma_max  = 5e-4

Q = 9
c, w, cs2, compute_feq = LB_stencil()

fPreLst  = np.empty((n_samples, Q))
fPostLst = np.empty((n_samples, Q))
fEqLst   = np.empty((n_samples, Q))

idx = 0
while idx < n_samples:
    rho, u = compute_rho_u(n_samples)
    rho = rho[:, np.newaxis]
    ux  = u[:, 0][:, np.newaxis]
    uy  = u[:, 1][:, np.newaxis]

    f_eq = np.zeros((n_samples, 1, Q))
    f_eq = compute_feq(f_eq, rho, ux, uy, c, w)[:, 0, :]

    f_neq = compute_f_rand(n_samples, sigma_min, sigma_max, c)
    _, f_pre, f_post = compute_f_pre_f_post(f_eq, f_neq)
    f_eq, f_pre, f_post = delete_negative_samples(n_samples, f_eq, f_pre, f_post)

    non_negatives = f_pre.shape[0]
    idx1        = min(idx + non_negatives, n_samples)
    to_be_added = min(n_samples - idx, non_negatives)

    fPreLst[idx:idx1]  = f_pre[:to_be_added]
    fPostLst[idx:idx1] = f_post[:to_be_added]
    fEqLst[idx:idx1]   = f_eq[:to_be_added]

    idx += non_negatives

np.savez(DATASET_PATH, f_pre=fPreLst, f_post=fPostLst, f_eq=fEqLst)


#########################################################
# 2. Training
#########################################################
feq, fpre, fpost = load_data(DATASET_PATH)

# normalize on density
feq   = feq   / np.sum(feq,   axis=1)[:, np.newaxis]
fpre  = fpre  / np.sum(fpre,  axis=1)[:, np.newaxis]
fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]

fpre_train, fpre_test, fpost_train, fpost_test = train_test_split(
    fpre, fpost, test_size=0.3, shuffle=True
)

batch_size = 32
n_epochs   = 200
patience   = 50
verbose    = 1

# Model — softmax activation throughout, matching the TF create_model call
# that passed ll_activation="softmax" for both activation slots.
model = LBMSymmetryModel(
    Q=Q,
    n_hidden_layers=2,
    n_per_layer=50,
    activation="softmax",
    bias=False,
).to(device)

optimizer = torch.optim.Adam(model.parameters())

# DataLoaders. num_workers=0 keeps numba (used inside utils.LB_stencil) and
# torch DataLoader's multiprocessing from colliding.
train_ds = TensorDataset(
    torch.from_numpy(fpre_train), torch.from_numpy(fpost_train)
)
val_ds = TensorDataset(
    torch.from_numpy(fpre_test),  torch.from_numpy(fpost_test)
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

best_val_loss    = float("inf")
best_state_dict  = None
patience_counter = 0
history = {"loss": [], "val_loss": []}

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")

    model.train()
    running = 0.0
    seen    = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb)
        loss = rmsre(yb, pred)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        seen    += xb.size(0)
    train_loss = running / seen

    model.eval()
    running_v = 0.0
    seen_v    = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            v = rmsre(yb, pred)
            running_v += v.item() * xb.size(0)
            seen_v    += xb.size(0)
    val_loss = running_v / seen_v

    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if verbose:
        print(
            f"           loss: {train_loss:.4e} - val_loss: {val_loss:.4e}",
            flush=True,
        )

    # ModelCheckpoint (save_best_only=True) + EarlyStopping (restore best)
    if val_loss < best_val_loss:
        best_val_loss   = val_loss
        best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(best_state_dict, WEIGHTS_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            break

# restore best weights and save the final model
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
torch.save(model.state_dict(), MODEL_PATH)

# final evaluation
model.eval()
running_v = 0.0
seen_v    = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        v = rmsre(yb, model(xb))
        running_v += v.item() * xb.size(0)
        seen_v    += xb.size(0)
print(f"Final val rmsre: {running_v / seen_v:.4e}")

plt.figure()
plt.semilogy(history["loss"],     lw=3, label="Training")
plt.semilogy(history["val_loss"], lw=3, label="Validation")
plt.legend(loc="best", frameon=False)
plt.savefig(LOSS_PLOT_PATH, dpi=120, bbox_inches="tight")
plt.close()


#########################################################
# 3. Simulation with ML collision step
#########################################################
# Reload to mirror the TF version's "load from disk" step.
model = LBMSymmetryModel(
    Q=Q, n_hidden_layers=2, n_per_layer=50, activation="softmax", bias=False
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("[pt] model loaded for simulation:")
print(model)

nx     = 32
ny     = 32
niter  = 1000
dumpit = 100
tau    = 1.0
u0     = 0.01

ndumps   = int(niter // dumpit)
dumpfile = np.zeros((ndumps * nx * ny, 4))


def data_collector(dumpfile, t, ux, uy, rho):
    it   = t // dumpit
    idx0 =  it       * (nx * ny)
    idx1 = (it + 1) * (nx * ny)
    dumpfile[idx0:idx1, 0] = t
    dumpfile[idx0:idx1, 1] = rho.reshape(nx * ny)
    dumpfile[idx0:idx1, 2] = ux.reshape( nx * ny)
    dumpfile[idx0:idx1, 3] = uy.reshape( nx * ny)


a = b = 1.0
ix, iy = np.meshgrid(range(nx), range(ny), indexing="ij")
x = 2.0 * np.pi * (ix / nx)
y = 2.0 * np.pi * (iy / ny)
ux =  1.0 * u0 * np.sin(a * x) * np.cos(b * y)
uy = -1.0 * u0 * np.cos(a * x) * np.sin(b * y)
rho = np.ones((nx, ny))

c, w, cs2, compute_feq = LB_stencil()

feq = np.zeros((nx, ny, Q))
feq = compute_feq(feq, rho, ux, uy, c, w)
f1 = np.copy(feq)
f2 = np.copy(feq)

data_collector(dumpfile, 0, ux, uy, rho)
m_initial = np.sum(f1.flatten())

for t in range(1, niter):
    # streaming
    for ip in range(Q):
        f1[:, :, ip] = np.roll(np.roll(f2[:, :, ip], c[ip, 0], axis=0), c[ip, 1], axis=1)

    rho = np.sum(f1, axis=2)
    ux = (1.0 / rho) * np.einsum("ijk,k", f1, c[:, 0])
    uy = (1.0 / rho) * np.einsum("ijk,k", f1, c[:, 1])

    fpre = f1.reshape((nx * ny, Q))
    norm = np.sum(fpre, axis=1)[:, np.newaxis]
    fpre = fpre / norm

    with torch.no_grad():
        fpre_t = torch.from_numpy(fpre).to(device)
        f2_t   = model(fpre_t)
        f2     = f2_t.cpu().numpy()

    f2 = norm * f2
    f2 = f2.reshape((nx, ny, Q))

    if (t % dumpit) == 0:
        data_collector(dumpfile, t, ux, uy, rho)

m_final = np.sum(f2.flatten())
print("Sim ended. Mass err:", np.abs(m_initial - m_final) / m_initial)


#########################################################
# 4. Plots
#########################################################
W = 3.46 * 3
H = 2.14 * 3

fig = plt.figure(figsize=(W, H))
ax  = fig.add_subplot(111)
tLst = np.arange(0, niter, dumpit)
for i, t in enumerate(tLst):
    ux_ = dumpfile[dumpfile[:, 0] == t, 2]
    uy_ = dumpfile[dumpfile[:, 0] == t, 3]
    Ft = np.average((ux_ ** 2 + uy_ ** 2) ** 0.5)
    if i == 0:
        F0 = Ft
        ax.semilogy(t, Ft, "ob", label="lbm")
    else:
        ax.semilogy(t, Ft, "ob")

nu = (tau - 0.5) * cs2
ax.semilogy(tLst, sol(tLst, nx, F0, nu),
            linewidth=2.0, linestyle="--", color="r", label="analytic")
ax.set_xlabel(r"$t~\rm{[L.U.]}$", fontsize=16)
ax.set_ylabel(r"$\langle |u| \rangle$", fontsize=16, rotation=90, labelpad=0)
ax.legend(loc="best", frameon=False, prop={"size": 16})
ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=14)
fig.savefig(DECAY_PLOT_PATH, dpi=120, bbox_inches="tight")
plt.close(fig)

X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
for i, t in enumerate(tLst):
    fig = plt.figure(figsize=(W, H))
    ax  = fig.add_subplot(111)
    ux_ = dumpfile[dumpfile[:, 0] == t, 2].reshape((nx, ny))
    uy_ = dumpfile[dumpfile[:, 0] == t, 3].reshape((nx, ny))
    u_mag = (ux_ ** 2 + uy_ ** 2) ** 0.5
    im = ax.imshow(u_mag)
    ax.streamplot(X, Y, ux_, uy_, density=0.5, color="w")
    fig.colorbar(im, ax=ax, orientation="vertical", pad=0, shrink=0.69)
    ax.set_title(f"Iteration {t}", size=16)
    fig.savefig(FIELDS_PLOT_DIR / f"velocity_field_t{int(t):05d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
