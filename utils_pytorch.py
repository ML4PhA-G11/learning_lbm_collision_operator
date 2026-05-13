"""PyTorch port of the TensorFlow custom layers in utils.py.

Names and forward semantics match the TF originals so that
run-all-pytorch.py is a 1:1 port of run-all-tensorflow.py.
"""

import torch
import torch.nn as nn


def rmsre(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Root-mean-square relative error, averaged across batch.

    Mirrors `utils.rmsre` (keras ops version).
    """
    per_sample = torch.sqrt(
        torch.mean(torch.square((y_true - y_pred) / (y_true + eps)), dim=-1)
    )
    return per_sample.mean()


def LBrot90(f: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Rotate the cardinal and diagonal blocks of the D2Q9 stencil by k.

    The D2Q9 layout is [center, cardinals(1..4), diagonals(5..8)]. The
    center stays, the cardinals and diagonals are independently rolled.
    """
    return torch.cat(
        [
            f[:, 0:1],
            torch.roll(f[:, 1:5], shifts=int(k), dims=-1),
            torch.roll(f[:, 5:9], shifts=int(k), dims=-1),
        ],
        dim=-1,
    )


_MIRROR_INDEX = [0, 1, 4, 3, 2, 8, 7, 6, 5]


def LBmirror(f: torch.Tensor) -> torch.Tensor:
    """Mirror across the y-axis: swap (north,south) and (NE,SE)+(NW,SW)."""
    return torch.stack(
        [
            f[:, 0], f[:, 1], f[:, 4], f[:, 3], f[:, 2],
            f[:, 8], f[:, 7], f[:, 6], f[:, 5],
        ],
        dim=-1,
    )


class D4Symmetry(nn.Module):
    """Produce the 8-element D4 orbit (rotations + mirrored rotations)."""

    def forward(self, x: torch.Tensor):  # noqa: D401
        return [
            x,
            LBrot90(x, k=1),
            LBrot90(x, k=2),
            LBrot90(x, k=3),
            LBmirror(x),
            LBmirror(LBrot90(x, k=1)),
            LBmirror(LBrot90(x, k=2)),
            LBmirror(LBrot90(x, k=3)),
        ]


class D4AntiSymmetry(nn.Module):
    """Inverse of D4Symmetry: un-rotate / un-mirror back to the original frame."""

    def forward(self, x):
        return [
            x[0],
            LBrot90(x[1], k=-1),
            LBrot90(x[2], k=-2),
            LBrot90(x[3], k=-3),
            LBmirror(x[4]),
            LBrot90(LBmirror(x[5]), k=-1),
            LBrot90(LBmirror(x[6]), k=-2),
            LBrot90(LBmirror(x[7]), k=-3),
        ]


class AlgReconstruction(nn.Module):
    """Project the network output back onto the mass+momentum-conserving
    subspace by algebraically reconstructing populations 2, 5, 8.
    """

    def forward(self, fpre: torch.Tensor, fpred: torch.Tensor) -> torch.Tensor:
        df = fpred - fpre

        df2 = -(df[:, 0] + 2 * df[:, 3] + df[:, 4] + 2 * df[:, 6] + 2 * df[:, 7])
        df5 = 0.5 * (
            df[:, 0]
            + 3 * df[:, 3]
            + 2 * df[:, 4]
            + 2 * df[:, 6]
            + 4 * df[:, 7]
            - df[:, 1]
        )
        df8 = -0.5 * (
            df[:, 0] + df[:, 1] + df[:, 3] + 2 * df[:, 4] + 2 * df[:, 7]
        )

        df_new = torch.stack(
            [df[:, 0], df[:, 1], df2, df[:, 3], df[:, 4], df5, df[:, 6], df[:, 7], df8],
            dim=-1,
        )
        return fpre + df_new


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "softmax":
        return nn.Softmax(dim=-1)
    if name == "relu":
        return nn.ReLU()
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"unsupported activation: {name}")


class SeqMLP(nn.Module):
    """Inner MLP that maps a 9-vector to a 9-vector. Mirrors the keras
    `sequential_model` used in the TF script (Q -> hidden -> ... -> Q).

    Matches the TF code path where `create_model` passes `ll_activation`
    for BOTH the hidden and output activations, so every layer uses the
    same activation (softmax in the default invocation).
    """

    def __init__(
        self,
        Q: int = 9,
        n_hidden_layers: int = 2,
        n_per_layer: int = 50,
        activation: str = "softmax",
        bias: bool = False,
    ):
        super().__init__()
        modules = [
            nn.Linear(Q, n_per_layer, bias=bias),
            _make_activation(activation),
        ]
        for _ in range(n_hidden_layers):
            modules += [
                nn.Linear(n_per_layer, n_per_layer, bias=bias),
                _make_activation(activation),
            ]
        modules += [
            nn.Linear(n_per_layer, Q, bias=bias),
            _make_activation(activation),
        ]
        self.net = nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LBMSymmetryModel(nn.Module):
    """Top-level model assembling the symmetry-averaging pipeline."""

    def __init__(
        self,
        Q: int = 9,
        n_hidden_layers: int = 2,
        n_per_layer: int = 50,
        activation: str = "softmax",
        bias: bool = False,
    ):
        super().__init__()
        self.seq = SeqMLP(
            Q=Q,
            n_hidden_layers=n_hidden_layers,
            n_per_layer=n_per_layer,
            activation=activation,
            bias=bias,
        )
        self.d4_sym = D4Symmetry()
        self.alg_recon = AlgReconstruction()
        self.d4_anti = D4AntiSymmetry()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_lst = self.d4_sym(x)
        out_lst = [self.seq(xi) for xi in input_lst]
        out_lst = [self.alg_recon(input_lst[k], oi) for k, oi in enumerate(out_lst)]
        out_lst = self.d4_anti(out_lst)
        return torch.stack(out_lst, dim=0).mean(dim=0)
