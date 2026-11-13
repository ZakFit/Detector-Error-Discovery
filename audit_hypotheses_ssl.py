"""Adapt DINOv2 final blocks to masked cafe crops without detector labels."""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from audit_hypotheses_features import CACHE, backbone
from audit_hypotheses_scene import DAYS
from audit_hypotheses_visual import critic, critic_data, train_masks
from audit_hypotheses_features import metadata
from features import CLASSES


class LowRankLinear(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__()
        self.linear = linear
        self.down = nn.Linear(linear.in_features, rank, bias=False, device="cuda")
        self.up = nn.Linear(rank, linear.out_features, bias=False, device="cuda")
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.linear(x) + self.up(self.down(x))


def adapters(blocks):
    blocks = copy.deepcopy(blocks).requires_grad_(False)
    for block in blocks:
        block.attn.qkv = LowRankLinear(block.attn.qkv)
        block.attn.proj = LowRankLinear(block.attn.proj)
        block.mlp.fc1 = LowRankLinear(block.mlp.fc1)
        block.mlp.fc2 = LowRankLinear(block.mlp.fc2)
    return blocks


def masked_fit(blocks, norm, pool, train):
    early, teacher, mask = pool
    student = adapters(blocks).eval()
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.00055)
    rng = torch.Generator().manual_seed(53)
    indices = torch.from_numpy(np.flatnonzero(train))
    for epoch in range(6):
        order = indices[torch.randperm(len(indices), generator=rng)]
        losses = []
        for batch in order.split(24):
            x, reference, hidden = early[batch].cuda(), teacher[batch].cuda(), mask[batch].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = norm(student(x))[:, -100:]
                distances = 1 - F.cosine_similarity(pred.float(), reference.float(), dim=-1)
                loss = distances[hidden].mean() + .25 * distances[~hidden].mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        print("ssl", epoch + 1, "examples", len(indices), "loss", round(float(torch.stack(losses).mean()), 4), flush=True)
    return student


@torch.no_grad()
def transform(blocks, norm, tokens):
    out, vectors = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in tokens.split(48):
            x = norm(blocks(batch.cuda())).half()
            out.append(x)
            vectors.append(torch.cat([x[:, 0], x[:, -100:].mean(1)], 1).float().cpu())
    return torch.cat(out), normalize(torch.cat(vectors).numpy())


def linear_probe(rows, features, train, predict):
    fit = train & (rows.part == "train")
    classes = np.eye(len(CLASSES), dtype=np.float32)[rows.class_name.map(CLASSES.index).to_numpy()]
    x = np.concatenate([features, .25 * classes], axis=1)
    model = LogisticRegression(C=10, max_iter=300)
    model.fit(x[fit], rows.loc[fit, "error"])
    return model.decision_function(x[predict])


def main():
    import sys
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(712)
    rows = pd.read_csv(CACHE / "rows.csv")
    _, gt, _ = metadata()
    rows = rows[rows.class_name.isin(CLASSES) & ~rows.occluded & (rows.view == 0)].copy()
    tokens = torch.load(CACHE / "context.pt", weights_only=True)[rows.index]
    rows.reset_index(drop=True, inplace=True)
    machine = pd.read_csv(CACHE / "pool.csv", dtype={"day": str})
    pool = torch.load(CACHE / "pool_masked.pt", weights_only=True)
    assert len(machine) == len(pool[0])
    model = backbone()
    blocks = model.blocks[-2:].requires_grad_(False).eval()
    norm = model.norm.requires_grad_(False).eval()
    del model
    torch.cuda.empty_cache()
    data = critic_data(rows, gt)
    frozen, frozen_vec = transform(blocks, norm, tokens)
    result = []
    path = CACHE / "ssl.csv"
    if sys.argv[-1] == "test" and path.exists():
        result.append(pd.read_csv(path).query("fold != 'test'"))
    days = ["test"] if sys.argv[-1] == "test" else DAYS
    for day in days:
        fit, predict = train_masks(rows, day)
        allowed = machine.day.isin(DAYS) if day == "test" else (machine.day != day)
        student = masked_fit(blocks, norm, pool, allowed.to_numpy())
        adapted, adapted_vec = transform(student, norm, tokens)
        values = {}
        for tag, representations, vector in [("frozen", frozen, frozen_vec), ("adapted", adapted, adapted_vec)]:
            values[f"ssl_{tag}_probe"] = linear_probe(rows, vector, fit, predict)
            values[f"ssl_{tag}_critic"] = np.mean([critic(rows, representations, data, fit, predict, seed, True)
                                                   for seed in (117, 312, 614)], axis=0)
        result.append(rows.loc[predict, ["id"]].assign(fold=day, **values))
        pd.concat(result, ignore_index=True).to_csv(path, index=False)
        del student, adapted
        print("SSL finished", day, flush=True)


if __name__ == "__main__":
    main()
