"""LightGCN 模型适配器——PyTorch 图卷积网络。

将区域坐标转为空间邻接图，通过 LightGCN 传播嵌入进行回归预测。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from taxipredict.models.base import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _LightGCN(nn.Module):
    def __init__(self, n_nodes, feat_dim, embed_dim, layer_num):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        self.fc_in = nn.Linear(feat_dim, embed_dim * 2)
        self.fc_out = nn.Linear(embed_dim * 2 + embed_dim, 1)
        self.layer_num = layer_num

    def forward(self, node_idx, features, norm_adj):
        emb = self.embedding(node_idx)
        h = emb
        for _ in range(self.layer_num):
            h = torch.sparse.mm(norm_adj, h)
        feat = self.fc_in(features)
        combined = torch.cat([feat, emb + h], dim=1)
        return self.fc_out(combined).squeeze()


def _build_graph(location_coords, k_neighbors=8):
    coords = np.array([[c[0], c[1]] for c in location_coords])
    n = len(coords)
    knn = NearestNeighbors(n_neighbors=min(k_neighbors, n), metric="euclidean")
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)

    edges = []
    for i in range(n):
        for j in indices[i]:
            if i != j:
                edges.append([i, j])
    edge_idx = torch.tensor(edges, dtype=torch.long).t()
    values = torch.ones(edge_idx.size(1))
    adj = torch.sparse_coo_tensor(edge_idx, values, (n, n))
    adj = adj.coalesce()
    d = torch.sparse.sum(adj, dim=1).to_dense().pow(-0.5)
    norm_adj = torch.sparse.mm(
        torch.sparse.mm(torch.diag(d), adj), torch.diag(d)
    ).coalesce()
    return norm_adj


class LightGCNModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._model = None
        self._feature_cols = []
        self._target_col = "pickups"
        self._norm_adj = None
        self._node_map = {}

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，LightGCN 需要 torch>=2.1")

        gcn_cfg = self.cfg.get("models", {}).get("lightgcn", {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._feature_cols = self._select_features(train_df, self._target_col)
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        # 构建节点映射
        if "geohash" in full_df.columns:
            all_regions = full_df["geohash"].unique()
        else:
            all_regions = sorted(full_df[self._feature_cols[0]].unique())
        self._node_map = {r: i for i, r in enumerate(all_regions)}
        n_nodes = len(all_regions)

        # 空间图（批量获取坐标，避免逐行扫描）
        if "latitude" in full_df.columns and "longitude" in full_df.columns:
            region_coords = full_df.groupby("geohash")[["latitude", "longitude"]].first()
        else:
            region_coords = pd.DataFrame(
                {"latitude": 0.0, "longitude": 0.0},
                index=pd.Index(self._node_map.keys(), name="geohash"),
            )
        loc_coords = [
            (float(region_coords.loc[r, "latitude"]) if r in region_coords.index else 0.0,
             float(region_coords.loc[r, "longitude"]) if r in region_coords.index else 0.0)
            for r in self._node_map
        ]
        self._norm_adj = _build_graph(loc_coords, gcn_cfg.get("k_neighbors", 8)).to(device)

        # 准备数据
        def _make_tensor(df):
            feats = torch.tensor(
                df[self._feature_cols].fillna(0).values, dtype=torch.float32
            )
            node_idx = torch.tensor(
                [self._node_map.get(g, 0) for g in df.get("geohash", df.index)],
                dtype=torch.long,
            )
            targets = torch.tensor(df[self._target_col].values, dtype=torch.float32)
            return node_idx, feats, targets

        train_nodes, train_feats, train_targets = _make_tensor(train_df)
        test_nodes, test_feats, test_targets = _make_tensor(test_df)

        scaler = StandardScaler()
        train_feats_np = scaler.fit_transform(train_feats.numpy())
        test_feats_np = scaler.transform(test_feats.numpy())
        train_feats = torch.tensor(train_feats_np, dtype=torch.float32)
        test_feats = torch.tensor(test_feats_np, dtype=torch.float32)

        bs = gcn_cfg.get("batch_size", 1024)
        train_loader = DataLoader(
            TensorDataset(train_nodes, train_feats, train_targets),
            batch_size=bs, shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(test_nodes, test_feats, test_targets),
            batch_size=bs, shuffle=False,
        )

        model = _LightGCN(
            n_nodes=n_nodes,
            feat_dim=len(self._feature_cols),
            embed_dim=gcn_cfg.get("embed_dim", 64),
            layer_num=gcn_cfg.get("layer_num", 3),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=gcn_cfg.get("learning_rate", 0.001))

        for epoch in range(gcn_cfg.get("epochs", 50)):
            model.train()
            for n, f, t in train_loader:
                n, f, t = n.to(device), f.to(device), t.to(device)
                pred = model(n, f, self._norm_adj)
                loss = nn.MSELoss()(pred, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  GCN epoch {epoch+1}")

        self._model = model

        model.eval()
        with torch.no_grad():
            train_pred = model(train_nodes.to(device), train_feats.to(device), self._norm_adj).cpu().numpy()
            test_pred = model(test_nodes.to(device), test_feats.to(device), self._norm_adj).cpu().numpy()

        train_targets_np = train_targets.numpy()
        test_targets_np = test_targets.numpy()

        self._pred_df = test_df[[self._target_col, "datetime", "geohash"]].copy()
        self._pred_df["pred_pickups"] = np.maximum(test_pred, 0)
        self._pred_df["error"] = self._pred_df["pred_pickups"] - self._pred_df[self._target_col]
        self._pred_df["abs_error"] = self._pred_df["error"].abs()
        self._processed_test = test_df

        return {
            "train_mae": float(mean_absolute_error(train_targets_np, np.maximum(train_pred, 0))),
            "train_rmse": float(np.sqrt(mean_squared_error(train_targets_np, np.maximum(train_pred, 0)))),
            "train_r2": float(r2_score(train_targets_np, np.maximum(train_pred, 0))),
            "test_mae": float(mean_absolute_error(test_targets_np, np.maximum(test_pred, 0))),
            "test_rmse": float(np.sqrt(mean_squared_error(test_targets_np, np.maximum(test_pred, 0)))),
            "test_r2": float(r2_score(test_targets_np, np.maximum(test_pred, 0))),
            "feature_count": len(self._feature_cols),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("LightGCN predict 需要图结构，请使用 train() 评估")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_state": self._model.state_dict() if self._model is not None else None,
            "features": self._feature_cols,
            "node_map": self._node_map,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGCNModel":
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj._feature_cols = data["features"]
        obj._node_map = data["node_map"]
        return obj
