# LightGCN 分 geohash 地区、区域×小时 pickups 预测
# 将表格数据转为空间图 + 地点-小时二部图，通过 LightGCN 传播嵌入并回归预测

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from taxi_pipeline import (
    REGION_GROUP_COL,
    add_shared_cli_arguments,
    build_prediction_dataframe,
    evaluate_predictions,
    make_result_dir,
    prepare_hourly_dataset,
    save_standard_outputs,
)

warnings.filterwarnings("ignore")

MODEL_NAME = "LightGCN"


@dataclass
class GraphData:
    """表格数据转化后的图结构。"""

    location_ids: list[str]
    location_index: dict[str, int]
    norm_adj: torch.Tensor
    n_locations: int
    n_hours: int
    edge_count: int


def build_location_spatial_graph(
    location_coords: pd.DataFrame,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 geohash 经纬度构建空间 k-NN 无向图，返回 COO 边列表。"""
    coords = location_coords[["latitude", "longitude"]].astype(float).values
    n_nodes = len(coords)
    k = min(k_neighbors + 1, n_nodes)

    nn_model = NearestNeighbors(n_neighbors=k, metric="haversine")
    nn_model.fit(np.radians(coords))

    distances, indices = nn_model.kneighbors(np.radians(coords))
    src_list: list[int] = []
    dst_list: list[int] = []

    for i in range(n_nodes):
        for j, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx == i:
                continue
            if distances[i][j] <= 0:
                continue
            src_list.append(i)
            dst_list.append(int(neighbor_idx))

    if not src_list:
        for i in range(n_nodes):
            src_list.append(i)
            dst_list.append(i)

    return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)


def build_bipartite_interaction_edges(
    df: pd.DataFrame,
    location_index: dict[str, int],
    hour_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """训练集地点-小时交互边：location <-> (hour + hour_offset)。"""
    src_list: list[int] = []
    dst_list: list[int] = []

    for row in df.itertuples(index=False):
        geohash = getattr(row, REGION_GROUP_COL)
        hour = int(getattr(row, "hour"))
        if geohash not in location_index:
            continue
        loc_idx = location_index[geohash]
        hour_node = hour_offset + hour
        src_list.extend([loc_idx, hour_node])
        dst_list.extend([hour_node, loc_idx])

    return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)


def normalize_adjacency(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}。"""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if src.size > 0:
        adj[src, dst] = 1.0

    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0, out=np.zeros_like(degree))
    norm_adj = (
        degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]
    ).astype(np.float32)

    rows, cols = np.nonzero(norm_adj)
    indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    values = torch.from_numpy(norm_adj[rows, cols].astype(np.float32))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_nodes, n_nodes),
        device=device,
    ).coalesce()


def tabular_to_graph(
    train_df: pd.DataFrame,
    all_df: pd.DataFrame,
    k_neighbors: int,
    device: torch.device,
) -> GraphData:
    """将区域×小时表格数据转为 LightGCN 使用的混合图。"""
    location_df = (
        all_df[[REGION_GROUP_COL, "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .groupby(REGION_GROUP_COL, as_index=False)[["latitude", "longitude"]]
        .mean()
        .sort_values(REGION_GROUP_COL)
        .reset_index(drop=True)
    )

    location_ids = location_df[REGION_GROUP_COL].astype(str).tolist()
    location_index = {loc: idx for idx, loc in enumerate(location_ids)}
    n_locations = len(location_ids)
    n_hours = 24
    hour_offset = n_locations
    n_nodes = n_locations + n_hours

    spatial_src, spatial_dst = build_location_spatial_graph(location_df, k_neighbors=k_neighbors)
    bip_src, bip_dst = build_bipartite_interaction_edges(
        train_df,
        location_index=location_index,
        hour_offset=hour_offset,
    )

    src = np.concatenate([spatial_src, bip_src]) if bip_src.size else spatial_src
    dst = np.concatenate([spatial_dst, bip_dst]) if bip_dst.size else spatial_dst
    norm_adj = normalize_adjacency(n_nodes, src, dst, device=device)

    print(
        f"\n图结构：{n_locations} 个地点节点 + {n_hours} 个小时节点 = {n_nodes} 节点，"
        f"{len(src):,} 条边（空间 k-NN + 训练集地点-小时交互）"
    )

    return GraphData(
        location_ids=location_ids,
        location_index=location_index,
        norm_adj=norm_adj,
        n_locations=n_locations,
        n_hours=n_hours,
        edge_count=int(len(src)),
    )


class LightGCNLayerStack(nn.Module):
    """LightGCN 层传播：E^(l+1) = A_norm E^(l)，最终嵌入为各层均值。"""

    def __init__(self, n_nodes: int, emb_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.node_emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

    def forward(self, norm_adj: torch.Tensor) -> torch.Tensor:
        embs = [self.node_emb.weight]
        for _ in range(self.n_layers):
            embs.append(torch.sparse.mm(norm_adj, embs[-1]))
        return torch.stack(embs, dim=0).mean(dim=0)


class LocationHourLightGCN(nn.Module):
    """地点-小时需求预测：LightGCN 图嵌入 + 侧特征回归。"""

    def __init__(
        self,
        graph: GraphData,
        n_side_features: int,
        emb_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.graph = graph
        self.hour_offset = graph.n_locations
        self.gcn = LightGCNLayerStack(
            n_nodes=graph.n_locations + graph.n_hours,
            emb_dim=emb_dim,
            n_layers=n_layers,
        )
        self.side_proj = nn.Linear(n_side_features, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def node_embeddings(self, norm_adj: torch.Tensor) -> torch.Tensor:
        return self.gcn(norm_adj)

    def forward(
        self,
        loc_indices: torch.Tensor,
        hour_indices: torch.Tensor,
        side_features: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        all_emb = self.node_embeddings(norm_adj)
        loc_emb = all_emb[loc_indices]
        hour_emb = all_emb[self.hour_offset + hour_indices]
        side_emb = self.side_proj(side_features)
        fused = torch.cat([loc_emb, hour_emb, side_emb], dim=-1)
        return self.head(fused).squeeze(-1)


def dataframe_to_tensors(
    df: pd.DataFrame,
    graph: GraphData,
    selected_features: list[str],
    target_col: str,
    scaler: StandardScaler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """将 DataFrame 行映射为图节点索引与张量。"""
    valid_mask = df[REGION_GROUP_COL].astype(str).isin(graph.location_index)
    valid_df = df.loc[valid_mask].copy()
    dropped = len(df) - len(valid_df)
    if dropped > 0:
        print(f"  跳过 {dropped:,} 行（geohash 不在图节点中）")

    loc_indices = valid_df[REGION_GROUP_COL].astype(str).map(graph.location_index).astype(int)
    hour_indices = valid_df["hour"].astype(int)
    side_features = scaler.transform(valid_df[selected_features].astype(float))
    targets = valid_df[target_col].astype(float).values

    return (
        torch.tensor(loc_indices.values, dtype=torch.long),
        torch.tensor(hour_indices.values, dtype=torch.long),
        torch.tensor(side_features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        valid_df.reset_index(drop=True),
    )


def train_lightgcn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    target_col: str,
    emb_dim: int,
    n_layers: int,
    k_neighbors: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    device: torch.device,
) -> tuple[LocationHourLightGCN, StandardScaler, GraphData, dict, pd.DataFrame]:
    graph = tabular_to_graph(
        train_df=train_df,
        all_df=pd.concat([train_df, test_df], ignore_index=True),
        k_neighbors=k_neighbors,
        device=device,
    )

    scaler = StandardScaler()
    scaler.fit(train_df[selected_features].astype(float))

    train_tensors = dataframe_to_tensors(
        train_df, graph, selected_features, target_col, scaler
    )
    test_tensors = dataframe_to_tensors(
        test_df, graph, selected_features, target_col, scaler
    )

    train_loc, train_hour, train_side, train_y, _ = train_tensors
    test_loc, test_hour, test_side, test_y, test_eval_df = test_tensors

    if len(train_y) == 0 or len(test_y) == 0:
        raise ValueError("LightGCN 有效样本不足，请检查 geohash 与经纬度字段。")

    model = LocationHourLightGCN(
        graph=graph,
        n_side_features=len(selected_features),
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_loc, train_hour, train_side, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"\n开始训练 {MODEL_NAME}...")
    print(f"训练样本数：{len(train_y):,}")
    print(f"测试样本数：{len(test_y):,}")
    print(f"嵌入维度：{emb_dim}，GCN 层数：{n_layers}，设备：{device}")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        for batch_loc, batch_hour, batch_side, batch_y in train_loader:
            batch_loc = batch_loc.to(device)
            batch_hour = batch_hour.to(device)
            batch_side = batch_side.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_loc, batch_hour, batch_side, graph.norm_adj)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"  Epoch {epoch:3d}/{epochs}  train_mse={avg_loss:.6f}")

    @torch.no_grad()
    def predict_all(
        loc: torch.Tensor,
        hour: torch.Tensor,
        side: torch.Tensor,
    ) -> np.ndarray:
        model.eval()
        preds: list[np.ndarray] = []
        dataset = TensorDataset(loc, hour, side)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_loc, batch_hour, batch_side in loader:
            batch_loc = batch_loc.to(device)
            batch_hour = batch_hour.to(device)
            batch_side = batch_side.to(device)
            batch_pred = model(batch_loc, batch_hour, batch_side, graph.norm_adj)
            preds.append(batch_pred.cpu().numpy())
        return np.concatenate(preds)

    train_pred = np.maximum(predict_all(train_loc, train_hour, train_side), 0.0)
    test_pred = np.maximum(predict_all(test_loc, test_hour, test_side), 0.0)

    metrics: dict = {}
    metrics.update(evaluate_predictions(train_y.numpy(), train_pred, "train"))
    metrics.update(evaluate_predictions(test_y.numpy(), test_pred, "test"))

    pred_df = build_prediction_dataframe(test_eval_df, test_pred, target_col)
    return model, scaler, graph, metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 may14 + jun14 数据训练 LightGCN，并预测六月最后一周 pickups。"
    )
    add_shared_cli_arguments(parser, result_prefix_default="result_lightgcn")
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=64,
        help="LightGCN 嵌入维度，默认 64。",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="LightGCN 传播层数，默认 3。",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="空间图 k-NN 邻居数，默认 8。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数，默认 20。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="批大小，默认 4096。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="学习率，默认 0.001。",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="预测头 Dropout，默认 0.2。",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = make_result_dir(args.result_prefix)
    input_paths = [Path(p) for p in args.inputs]

    print(f"结果目录：{result_dir}")
    print(f"特征模式：{args.feature_mode}")
    print(f"计算设备：{device}")

    df, train_df, test_df, test_start, selected_features, ranking_df = prepare_hourly_dataset(
        input_paths=input_paths,
        target_col=args.target,
        feature_mode=args.feature_mode,
        top_k=args.k,
    )

    model, scaler, graph, metrics, pred_df = train_lightgcn(
        train_df=train_df,
        test_df=test_df,
        selected_features=selected_features,
        target_col=args.target,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        k_neighbors=args.k_neighbors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        device=device,
    )

    print("\n评估结果：")
    print(f"Train MAE  = {metrics['train_mae']:.6f}")
    print(f"Train RMSE = {metrics['train_rmse']:.6f}")
    print(f"Train R2   = {metrics['train_r2']:.6f}")
    print(f"Test MAE   = {metrics['test_mae']:.6f}")
    print(f"Test RMSE  = {metrics['test_rmse']:.6f}")
    print(f"Test R2    = {metrics['test_r2']:.6f}")

    joblib.dump(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "graph": {
                "location_ids": graph.location_ids,
                "location_index": graph.location_index,
                "n_locations": graph.n_locations,
                "n_hours": graph.n_hours,
                "edge_count": graph.edge_count,
            },
            "selected_features": selected_features,
            "model_hparams": {
                "emb_dim": args.emb_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
        },
        result_dir / "lightgcn_model.pkl",
    )

    config = {
        "model": MODEL_NAME,
        "input_files": [str(p) for p in input_paths],
        "target_col": args.target,
        "result_dir": str(result_dir),
        "test_start": str(test_start),
        "emb_dim": args.emb_dim,
        "n_layers": args.n_layers,
        "k_neighbors": args.k_neighbors,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "device": str(device),
        "graph_nodes": graph.n_locations + graph.n_hours,
        "graph_edges": graph.edge_count,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_features)),
        "prediction_granularity": "hourly_by_geohash",
        "region_count": int(df[REGION_GROUP_COL].nunique()),
        "feature_mode": args.feature_mode,
        "k": args.k,
    }

    save_standard_outputs(
        result_dir=result_dir,
        df=df,
        train_df=train_df,
        test_df=test_df,
        pred_df=pred_df,
        metrics=metrics,
        config=config,
        selected_features=selected_features,
        ranking_df=ranking_df,
        test_start=test_start,
        model_name=MODEL_NAME,
        model_filename="lightgcn_model.pkl",
        top_regions=args.top_regions,
        save_full_data=args.save_full_data,
    )

    print("\n全部完成。")
    print(f"所有结果已保存到：{result_dir}")


if __name__ == "__main__":
    main()
