"""One training step on a tiny synthetic hypergraph.

Confirms PyTorch, torch-sparse and the model all import and run on the
visible device. It does not need the Foursquare files.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from layer import NeighborSampler
from model import STHGCN
from utils import Cfg, seed_torch


def main():
    cfg = Cfg('best_conf/nyc.yml')
    cfg.model_args.sizes = [4, 4]
    cfg.model_args.intra_jaccard_threshold = 0.0
    cfg.model_args.inter_jaccard_threshold = 0.0
    cfg.run_args.batch_size = 2
    cfg.run_args.eval_batch_size = 2
    cfg.dataset_args.num_user = 8
    cfg.dataset_args.num_poi = 16
    cfg.dataset_args.num_category = 8
    cfg.dataset_args.padding_poi_id = 0
    cfg.dataset_args.padding_user_id = 0
    cfg.dataset_args.padding_poi_category = 0
    cfg.dataset_args.padding_hour_id = 0
    cfg.dataset_args.padding_weekday_id = 0

    if torch.cuda.is_available() and int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device
    seed_torch(0)

    checkin_offset = 4
    # user, poi, category, time, lon, lat, weekday, hour
    ci_x = torch.tensor([
        [1, 1, 1, 100, 116.40, 39.90, 1, 8],
        [1, 2, 1, 200, 116.41, 39.91, 1, 10],
        [1, 3, 1, 300, 116.42, 39.92, 1, 12],
        [2, 1, 1, 150, 116.43, 39.93, 2, 9],
    ], dtype=torch.float)
    # size, mean_lon, mean_lat, mean_time, start_time, end_time, pad, pad
    traj_x = torch.tensor([
        [2, 116.40, 39.90, 150, 100, 200, 0, 0],
        [2, 116.42, 39.92, 225, 150, 300, 0, 0],
    ], dtype=torch.float)

    ci2traj_edge_index = torch.tensor([
        [0, 1, 2, 3],
        [4, 4, 5, 5],
    ], dtype=torch.long)
    traj2traj_edge_index = torch.tensor([[4], [5]], dtype=torch.long)
    edge_attr = torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float)
    edge_type = torch.tensor([0], dtype=torch.long)

    node_idx = torch.tensor([4, 5], dtype=torch.long)
    sample_idx = torch.tensor([0, 1], dtype=torch.long)
    max_time = torch.tensor([250, 400], dtype=torch.long)
    label = torch.tensor([
        [2, 1, 116.4, 39.9, 0.4],
        [3, 1, 116.4, 39.9, 0.5],
    ], dtype=torch.float)

    sampler = NeighborSampler(
        [traj_x, ci_x],
        [traj2traj_edge_index, ci2traj_edge_index],
        [edge_attr, None],
        [None, torch.tensor([100, 200, 300, 150], dtype=torch.float)],
        [torch.zeros(1), torch.zeros(4)],
        [edge_type, None],
        cfg.model_args.sizes,
        sample_idx,
        node_idx,
        label,
        edge_delta_s=[torch.zeros(1), torch.zeros(4)],
        max_time=max_time,
        intra_jaccard_threshold=0.0,
        inter_jaccard_threshold=0.0,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    model = STHGCN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = next(iter(sampler))
    split_index = torch.max(batch.adjs_t[1].storage.row()).tolist()
    batch = batch.to(device)
    out, loss = model({
        'x': batch.x,
        'edge_index': batch.adjs_t,
        'edge_attr': batch.edge_attrs,
        'split_index': split_index,
        'delta_ts': batch.edge_delta_ts,
        'delta_ss': batch.edge_delta_ss,
        'edge_type': batch.edge_types,
    }, label=batch.y[:, 0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"smoke ok | device={device} | loss={loss.item():.4f} | "
        f"logits={tuple(out.shape)} | params={sum(p.numel() for p in model.parameters())}"
    )


if __name__ == '__main__':
    main()
