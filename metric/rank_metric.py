"""Ranking metrics for next-POI recommendation.

Each sample has one ground-truth next POI, the setting used by STHGCN,
GETNext, STAN and Flashback. Under that setting the hit rate at K is reported
under three names that refer to the same number:

    Acc@K = HR@K = Recall@K

NDCG@K and MAP@K use a single relevant item, so the ideal DCG is 1 and
NDCG@1 = MAP@1 = Acc@1. MAP@K is average precision truncated at K. MRR is the
mean reciprocal rank over the full candidate list. Precision@K and F1@K are
the single-label versions used by some session and POI papers.
"""
import csv
import json
import os
import os.path as osp

import torch


DEFAULT_KS = (1, 5, 10, 20)
_NOTE = 'One ground-truth next POI: Acc@K = HR@K = Recall@K. Scores are in [0, 1].'


def label_ranks(label, pred):
    """1-based rank of each ground-truth id inside a descending ranking.

    ``label`` is ``[N]`` or ``[N, 1]``. ``pred`` is ``[N, C]`` item ids, best
    first. A label that never appears gets rank ``C + 1``.
    """
    label = label.reshape(-1).to(dtype=torch.long)
    pred = pred.to(dtype=torch.long)
    if label.numel() != pred.size(0):
        raise ValueError(
            f'label has {label.numel()} rows but pred has {pred.size(0)}'
        )
    matches = pred.eq(label.unsqueeze(1))
    found = matches.any(dim=1)
    first = matches.to(dtype=torch.int64).argmax(dim=1) + 1
    missing = pred.size(1) + 1
    ranks = torch.where(found, first, first.new_full(first.shape, missing))
    return ranks, pred.size(1)


def _series_at_k(ranks, n_items, ks):
    """Return Acc, NDCG, MAP, Precision and F1 for every cutoff."""
    n = ranks.numel()
    ranks_f = ranks.to(dtype=torch.float64)
    found = ranks <= n_items
    acc, ndcg, mean_ap, precision, f1 = {}, {}, {}, {}, {}
    for k in ks:
        hit = found & (ranks <= k)
        hit_f = hit.to(dtype=torch.float64)
        acc[k] = hit_f.mean().item() if n else 0.0
        gain = torch.where(
            hit,
            1.0 / torch.log2(ranks_f + 1.0),
            torch.zeros(n, dtype=torch.float64, device=ranks.device),
        )
        ndcg[k] = gain.mean().item() if n else 0.0
        ap = torch.where(
            hit,
            1.0 / ranks_f,
            torch.zeros(n, dtype=torch.float64, device=ranks.device),
        )
        mean_ap[k] = ap.mean().item() if n else 0.0
        precision[k] = (hit_f / float(k)).mean().item() if n else 0.0
        f1_i = torch.where(
            hit,
            torch.full((n,), 2.0 / (k + 1.0), dtype=torch.float64, device=ranks.device),
            torch.zeros(n, dtype=torch.float64, device=ranks.device),
        )
        f1[k] = f1_i.mean().item() if n else 0.0
    reciprocal = torch.where(
        found,
        1.0 / ranks_f,
        torch.zeros(n, dtype=torch.float64, device=ranks.device),
    )
    mrr = reciprocal.mean().item() if n else 0.0
    return acc, ndcg, mean_ap, precision, f1, mrr


class EvalResult:
    """Metrics for one evaluation split, plus a report that lines up with papers."""

    def __init__(self, ks, acc, ndcg, map_at_k, precision, f1, mrr, loss, num_samples):
        self.ks = tuple(int(k) for k in ks)
        self.acc = dict(acc)
        # Same value as Acc@K. Kept under both names so a paper table can be
        # filled without translating the column header.
        self.hr = dict(acc)
        self.recall = dict(acc)
        self.ndcg = dict(ndcg)
        self.map_at_k = dict(map_at_k)
        self.precision = dict(precision)
        self.f1 = dict(f1)
        self.mrr = float(mrr)
        self.loss = None if loss is None else float(loss)
        self.num_samples = int(num_samples)

    def iter_series(self, aliases=True):
        rows = [('Acc', self.acc)]
        if aliases:
            rows.append(('HR', self.hr))
            rows.append(('Recall', self.recall))
        rows.extend([
            ('NDCG', self.ndcg),
            ('MAP', self.map_at_k),
            ('Precision', self.precision),
            ('F1', self.f1),
        ])
        return rows

    def as_dict(self, aliases=True):
        flat = {}
        for name, series in self.iter_series(aliases=aliases):
            for k in self.ks:
                flat[f'{name}@{k}'] = series[k]
        flat['MRR'] = self.mrr
        return flat

    def format_report(self, split='eval'):
        metric_rows = [
            ('Acc (= HR = Recall)', self.acc),
            ('NDCG', self.ndcg),
            ('MAP', self.map_at_k),
            ('Precision', self.precision),
            ('F1', self.f1),
        ]
        headers = ['Metric'] + [f'@{k}' for k in self.ks]
        body = []
        for name, series in metric_rows:
            body.append([name] + [_fmt_score(series[k]) for k in self.ks])
        main = _boxed_table(headers, body, first_left=True)

        compare = [(f'Acc@{k}', self.acc[k]) for k in self.ks]
        compare.append(('MRR', self.mrr))
        compare.extend((f'NDCG@{k}', self.ndcg[k]) for k in self.ks if k > 1)
        compare.extend((f'MAP@{k}', self.map_at_k[k]) for k in self.ks if k > 1)
        compare_table = _boxed_table(
            [name for name, _ in compare],
            [[_fmt_score(value) for _, value in compare]],
            first_left=False,
        )
        markdown = _markdown_table(
            [name for name, _ in compare],
            [_fmt_score(value) for _, value in compare],
        )

        loss_text = '-' if self.loss is None else f'{self.loss:.4f}'
        title = (
            f'Evaluation [{split}]'
            f'    samples {self.num_samples:,}'
            f'    loss {loss_text}'
        )
        caption = 'Paper comparison   Acc@K + MRR (STHGCN / GETNext / STAN), plus NDCG@K and MAP@K'
        identity = 'NDCG@1 = MAP@1 = Acc@1 for a single relevant POI.'
        lines = [
            title,
            _NOTE,
            identity,
            '',
            caption,
            *compare_table,
            '',
            'Full ranking metrics',
            *main,
            '',
            'Markdown',
            *markdown,
        ]
        return '\n'.join(lines)

    def save(self, directory, split='test', extra=None):
        """Write a text report, a JSON object and a one-row CSV."""
        os.makedirs(directory, exist_ok=True)
        payload = {
            'split': split,
            'num_samples': self.num_samples,
            'loss': self.loss,
            'note': _NOTE,
        }
        if extra:
            payload.update(extra)
        payload.update(self.as_dict(aliases=True))
        payload = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in payload.items()
        }

        text_path = osp.join(directory, f'{split}_metrics.txt')
        json_path = osp.join(directory, f'{split}_metrics.json')
        csv_path = osp.join(directory, f'{split}_metrics.csv')
        with open(text_path, 'w', encoding='utf-8') as handle:
            handle.write(self.format_report(split))
            handle.write('\n')
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write('\n')
        fieldnames = list(payload.keys())
        with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(payload)
        return text_path, json_path, csv_path


def evaluate_ranking(label, pred, ks=DEFAULT_KS, loss=None):
    """Score a ranked list of POI ids against a single ground-truth column."""
    if label.reshape(-1).numel() == 0:
        raise ValueError('evaluate_ranking received no samples')
    ks = tuple(int(k) for k in ks)
    ranks, n_items = label_ranks(label, pred)
    acc, ndcg, mean_ap, precision, f1, mrr = _series_at_k(ranks, n_items, ks)
    return EvalResult(
        ks=ks,
        acc=acc,
        ndcg=ndcg,
        map_at_k=mean_ap,
        precision=precision,
        f1=f1,
        mrr=mrr,
        loss=loss,
        num_samples=int(label.reshape(-1).numel()),
    )


def recall(lab, prd, k):
    """Backward-compatible Recall@K. Equals Acc@K for one ground-truth POI."""
    return _scalar(evaluate_ranking(lab, prd, ks=(k,)).recall[k])


def ndcg(lab, prd, k):
    return _scalar(evaluate_ranking(lab, prd, ks=(k,)).ndcg[k])


def map_k(lab, prd, k):
    return _scalar(evaluate_ranking(lab, prd, ks=(k,)).map_at_k[k])


def mrr(lab, prd):
    return _scalar(evaluate_ranking(lab, prd, ks=(1,)).mrr)


def _scalar(value):
    return torch.tensor(value, dtype=torch.float64)


def _fmt_score(value):
    return f'{value:.4f}'


def _fit(text, width, align):
    if align == 'left':
        return text.ljust(width)
    return text.rjust(width)


def _boxed_table(headers, rows, first_left):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    widths = [max(width, 6) for width in widths]

    def align_at(index):
        if index == 0 and first_left:
            return 'left'
        return 'right'

    def paint(cells):
        painted = [_fit(cell, widths[index], align_at(index)) for index, cell in enumerate(cells)]
        return '| ' + ' | '.join(painted) + ' |'

    rule = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
    return [rule, paint(headers), rule, *[paint(row) for row in rows], rule]


def _markdown_table(headers, cells):
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join('---:' for _ in headers) + ' |'
    body = '| ' + ' | '.join(cells) + ' |'
    return [head, sep, body]


def _legacy_recall(lab, prd, k):
    return torch.sum(torch.sum(lab == prd[:, :k], dim=1)).double() / lab.shape[0]


def _legacy_ndcg(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    if exist_pos.numel() == 0:
        return lab.new_zeros((), dtype=torch.float64)
    dcg = 1.0 / torch.log2(exist_pos.double() + 1)
    return torch.sum(dcg) / lab.shape[0]


def _legacy_map(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    if exist_pos.numel() == 0:
        return lab.new_zeros((), dtype=torch.float64)
    return torch.sum(1.0 / exist_pos.double()) / lab.shape[0]


def _legacy_mrr(lab, prd):
    exist_pos = torch.nonzero(prd == lab, as_tuple=False)[:, 1] + 1
    if exist_pos.numel() == 0:
        return lab.new_zeros((), dtype=torch.float64)
    return torch.sum(1.0 / exist_pos.double()) / lab.shape[0]


def verify_against_legacy():
    """Check hand-computed scores and agreement with the previous formulas."""
    perfect = torch.tensor([[0, 1, 2], [3, 0, 1]])
    perfect_label = torch.tensor([[0], [3]])
    result = evaluate_ranking(perfect_label, perfect, ks=(1, 2), loss=0.5)
    assert result.acc[1] == 1.0
    assert result.ndcg[1] == 1.0
    assert result.mrr == 1.0
    assert result.hr[1] == result.recall[1] == result.acc[1]
    assert abs(result.precision[2] - 0.5) < 1e-8
    assert abs(result.f1[1] - 1.0) < 1e-8

    second = torch.tensor([[1, 0, 2], [4, 3, 0]])
    second_label = torch.tensor([[0.0], [3.0]])
    result = evaluate_ranking(second_label, second, ks=(1, 5))
    assert result.acc[1] == 0.0
    assert result.acc[5] == 1.0
    assert abs(result.mrr - 0.5) < 1e-8
    assert abs(result.ndcg[5] - (1.0 / torch.log2(torch.tensor(3.0)))) < 1e-8
    assert abs(result.map_at_k[1] - 0.0) < 1e-8
    assert abs(result.map_at_k[5] - 0.5) < 1e-8
    assert abs(result.precision[5] - 0.2) < 1e-8
    assert abs(result.f1[5] - (2.0 / 6.0)) < 1e-8

    missing = torch.tensor([[1, 2, 3]])
    missing_label = torch.tensor([[9]])
    result = evaluate_ranking(missing_label, missing, ks=(1, 3))
    assert result.acc[3] == 0.0
    assert result.mrr == 0.0
    assert result.ndcg[3] == 0.0

    generator = torch.Generator().manual_seed(0)
    pred = torch.stack([torch.randperm(30, generator=generator) for _ in range(40)])
    label = torch.randint(0, 35, (40, 1), generator=generator)
    got = evaluate_ranking(label, pred, ks=(1, 5, 10, 20))
    for k in got.ks:
        assert abs(got.recall[k] - _legacy_recall(label, pred, k).item()) < 1e-6
        assert abs(got.ndcg[k] - _legacy_ndcg(label, pred, k).item()) < 1e-6
        assert abs(got.map_at_k[k] - _legacy_map(label, pred, k).item()) < 1e-6
    assert abs(got.mrr - _legacy_mrr(label, pred).item()) < 1e-6
    report = got.format_report('test')
    assert 'Acc (= HR = Recall)' in report
    assert 'Acc@1' in report and 'MRR' in report
    text_path, json_path, csv_path = got.save(
        '/tmp/sthgcn_metric_check',
        split='test',
        extra={'num_params': 12},
    )
    with open(json_path, encoding='utf-8') as handle:
        payload = json.load(handle)
    assert payload['Acc@1'] == payload['HR@1'] == payload['Recall@1']
    assert payload['num_params'] == 12
    assert osp.isfile(text_path) and osp.isfile(csv_path)
    return report


if __name__ == '__main__':
    print(verify_against_legacy())
    print('rank metrics ok')
