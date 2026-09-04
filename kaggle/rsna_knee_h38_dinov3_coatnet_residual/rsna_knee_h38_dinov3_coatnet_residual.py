"""H-38: small DINOv3 residual fused with the H-36 CoAtNet arm.

The DINOv3 and CoAtNet bodies are adapted from public Apache-2.0 Kaggle
notebooks; the two model-weight datasets are public CC0-1.0 sources. This file
is a standalone, no-internet Kaggle script assembled from audited public cells.
The transformer arm is persisted first, then the CoAtNet arm creates a
per-target rank fusion. H-37's 50/50 blend scored 0.922 versus H-36's
0.928, so this run tests a conservative 20% DINOv3 residual and 80% CoAtNet.
No private kernel output is an input.
"""

from __future__ import annotations
import os as _os
def _comp_root():
    # An API-attached competition mounts at /kaggle/input/competitions/<slug>/; only a UI-added one
    # uses the short /kaggle/input/<slug>/ path. This notebook already resolves BOTH for the test
    # root but hardcodes ROOT/COMP to the short form, so an API-pushed fork dies at cell 1 with
    # FileNotFoundError on train.csv. Resolve it the same way instead of assuming.
    for _c in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
               "/kaggle/input/rsna-knee-abnormality-detection"):
        if _os.path.isdir(_c):
            return _c
    raise RuntimeError("competition data not found under /kaggle/input")
_COMP_ROOT = _comp_root()
import os
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
ASSET = Path('/kaggle/input/datasets/mattiaangeli')
ROOT = Path(_COMP_ROOT)
DINO = Path('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1')
T0 = time.time()
DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
SEED = 2026
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b')
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')

def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)
IMG = CACHE_IMG

def available_gb():
    try:
        with open('/proc/meminfo') as fh:
            info = {k.strip(): v for k, v in (l.split(':', 1) for l in fh if ':' in l)}
        return int(info['MemAvailable'].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION

def plan_cache(n_study, n_test=0):
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f'memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; sizing for {n_study} train + {n_total - n_study} test studies -> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot' + (f' (wanted {N_GROUP_MAX})' if groups < N_GROUP_MAX else ''))
    return groups
N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))
CACHE_SLICES = GROUP * N_GROUP
HDR_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'Laterality', 'PixelSpacing', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'ImagePositionPatient', 'ImageOrientationPatient']

def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split('|')]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None

def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
        iop = _hdr_vec(getattr(r, 'ImageOrientationPatient', None), 6)
        ps = _hdr_vec(getattr(r, 'PixelSpacing', None), 2)
        rows, cols = (getattr(r, 'Rows', None), getattr(r, 'Columns', None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else 'R' if m < 0 else 'L'
    return out

def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby('StudyInstanceUID'):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else 'R' if x < 0 else 'L'
    return out

def lat_of(h, tag=''):
    geo = side_from_corner_x(h) if RULES['lat'] == 'corner_x' else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby('StudyInstanceUID'):
        v = [str(x).strip().upper() for x in g['Laterality'].dropna()]
        if RULES['lat'] == 'corner_x' and 'ImageLaterality' in g.columns:
            v += [str(x).strip().upper() for x in g['ImageLaterality'].dropna()]
        v = [x[0] for x in v if x and x[0] in ('L', 'R')]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f'{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)')
    return d

def probe(item):
    split, study, series, path = item
    row = {'split': split, 'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'dir': path}
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith('.dcm')))
        row['files'] = files
        row['n_slices'] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == 'MultiValue':
                row[t] = '|'.join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row['err'] = str(exc)[:120]
    return row

def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=['split', 'StudyInstanceUID', 'SeriesInstanceUID', 'dir', 'files', 'n_slices'] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)

def annotate(df):
    desc = df['SeriesDescription'].fillna('') + ' ' + df['SequenceName'].fillna('')
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df['fatsat'] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').str.upper().str.contains('GR')
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX))
    df['weight'] = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['fluid'] = np.isin(df['weight'], ['PD', 'T2'])
    df['px'] = pd.to_numeric(df['PixelSpacing'].fillna('').str.split('|').str[0].replace('', np.nan), errors='coerce')
    return df

def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df['plane'] = series_df['SeriesInstanceUID'].map(plane_map)
    out = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g['plane'] == plane) & (g['fatsat'] == fs)
            if fluid is not None:
                sel &= g['fluid'] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES['slot_fallback'] and (fluid is False):
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0]
        out[study] = chosen
    return out
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []

def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))

def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)

def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get('ordered') or rec['files'], rec['dir'], rec['px'])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES['decode_fill'] == 'zero':
        if not got:
            DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)

def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
ORDER_CACHE = os.environ.get('RSNA_ORDER_CACHE') or None

def build_cache(slot_map, plane_map, lat_map, tag):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f'{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum((len(j[3]['files']) for j in jobs))
    log(f'{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)')
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec['SeriesInstanceUID'])
            if e and len(e['files']) == len(rec['files']):
                rec['ordered'] = e['files']
                ok += int(e['good'])
                hit += 1
        jobs = [j for j in jobs if 'ordered' not in j[3]]
        log(f'{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read')
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(block, pool.map(lambda j: order_slices(j[3]), block)):
                rec['ordered'] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec['SeriesInstanceUID']] = {'files': files, 'good': bool(good)}
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f'{tag}: ordering budget spent at {done}/{len(jobs)}; the rest keep file order')
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix('.tmp')
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f'{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: decoding {len(jobs)} slot-series')
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f'  {tag} {done}/{len(jobs)}')
            if time.time() - T0 > TIME_BUDGET:
                log(f'  {tag}: time budget reached during decode')
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f'{tag}: {int(mask.sum())}/{len(jobs)} slots filled' + (f'; {n_failed} series had a slice that would not decode' if n_failed else ''))
    gc.collect()
    return (studies, cache, mask)

class SlotHead(nn.Module):

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer('slot_prior', p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias

class Model(nn.Module):

    def __init__(self, backbone, dim, pool='cls_mean', prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)

def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)
FINGERPRINT_TOL = 0.002

def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out

def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d

class WeightsError(RuntimeError):
    pass
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
LEGACY_FOLD_SOFTPOOL_BETA = {'ACL': 6.0, 'MCL': 6.0, 'Medial Meniscus': 8.0, 'Lateral Meniscus': 8.0, "Baker's": 8.0, 'Contusion': 8.0, 'Fracture': 10.0}
LEGACY_FOLD_SOFTPOOL_ALPHA = {'ACL': 0.2, 'MCL': 0.2, 'Medial Meniscus': 0.25, 'Lateral Meniscus': 0.25, "Baker's": 0.2, 'Contusion': 0.2, 'Fracture': 0.15}

def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values

def legacy_fold_soft_window_pool(original_probs, target_idx):
    values = original_probs.mean(0).clone()
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values

@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    public_soft = np.concatenate(public_soft_out) if public_soft_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier, public_soft)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            state, fp = (m['state'], None)
        else:
            ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
            state, fp = (ck['model'], ck.get('fingerprint'))
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(f"  {m['id']}: no stored fingerprint (legacy bundle) -- accepted at reduced weight")
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))

def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])

def combine_public_members_by_fold(per_member, pred_key='pred'):
    all_ids = sorted({study for member in per_member for study in member['ids']})
    position = {study: i for i, study in enumerate(all_ids)}
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get('fold')
        key = f'fold_{fold}' if fold is not None else f'member_{i}'
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member['ids']]] = np.asarray(member[pred_key], np.float64)
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(pd.DataFrame(raw_fold_mean).rank(method='average', pct=True).to_numpy(np.float64))
        diagnostics.append({'ensemble_group': key, 'members': len(members_in_fold)})
    if len(fold_ranks) != 5:
        raise WeightsError(f'legacy branch requires five folds, found {len(fold_ranks)}')
    return (all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics))

def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    output = np.asarray(frontier_rank, np.float64).copy()
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output

def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if float(np.std(pred)) < 1e-09:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                log(f"  {m['id']}: public-frontier vote omitted because only {len(starts)} / {len(starts_full)} windows completed")
            all_ids, acc = _combine(per_member)
            write_submission(acc, all_ids, test_df, 'submission.csv')
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s){(', jitter' if jitter else '')}); submission.csv = weighted rank mean of {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))
                starts, jit = (starts_full, False)
                if est['fixed'] is not None and est['win'] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    if est['fixed'] + est['win'] > room:
                        log(f'  {left / 60:.0f} min left: surrendering {len(pending)} member(s); not one more fits')
                        pending.clear()
                        return (None, None, False)
                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6
                    per_win = est['win'] * (2 if jit else 1)
                    n_win = int((room - est['fixed']) / per_win) if per_win > 0 else len(starts_full)
                    n_win = max(1, min(len(starts_full), n_win))
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid:mid + n_win]
                return (pending.pop(0), starts, jit)

        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, public_p, public_soft, (fs, ws) = _run_member(path, m, d, Cte, Mte, idx, starts, jit)
                        with STATE_LOCK:
                            est['fixed'], est['win'] = (fs, ws)
                        bank(m, st_te, p, starts, jit, public_p, public_soft)
                        break
                    except Exception as exc:
                        log(f"  MEMBER {m['id']} failed on {d} ({type(exc).__name__}: {exc}); " + ('retrying on peer device' if attempt == 0 and others else 'dropped -- costs one vote, not the run'))
                        if d.type == 'cuda':
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        del Cte, Mte
        gc.collect()
    if not per_member:
        raise WeightsError('no member produced predictions; submission stays at 0.5')
    all_ids, acc = _combine(per_member)
    sub = write_submission(acc, all_ids, test_df, 'submission.csv')
    log(f'final submission.csv = weighted rank mean of {len(per_member)} member(s); {sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}')
    if len(public_frontier_members) == len(members):
        frontier_ids, frontier_acc = _combine(public_frontier_members)
        frontier_sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission_public_0899.csv')
        log(f'submission_public_0899.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {frontier_sub.shape}; nulls {int(frontier_sub[TARGETS].isna().sum().sum())}')
        fold_ids, fold_frontier, fold_diagnostics = combine_public_members_by_fold(public_frontier_members, 'pred')
        soft_ids, fold_soft, _ = combine_public_members_by_fold(public_frontier_members, 'soft_pred')
        if fold_ids != soft_ids:
            raise WeightsError('legacy hard/soft study order mismatch')
        legacy_prediction = blend_legacy_frontier_and_soft(fold_frontier, fold_soft)
        legacy_sub = write_submission(legacy_prediction, fold_ids, test_df, 'submission_legacy_fold_blend.csv')
        fold_diagnostics.to_csv('legacy_fold_diagnostics.csv', index=False)
        log(f'legacy DINO aggregation written from five folds; {legacy_sub.shape}')
    else:
        log(f'public-frontier fallback not emitted: {len(public_frontier_members)} / {len(members)} required public members completed')
    return sub

def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")

def augment(imgs, generator=None):
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = (x.shape[0], x.device)
    rot = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev, generator=generator) * AUG_SCALE
    tx = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    cos, sin = (torch.cos(rot) / sc, torch.sin(rot) / sc)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = (cos, -sin, tx)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = (sin, cos, ty)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev, generator=generator) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)

def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub

def find_dinov2(variant='small'):
    if not (DINO / 'config.json').is_file():
        raise FileNotFoundError(DINO)
    return DINO

def legacy_group_members():
    return {}

def run_dinov2():
    path = ASSET / 'rsna-knee-weights'
    infer_from_package(path, DEVS[0])
    public = Path('/kaggle/working/submission_public_0899.csv')
    if not public.is_file():
        raise RuntimeError('public DINOv2 frontier was not produced')
    public.replace('/kaggle/working/submission.csv')
    for name in ('submission_legacy_fold_blend.csv', 'legacy_fold_diagnostics.csv'):
        candidate = Path('/kaggle/working') / name
        if candidate.is_file():
            candidate.unlink()

# H-38 deliberately omits the DINOv2/RadImageNet branch from the source notebook.
# The primary transformer arm is the independent Mattia DINOv3 model below.

_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
CROP_MM = 130.0
SIZE = 336
SLICE_BAND = (0.12, 0.88)
N_SLICE = 16
INTENSITY = 'slice'
SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
N_SLOT = len(SLOTS)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
COMP = Path(_COMP_ROOT)
CKPT = ASSET / 'knee-mri-fold-weights'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'competition : {COMP}')
print(f'checkpoints : {CKPT}')
print(f'device      : {DEV}')
for i in range(torch.cuda.device_count() if DEV == 'cuda' else 0):
    cc = torch.cuda.get_device_capability(i)
    print(f'  gpu{i}       : {torch.cuda.get_device_name(i)} sm_{cc[0]}{cc[1]}, {torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GiB, native bf16={cc >= (8, 0)}')
SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.exists():
    SERIES_ROOT = COMP / 'train_series'
print('series root:', SERIES_ROOT)

def ordered_files(sdir, cap=64):
    keyed = []
    for f in sdir.glob('*.dcm'):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            keyed.append((int(ds.InstanceNumber), str(f)))
        except Exception:
            continue
        if len(keyed) >= cap * 4:
            break
    return [f for _, f in sorted(keyed)]

def series_side(path):
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0

def read_crop(path):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = (arr.shape[0] // 2, arr.shape[1] // 2)
    y0, y1 = (max(0, cy - half), min(arr.shape[0], cy + half))
    x0, x1 = (max(0, cx - half), min(arr.shape[1], cx + half))
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def window(crop, lo, hi, flip):
    c = np.clip((crop - lo) / max(hi - lo, 1e-06), 0, 1)
    img = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def render(path, flip):
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return window(crop, lo, hi, flip)

def build_study(args):
    idx, study, recs = args
    out = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    if len(rows):
        for s_i, (plane, fs) in enumerate(SLOTS):
            sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
            if sub.empty:
                continue
            files = ordered_files(SERIES_ROOT / study / sub.iloc[0].SeriesInstanceUID)
            if not files:
                continue
            flip = plane != 'Sagittal' and series_side(files[0]) < 0
            lo, hi = SLICE_BAND
            i0 = int(round(lo * (len(files) - 1)))
            i1 = int(round(hi * (len(files) - 1)))
            avail = list(range(i0, i1 + 1))
            if len(avail) >= N_SLICE:
                picks = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, N_SLICE)]
                off = 0
            else:
                picks, off = (avail, (N_SLICE - len(avail)) // 2)
            if INTENSITY == 'series':
                crops = [read_crop(files[p]) for p in picks]
                got = [x for x in crops if x is not None]
                if got:
                    samp = np.concatenate([x[::4, ::4].ravel() for x in got])
                    lo_, hi_ = np.percentile(samp, [1, 99])
                    for c, x in enumerate(crops):
                        if x is None:
                            x = read_crop(files[min(len(files) - 1, picks[c] + 1)])
                        if x is not None:
                            out[s_i, off + c] = (window(x, lo_, hi_, flip) * 255).astype(np.uint8)
            else:
                for c, p in enumerate(picks):
                    img = render(files[p], flip)
                    if img is None:
                        img = render(files[min(len(files) - 1, p + 1)], flip)
                    if img is not None:
                        out[s_i, off + c] = (img * 255).astype(np.uint8)
            mask[s_i] = len(picks)
    return (idx, out, mask)
sub_df = pd.read_csv(COMP / 'sample_submission.csv')
ser_csv = pd.read_csv(COMP / 'test_series.csv')
if not (COMP / 'test_series').exists():
    ser_csv = pd.read_csv(COMP / 'train_series.csv')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')
N_SLOT_TYPES, MASK_IDX = (6, 0)

def segment_softmax(scores, sidx, B):
    T, K = scores.shape
    idx = sidx.unsqueeze(1).expand(-1, K)
    m = torch.full((B, K), float('-inf'), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce='amax', include_self=True)
    e = (scores - m[sidx]).exp()
    s = torch.zeros(B, K, device=scores.device, dtype=scores.dtype).index_add_(0, sidx, e)
    return e / s[sidx].clamp(min=1e-06)

class MeanMaxPool(nn.Module):

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        D = f.shape[1]
        cnt = torch.zeros(B, device=f.device, dtype=f.dtype).index_add_(0, sidx, torch.ones(f.shape[0], device=f.device, dtype=f.dtype))
        mean = torch.zeros(B, D, device=f.device, dtype=f.dtype).index_add_(0, sidx, f)
        mean = mean / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=f.device, dtype=f.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), f, reduce='amax', include_self=True)
        return (torch.cat([mean, mx], 1), None)

class LabelAttentionPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=4, slot_bias=True):
        super().__init__()
        self.d, self.k, self.h = (d, n_labels, n_heads)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.key, self.val = (nn.Linear(d, d), nn.Linear(d, d))
        self.slot_bias = nn.Parameter(torch.zeros(n_labels, N_SLOT_TYPES + 1)) if slot_bias else None

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        scores = self.key(f) @ self.q.t() / self.d ** 0.5
        if self.slot_bias is not None and slot is not None:
            scores = scores + self.slot_bias.t()[slot]
        a = segment_softmax(scores, sidx, B)
        out = torch.zeros(B, self.k, self.d, device=f.device, dtype=f.dtype)
        out = out.index_add_(0, sidx, a.unsqueeze(-1) * self.val(f).unsqueeze(1))
        return (out, a)

class TokenXAttnPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=6, dropout=0.2):
        super().__init__()
        self.d, self.k = (d, n_labels)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, d, padding_idx=0)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tok, sidx, B, slot=None, return_attn=False):
        T, N, D = tok.shape
        cnt = torch.bincount(sidx, minlength=B)
        S = int(cnt.max().item())
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(T, device=tok.device) - starts[sidx]
        kv = tok + self.slot_emb(slot).unsqueeze(1)
        pad = tok.new_zeros(B, S, N, D)
        pad[sidx, pos] = kv
        keep = torch.zeros(B, S, dtype=torch.bool, device=tok.device)
        keep[sidx, pos] = True
        kpm = ~keep.repeat_interleave(N, dim=1)
        pad = self.kv_norm(pad.reshape(B, S * N, D))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, pad, pad, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        cls = tok[:, 0]
        mean = torch.zeros(B, D, device=tok.device, dtype=tok.dtype).index_add_(0, sidx, cls) / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=tok.device, dtype=tok.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), cls, reduce='amax', include_self=True)
        base = torch.cat([mean, mx], 1).unsqueeze(1).expand(-1, self.k, -1)
        return (torch.cat([att, base], -1), w)

class ViTSlotToken(nn.Module):

    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, 'num_prefix_tokens', 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, 'attn', None)
            if a is not None and hasattr(a, 'num_prefix_tokens'):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, 'patch_drop', None), x)
        x = self._maybe(getattr(v, 'norm_pre', None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, 'rope_mixed', False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class _GatedDepthBlock(nn.Module):

    def __init__(self, n_slice, dropout=0.0, ls_init=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_slice)
        self.v = nn.Conv2d(n_slice, n_slice, 1)
        self.g = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), ls_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.v(z) * F.silu(self.g(z))))

class DepthCompress(nn.Module):

    def __init__(self, n_slice=16, out_ch=3, depth=1, dropout=0.0, ls_init=0.1, imagenet=True, proj_noise=0.25):
        super().__init__()
        self.imagenet = imagenet
        self.blocks = nn.ModuleList([_GatedDepthBlock(n_slice, dropout, ls_init) for _ in range(depth)])
        self.proj = nn.Conv2d(n_slice, out_ch, 1, bias=True)
        if imagenet:
            self.register_buffer('mu', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer('sd', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))

    def forward(self, x):
        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.proj(z)
        if self.imagenet:
            z = (z - self.mu.to(z.dtype)) / self.sd.to(z.dtype)
        return z * keep
N_PLANE, N_CONTRAST = (3, 2)
_PLANE_OF = lambda s: torch.clamp(s - 1, 0, 5) // 2
_CONTRAST_OF = lambda s: torch.clamp(s - 1, 0, 5) % 2

class SlotDepthMixer(nn.Module):

    def __init__(self, n_slice=16, ksize=5, alpha_max=0.25):
        super().__init__()
        self.n_slice, self.ksize, self.r = (n_slice, ksize, ksize // 2)
        self.alpha_max = alpha_max
        b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        self.register_buffer('base', b.log()[self.r:])
        n_u = self.r + 1
        self.shared = nn.Parameter(torch.zeros(n_u))
        self.plane_k = nn.Parameter(torch.zeros(N_PLANE, n_u))
        self.contrast_k = nn.Parameter(torch.zeros(N_CONTRAST, n_u))
        self.g0 = nn.Parameter(torch.zeros(()))
        self.gate_p = nn.Parameter(torch.zeros(N_PLANE))
        self.gate_c = nn.Parameter(torch.zeros(N_CONTRAST))
        idx = torch.arange(n_slice)
        self.register_buffer('off', idx[None, :] - idx[:, None])

    def kernel(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        half = self.base + self.shared + self.plane_k[p] + self.contrast_k[c]
        full = torch.cat([half.flip(-1)[..., :self.r], half], dim=-1)
        return F.softmax(full, dim=-1)

    def alpha(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        return self.alpha_max * torch.tanh(self.g0 + self.gate_p[p] + self.gate_c[c])

    def forward(self, x, slot, vmask):
        T, S, H, W = x.shape
        if vmask is None:
            raise ValueError('stem=mixer requires the padding mask')
        k = self.kernel(slot)
        v = vmask.to(k.dtype)
        d = self.off + self.r
        inb = (d >= 0) & (d < self.ksize)
        kk = k[:, d.clamp(0, self.ksize - 1)] * inb
        M = kk * v[:, None, :]
        den = M.sum(-1, keepdim=True)
        eye = torch.eye(S, device=x.device, dtype=M.dtype).expand(T, S, S)
        ok = (den > 1e-06) & v[:, :, None].bool()
        M = torch.where(ok, M / den.clamp(min=1e-06), eye)
        a = self.alpha(slot)[:, None, None]
        Aop = ((1.0 - a) * eye + a * M).to(x.dtype)
        if x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()):
            y = torch.bmm(x.permute(0, 2, 3, 1).reshape(T, H * W, S), Aop.transpose(1, 2))
            return y.reshape(T, H, W, S).permute(0, 3, 1, 2)
        return torch.bmm(Aop, x.reshape(T, S, H * W)).reshape(T, S, H, W)

def _seg_mean_max(v, sidx, B):
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce='amax', include_self=True)
    return torch.cat([mean, mx], 1)

def _pad_kv(x, sidx, B, norm):
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))

class _GatedDelta(nn.Module):

    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)

class TokenResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class CodexResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class ClsAddPool(nn.Module):

    def __init__(self, d, n_labels=12, pe=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4 * d + pe), nn.Dropout(dropout), nn.Linear(4 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        return (self.net(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), _seg_mean_max(tok[:, 0], sidx, B), pres], 1)), None)

class Readout(nn.Module):

    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        if pool in ('xres', 'clsadd', 'xcodex'):
            self.pool = {'xres': TokenResidualPool, 'clsadd': ClsAddPool, 'xcodex': CodexResidualPool}[pool](d, n_labels, pe=pe)
        elif pool in ('attn', 'xattn'):
            if pool == 'xattn':
                self.pool = TokenXAttnPool(d, n_labels)
                wd = 3 * d + pe
            else:
                self.pool = LabelAttentionPool(d, n_labels)
                wd = d + pe
            self.norm = nn.LayerNorm(wd)
            self.w = nn.Parameter(torch.randn(n_labels, wd) * (1.0 / wd ** 0.5))
            self.b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.pool = MeanMaxPool()
            self.net = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(0.2), nn.Linear(2 * d + pe, n_labels))
        self.drop = nn.Dropout(0.2)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        if self.pool_kind in ('xres', 'clsadd', 'xcodex'):
            return self.pool(f, slot, sidx, B, pres)[0]
        pooled, attn = self.pool(f, sidx, B, slot=slot, return_attn=return_attn)
        if self.pool_kind in ('attn', 'xattn'):
            x = torch.cat([pooled, pres.unsqueeze(1).expand(-1, self.k, -1)], -1)
            x = self.drop(self.norm(x))
            return (x * self.w).sum(-1) + self.b
        return self.net(torch.cat([pooled, pres], 1))

class Net(nn.Module):

    def __init__(self, enc, cond, n_meta=0, pool='mean_max', stem='native', n_slice=16):
        super().__init__()
        self.enc, self.cond = (enc, cond)
        self.compress = DepthCompress(n_slice, 3) if stem == 'compress' else None
        self.mixer = SlotDepthMixer(n_slice) if stem == 'mixer' else None
        self.tokens = pool in ('xattn', 'xres', 'clsadd', 'xcodex')
        D = enc.num_features
        self.meta_mlp = nn.Sequential(nn.LayerNorm(n_meta), nn.Linear(n_meta, 128), nn.GELU(), nn.Linear(128, D)) if n_meta > 0 else None
        self.readout = Readout(pool, D)
        if cond == 'post':
            self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, D, padding_idx=MASK_IDX)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        if self.mixer is not None:
            im = self.mixer(im, slot, vm)
        if self.compress is not None:
            im = self.compress(im)
        f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
        if self.tokens:
            inner = getattr(self.enc, 'vit', self.enc)
            orig = getattr(self.enc, '_orig_prefix', getattr(inner, 'num_prefix_tokens', 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        else:
            f = self.enc.forward_head(f, pre_logits=True)
            if f.dim() > 2:
                f = f.flatten(1)
        ex = (lambda v: v.unsqueeze(1)) if self.tokens else lambda v: v
        if self.cond == 'post':
            f = f + ex(self.slot_emb(slot))
        if self.meta_mlp is not None and smeta.shape[1] > 0:
            mt = self.meta_mlp(smeta)
            f = torch.cat([f, mt.unsqueeze(1)], 1) if self.tokens else f + mt
        return self.readout(f, slot, sidx, B)
models = []
for ckpt_path in sorted(CKPT.glob('*_f*.pt')):
    z = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = z['cfg']
    _stem = cfg.get('stem', 'native')
    _in = 3 if _stem == 'compress' else cfg.get('n_slice', 16)
    enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=_in, **{'img_size': cfg['img']} if 'vit_' in cfg['backbone'] else {})
    if cfg['cond'] == 'token':
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
    m = Net(enc, cfg['cond'], cfg.get('n_meta', 0), cfg['pool'], stem=_stem, n_slice=cfg.get('n_slice', 16))
    missing, unexpected = m.load_state_dict(z['state_dict'], strict=False)
    assert not missing, f'missing {missing[:5]}'
    assert not unexpected, f'unexpected {unexpected[:5]}'
    models.append(m.eval())
    print(f"loaded {ckpt_path.name}  fold {z['fold']}  {cfg['backbone']} pool={cfg['pool']} meta={cfg['meta']}")
CFG = cfg
assert CFG.get('n_meta', 0) == 0, f"checkpoint expects {CFG['n_meta']} metadata features -- build slot_meta for the TEST studies and pass it to predict() before submitting"
print(f"\n{len(models)} fold models ready | input norm: {CFG.get('norm', 'none')}")
AMP_PREF = 'bf16'

def amp_for(dev):
    if not str(dev).startswith('cuda'):
        return (torch.float32, False)
    cc = torch.cuda.get_device_capability(dev)
    if AMP_PREF == 'bf16':
        return (torch.bfloat16, True)
    if AMP_PREF == 'fp16':
        return (torch.float16, True)
    if AMP_PREF == 'fp32':
        return (torch.float32, False)
    return (torch.bfloat16 if cc >= (8, 0) else torch.float16, True)
AMP_DT, AMP_ON = amp_for(DEV)
WORKERS = max(1, min(4, os.cpu_count() or 4))
CHUNK = 48
MICRO = 8
models = [m.to(DEV).eval() for m in models]
print(f"device {DEV} | amp {str(AMP_DT).split('.')[-1]} (on={AMP_ON}) | workers {WORKERS} | chunk {CHUNK} | micro {MICRO}")

def _norm_(im):
    k = CFG.get('norm', 'none')
    if k == 'zscore':
        m = (im > 0).float()
        n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (im * m).sum(dim=(1, 2, 3), keepdim=True) / n
        var = (((im - mu) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n
        return (im - mu) / (var.sqrt() + 1e-06) * m
    if k == 'imagenet':
        m = (im > 0).float()
        return (im - 0.485) / 0.229 * m
    return im

@torch.no_grad()
def _micro(images, masks):
    dev = DEV
    ims, slots, sidx, vms = ([], [], [], [])
    for b in range(len(masks)):
        present = np.nonzero(masks[b] > 0)[0]
        if len(present) == 0:
            continue
        blk = images[b][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), b, dtype=torch.long))
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    if not ims:
        return out
    im = _norm_(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), CFG.get('n_meta', 0), device=dev)
    per = torch.zeros(len(models), len(masks), len(LABELS), device=dev, dtype=torch.float32)
    with torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu', dtype=AMP_DT, enabled=AMP_ON):
        for fold_index, model in enumerate(models):
            per[fold_index] = torch.sigmoid(model(im, sl, sm, si, len(masks), vm=vm).float())
    got = per.cpu().numpy()
    keep = np.array([(masks[b] > 0).any() for b in range(len(masks))])
    out[:, keep] = got[:, keep]
    return out

def predict(images, masks):
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    for a in range(0, len(masks), MICRO):
        b = min(a + MICRO, len(masks))
        out[:, a:b] = _micro(images[a:b], masks[a:b])
    return out
preds = np.full((len(models), len(studies), len(LABELS)), np.nan, np.float32)
t0, done = (time.time(), 0)
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    for c0 in range(0, len(studies), CHUNK):
        block = studies[c0:c0 + CHUNK]
        imgs = np.zeros((len(block), N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
        msks = np.zeros((len(block), N_SLOT), np.uint8)
        futs = [ex.submit(build_study, (i, s, by.get(s, []))) for i, s in enumerate(block)]
        for f in as_completed(futs):
            try:
                i, a, k = f.result()
                imgs[i], msks[i] = (a, k)
            except Exception as e:
                print(f'  study failed: {type(e).__name__}: {e}')
        preds[:, c0:c0 + len(block)] = predict(imgs, msks)
        done += len(block)
        el = time.time() - t0
        print(f'  {done:,}/{len(studies):,}  {el / 60:.1f}m  eta {el / done * (len(studies) - done) / 60:.1f}m', flush=True)
        del imgs, msks
        gc.collect()
print(f'\ninference done in {(time.time() - t0) / 60:.1f} min')
A5_W = 0.45
A5_LABELS = list(LABELS)
_a5_ok = np.isfinite(preds).all(axis=(0, 2))
_a5_rank_mean = np.zeros((len(studies), len(LABELS)), np.float64)
for fold_index in range(preds.shape[0]):
    fold = preds[fold_index][_a5_ok]
    ordinal = fold.argsort(0).argsort(0).astype(np.float64)
    _a5_rank_mean[_a5_ok] += ordinal / max(len(fold) - 1, 1)
_a5_rank_mean /= preds.shape[0]
_a5_rank_mean[~_a5_ok] = np.nan
A5_PREDS = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_rank_mean.astype(np.float32)))
for _a5k, _a5v in _A5_SAVED.items():
    globals()[_a5k] = _a5v
del _A5_SAVED, _a5k, _a5v
# H-38 has no pre-existing transformer parent: persist the exact DINOv3 fold rank mean
# as the parent that the CoAtNet branch will fuse below.
_a5_out = sub_df[['StudyInstanceUID']].copy()
_a5_out[A5_LABELS] = _a5_rank_mean
assert list(_a5_out.columns) == ['StudyInstanceUID'] + A5_LABELS
assert _a5_out['StudyInstanceUID'].tolist() == sub_df['StudyInstanceUID'].astype(str).tolist()
assert np.isfinite(_a5_out[A5_LABELS].to_numpy()).all()
_a5_out.to_csv('/kaggle/working/submission_dinov3.csv', index=False)
_a5_out.to_csv('/kaggle/working/submission.csv', index=False)
print('H-38 transformer parent = DINOv3 five-fold rank mean', _a5_out.shape, flush=True)


# External Raptor checkpoints retain their original provenance and training recipe.
import os, sys, glob, time, json, gc
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 62   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Three arms: (weights filename, fallback arch, fallback res). ck carries arch+res too.
# Selected 2026-08-19 by greedy forward selection AND exhaustive subset search over a 7-arm
# panel on the 45-study gold set (phase2/blend_panel.py); both agree on this exact set.
# Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
# Blend {coatnet+swin+effv2l} = 0.9068 (2-arm {coatnet+swin} = 0.9059, coatnet alone 0.9025).
# Dropped as redundant: cnn336 (0.8833, the former champion), cnbase384 (0.8754),
# cnlarge384 (0.8752), maxvit384 (0.8438).
#
# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus.
#
# Why one arm and not the 3-arm blend: on the live leaderboard CoAtNet alone scored 0.914 while
# every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there -- the ~+0.010 it showed
# on the old 45-study gold set was gold-set noise. One arm is also 1/3 the kernel runtime.
#
# Corpus expansion: the corpus previously held 3,200 of the 4,349 labelled studies and only 45
# of the 58 gold studies. Rebuilt to 4,407 studies (+37.8% training data, 58-study gate).
#
# Measured on the 58-study gate (the incumbent re-scored on the SAME gate for a fair compare):
#   incumbent CoAtNet (3,155-study corpus) 0.8923
#   this model       (4,349-study corpus) 0.9054   (+0.0131, better in 92.7% of 2000 bootstraps)
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
ARMS = [
    {
        "file": "raptor_ft_coatnet_v5_full_swa.pt",
        "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "res": 384,
        "w": 1.0,
    },
]

LEGACY_ARM = {
    "file": "raptor_ft_coatnet_v4_full.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "res": 384,
    "k_eval": 42,
    "span_lo": 0.06,
    "span_hi": 0.94,
}

PRIMARY_SPAN_LO = 0.02
PRIMARY_SPAN_HI = 0.98


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    use_cuda = (
        str(device).startswith("cuda")
    )

    def _forward():
        return torch.sigmoid(
            model(x).float()
        )[0].cpu().numpy()

    if use_cuda:
        try:
            with torch.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                return _forward()

        except RuntimeError as error:
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                pass

            print(
                "[DINOsaur V4.5] "
                f"{device} fp16 retry in fp32: "
                f"{type(error).__name__}",
                flush=True,
            )

            return _forward()

    return _forward()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _fill_variant_volume(
    target_volume,
    offset,
    picks,
    pixel_cache,
    files,
    med_ps,
    mm_crop_resize,
):
    arrays = []
    spacings = []

    for position in picks:
        position = min(
            int(position),
            len(files) - 1,
        )
        file_path, spacing = files[
            position
        ]
        arrays.append(
            pixel_cache.get(
                position
            )
        )
        spacings.append(
            spacing
            if spacing > 0
            else med_ps
        )

    valid = [
        array
        for array in arrays
        if array is not None
    ]

    if valid:
        all_pixels = np.concatenate(
            [
                array.ravel()
                for array in valid
            ]
        )
        low, high = np.percentile(
            all_pixels,
            [2.0, 98.0],
        )
    else:
        low, high = 0.0, 1.0

    for local_index, (
        array,
        spacing,
    ) in enumerate(
        zip(
            arrays,
            spacings,
        )
    ):
        output_index = (
            offset
            + local_index
        )

        if (
            output_index
            >= MAXS
        ):
            break

        if array is None:
            continue

        normalized = np.clip(
            (
                array - low
            )
            / (
                high - low
                + 1e-6
            ),
            0,
            1,
        )

        normalized = mm_crop_resize(
            normalized,
            spacing,
        )

        target_volume[
            output_index
        ] = (
            normalized
            * 255
        ).astype(
            np.uint8
        )


def build_study_pair(
    sid,
    ser_records,
    tsdir,
    reader,
):
    """
    Produce exact MaxSpan and legacy-span volumes while reading every required
    DICOM only once. Both checkpoints keep their own percentile normalization.
    """
    (
        order_and_meta,
        read_px,
        mm_crop_resize,
    ) = reader

    rows = ser_records.get(
        sid,
        [],
    )

    primary_volume = np.zeros(
        (
            MAXS,
            IMG,
            IMG,
        ),
        np.uint8,
    )
    legacy_volume = np.zeros_like(
        primary_volume
    )

    used = set()
    offset = 0

    for plane, fluid, count in SLOTS:
        record = _pick_series_for_slot(
            rows,
            plane,
            fluid,
            used,
        )

        if record is None:
            offset += count
            continue

        used.add(
            record[
                "SeriesInstanceUID"
            ]
        )

        files, med_ps = order_and_meta(
            f"{tsdir}/{sid}/"
            f"{record['SeriesInstanceUID']}"
        )

        if not files:
            offset += count
            continue

        number = len(files)

        primary_low = int(
            number
            * PRIMARY_SPAN_LO
        )
        primary_high = int(
            number
            * PRIMARY_SPAN_HI
        ) - 1
        primary_high = max(
            primary_high,
            primary_low,
        )

        legacy_low = int(
            number
            * float(
                LEGACY_ARM[
                    "span_lo"
                ]
            )
        )
        legacy_high = int(
            number
            * float(
                LEGACY_ARM[
                    "span_hi"
                ]
            )
        ) - 1
        legacy_high = max(
            legacy_high,
            legacy_low,
        )

        if number > 1:
            primary_picks = np.linspace(
                primary_low,
                primary_high,
                count,
            ).round().astype(int)

            legacy_picks = np.linspace(
                legacy_low,
                legacy_high,
                count,
            ).round().astype(int)
        else:
            primary_picks = np.zeros(
                count,
                dtype=int,
            )
            legacy_picks = np.zeros(
                count,
                dtype=int,
            )

        required_positions = sorted(
            set(
                primary_picks.tolist()
                + legacy_picks.tolist()
            )
        )

        pixel_cache = {}

        for position in required_positions:
            position = min(
                int(position),
                number - 1,
            )

            file_path, _ = files[
                position
            ]

            try:
                pixel_cache[
                    position
                ] = read_px(
                    file_path
                )
            except Exception:
                pixel_cache[
                    position
                ] = None

        _fill_variant_volume(
            primary_volume,
            offset,
            primary_picks,
            pixel_cache,
            files,
            med_ps,
            mm_crop_resize,
        )

        _fill_variant_volume(
            legacy_volume,
            offset,
            legacy_picks,
            pixel_cache,
            files,
            med_ps,
            mm_crop_resize,
        )

        offset += count

        if offset >= MAXS:
            break

    primary_mask = (
        primary_volume.reshape(
            MAXS,
            -1,
        ).sum(1)
        > 0
    ).astype(
        np.uint8
    )

    legacy_mask = (
        legacy_volume.reshape(
            MAXS,
            -1,
        ).sum(1)
        > 0
    ).astype(
        np.uint8
    )

    return (
        primary_volume,
        primary_mask,
        legacy_volume,
        legacy_mask,
    )


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(
    fname,
    required=True,
):
    direct = [
        f"/kaggle/input/raptor-knee-arms/{fname}",
        f"/kaggle/input/raptor-knee-arms/1/{fname}",
        f"/kaggle/input/raptor-cnn336/{fname}",
    ]

    for path in direct:
        if os.path.exists(path):
            return path

    for directory in sorted(
        glob.glob(
            "/kaggle/input/*/"
        )
    ):
        if (
            "competition"
            in directory.lower()
        ):
            continue

        hits = glob.glob(
            os.path.join(
                directory,
                "**",
                fname,
            ),
            recursive=True,
        )

        if hits:
            return hits[0]

    if required:
        raise RuntimeError(
            f"{fname} not found "
            "under /kaggle/input"
        )

    return None



def _d45_find_swin_checkpoint():
    roots = [
        "/kaggle/input/raptor-knee-arms",
        "/kaggle/input/raptor-knee-arms/1",
        "/kaggle/input/raptor-cnn336",
    ]

    candidates = []

    for root in roots:
        if not os.path.isdir(root):
            continue

        for pattern in (
            "**/*swin*.pt",
            "**/*swin*.pth",
            "**/*swin*.bin",
        ):
            candidates.extend(
                glob.glob(
                    os.path.join(
                        root,
                        pattern,
                    ),
                    recursive=True,
                )
            )

    seen = set()

    for path in sorted(candidates):
        if path in seen:
            continue

        seen.add(path)

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            architecture = str(
                checkpoint.get(
                    "arch",
                    "",
                )
            )

            resolution = int(
                checkpoint.get(
                    "res",
                    384,
                )
            )

            has_model = (
                isinstance(
                    checkpoint,
                    dict,
                )
                and "model" in checkpoint
            )

            del checkpoint

            if (
                has_model
                and "swin" in architecture.lower()
            ):
                return {
                    "path": path,
                    "arch": architecture,
                    "res": resolution,
                }

        except Exception:
            continue

    return None


def _d45_run_sparse_swin(
    arm,
    test_ids,
    series_map,
    series_dir,
    reader,
):
    if arm is None:
        return None, 0.0

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
    ):
        return None, 0.0

    device = torch.device(
        "cuda:1"
        if torch.cuda.device_count() >= 2
        else "cuda:0"
    )

    model = None

    try:
        model, resolution = load_model(
            arm["path"],
            arm["arch"],
            arm["res"],
            device,
        )

        prediction = np.full(
            (
                len(test_ids),
                len(LAB),
            ),
            0.5,
            dtype=np.float32,
        )

        success = np.zeros(
            len(test_ids),
            dtype=np.bool_,
        )

        for study_index, study_id in enumerate(
            test_ids
        ):
            try:
                (
                    _primary_volume,
                    _primary_mask,
                    legacy_volume,
                    legacy_mask,
                ) = build_study_pair(
                    study_id,
                    series_map,
                    series_dir,
                    reader,
                )

                windows = eval_windows(
                    legacy_volume,
                    legacy_mask,
                    k=24,
                    res=resolution,
                    norm=NORM,
                )

                prediction[
                    study_index
                ] = infer_probs(
                    model,
                    windows,
                    device,
                )

                success[
                    study_index
                ] = True

                del (
                    _primary_volume,
                    _primary_mask,
                    legacy_volume,
                    legacy_mask,
                    windows,
                )

            except Exception as error:
                print(
                    "[DINOsaur V4.5] "
                    f"Swin study {study_index} fallback: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

            if (
                (study_index + 1) % 150 == 0
                or study_index + 1 == len(test_ids)
            ):
                print(
                    "[DINOsaur V4.5] "
                    f"Swin {study_index+1}/{len(test_ids)}",
                    flush=True,
                )

        fraction = float(
            success.mean()
        )

        if fraction < 0.97:
            return None, fraction

        return prediction, fraction

    except Exception as error:
        print(
            "[DINOsaur V4.5] "
            "optional Swin disabled safely: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return None, 0.0

    finally:
        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def _d53_discover_aux_arms():
    roots = []
    for pattern in (
        "/kaggle/input/raptor-knee-arms*",
        "/kaggle/input/*raptor*arms*",
        "/kaggle/input/*raptor*",
    ):
        roots.extend(glob.glob(pattern))
    roots = [p for p in sorted(set(roots)) if os.path.isdir(p)]
    if not roots:
        print('[DINOsaur V5.3] no Raptor architecture-diversity dataset attached', flush=True)
        return []

    candidates = []
    for root in roots:
        for pattern in ("**/*.pt", "**/*.pth", "**/*.bin"):
            candidates.extend(glob.glob(os.path.join(root, pattern), recursive=True))

    def classify(arch):
        low = arch.lower()
        if 'swin' in low:
            return 'swin'
        if any(k in low for k in ('efficientnetv2', 'effv2', 'tf_efficientnetv2')):
            return 'effv2'
        return None

    def priority(family, arch, resolution, path):
        low = (arch + ' ' + os.path.basename(path)).lower()
        score = 0.0
        if family == 'swin':
            score += 4.0 if 'base' in low else 0.0
            score += 2.0 if resolution >= 384 else 0.0
            score += 1.0 if 'v2' in low else 0.0
        elif family == 'effv2':
            score += 5.0 if any(k in low for k in ('effv2l', 'efficientnetv2_l', 'large')) else 0.0
            score += 2.0 if resolution >= 448 else 0.0
        score += min(float(resolution), 512.0) / 1024.0
        return score

    best = {'swin': None, 'effv2': None}
    seen = set()
    for path in sorted(candidates):
        if path in seen:
            continue
        seen.add(path)
        name = os.path.basename(path).lower()
        if 'coatnet' in name or 'maxspan' in name:
            continue
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
                del checkpoint
                continue
            architecture = str(checkpoint.get('arch', ''))
            resolution = int(checkpoint.get('res', 384))
            family = classify(architecture)
            del checkpoint
            if family is None:
                continue
            item = {
                'family': family,
                'path': path,
                'arch': architecture,
                'res': resolution,
                'priority': priority(family, architecture, resolution, path),
            }
            if best[family] is None or item['priority'] > best[family]['priority']:
                best[family] = item
        except Exception:
            continue

    arms = [best[k] for k in ('swin', 'effv2') if best[k] is not None]
    for arm in arms:
        print(
            '[DINOsaur V5.3] discovered '
            f"{arm['family']}: {os.path.basename(arm['path'])} | "
            f"{arm['arch']} | res={arm['res']}",
            flush=True,
        )
    return arms


def _d53_run_aux_arm(arm, test_ids, series_map, series_dir, reader, device):
    if arm is None or not torch.cuda.is_available():
        return None, 0.0
    family = str(arm['family'])
    k_eval = 36 if family == 'swin' else 30
    model = None
    try:
        model, resolution = load_model(
            arm['path'], arm['arch'], arm['res'], device,
        )
        prediction = np.full((len(test_ids), len(LAB)), 0.5, dtype=np.float32)
        success = np.zeros(len(test_ids), dtype=np.bool_)
        for study_index, study_id in enumerate(test_ids):
            try:
                primary_volume, primary_mask, legacy_volume, legacy_mask = build_study_pair(
                    study_id, series_map, series_dir, reader,
                )
                windows = eval_windows(
                    legacy_volume,
                    legacy_mask,
                    k=k_eval,
                    res=resolution,
                    norm=NORM,
                )
                prediction[study_index] = infer_probs(model, windows, device)
                success[study_index] = True
                del primary_volume, primary_mask, legacy_volume, legacy_mask, windows
            except Exception as error:
                print(
                    f"[DINOsaur V5.3] {family} study {study_index} fallback: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            if (study_index + 1) % 150 == 0 or study_index + 1 == len(test_ids):
                print(
                    f"[DINOsaur V5.3] {family} {study_index+1}/{len(test_ids)}",
                    flush=True,
                )
        fraction = float(success.mean())
        if fraction < 0.98:
            print(
                f"[DINOsaur V5.3] {family} coverage {fraction:.3f}; arm rejected",
                flush=True,
            )
            return None, fraction
        return prediction, fraction
    except Exception as error:
        print(
            f"[DINOsaur V5.3] {family} disabled safely: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return None, 0.0
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                torch.cuda.empty_cache()


def _d53_family_core(base_rank, aux_ranks):
    base_rank = np.asarray(base_rank, np.float64)
    if not aux_ranks:
        return base_rank.copy(), {}

    protected = {
        'ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Lateral OA', 'Fracture'
    }
    diagnostics = {}
    output = base_rank.copy()

    for j, target in enumerate(LAB):
        present = [f for f in ('swin', 'effv2') if f in aux_ranks]
        if not present:
            continue

        if set(present) == {'swin', 'effv2'}:
            if target in protected:
                weights = {'base': 0.80, 'swin': 0.13, 'effv2': 0.07}
            else:
                weights = {'base': 0.65, 'swin': 0.23, 'effv2': 0.12}
        elif present == ['swin']:
            weights = {'base': 0.86 if target in protected else 0.76, 'swin': 0.14 if target in protected else 0.24}
        else:
            weights = {'base': 0.91 if target in protected else 0.85, 'effv2': 0.09 if target in protected else 0.15}

        mixed = weights['base'] * base_rank[:, j]
        corr = {}
        for family in present:
            mixed += weights[family] * aux_ranks[family][:, j]
            value = float(np.corrcoef(base_rank[:, j], aux_ranks[family][:, j])[0, 1])
            corr[family] = value if np.isfinite(value) else None
        output[:, j] = mixed
        diagnostics[target] = {'weights': weights, 'corr_to_base': corr}

    return rankpct(output), diagnostics


def _d53_move_toward(base_rank, family_rank, strength):
    return rankpct(
        (1.0 - float(strength)) * np.asarray(base_rank, np.float64)
        + float(strength) * np.asarray(family_rank, np.float64)
    )

def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()

    number_studies = len(
        test_ids
    )

    primary_predictions = np.full(
        (
            number_studies,
            len(LAB),
        ),
        0.5,
        np.float32,
    )

    legacy_predictions = np.full_like(
        primary_predictions,
        0.5,
    )

    legacy_success = np.zeros(
        number_studies,
        dtype=np.bool_,
    )

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 1
    ):
        primary_device = torch.device(
            "cuda:0"
        )
    else:
        primary_device = torch.device(
            "cpu"
        )

    legacy_path = find_weight_file(
        LEGACY_ARM[
            "file"
        ],
        required=False,
    )

    legacy_enabled = (
        legacy_path is not None
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    )

    primary_path = find_weight_file(
        ARMS[0][
            "file"
        ],
        required=True,
    )

    primary_model, primary_res = load_model(
        primary_path,
        ARMS[0][
            "arch"
        ],
        ARMS[0][
            "res"
        ],
        primary_device,
    )

    print(
        "[DINOsaur V4.5] primary "
        f"{ARMS[0]['file']} "
        f"on {primary_device}",
        flush=True,
    )

    legacy_model = None
    legacy_device = None
    legacy_res = None

    if legacy_enabled:
        legacy_device = torch.device(
            "cuda:1"
        )

        try:
            legacy_model, legacy_res = load_model(
                legacy_path,
                LEGACY_ARM[
                    "arch"
                ],
                LEGACY_ARM[
                    "res"
                ],
                legacy_device,
            )

            print(
                "[DINOsaur V4.5] complement "
                f"{LEGACY_ARM['file']} "
                f"on {legacy_device}",
                flush=True,
            )

        except Exception as error:
            legacy_enabled = False
            legacy_model = None

            print(
                "[DINOsaur V4.5] "
                "legacy checkpoint disabled "
                f"safely: "
                f"{type(error).__name__}: "
                f"{error}",
                flush=True,
            )

    else:
        print(
            "[DINOsaur V4.5] "
            "legacy complement unavailable "
            "or second GPU absent; "
            "exact 0.935 Raptor retained",
            flush=True,
        )

    executor = (
        ThreadPoolExecutor(
            max_workers=2
        )
        if legacy_enabled
        else None
    )

    for study_index, study_id in enumerate(
        test_ids
    ):
        try:
            (
                primary_volume,
                primary_mask,
                legacy_volume,
                legacy_mask,
            ) = build_study_pair(
                study_id,
                SER,
                tsdir,
                reader,
            )

            primary_windows = eval_windows(
                primary_volume,
                primary_mask,
                k=K_EVAL,
                res=primary_res,
                norm=NORM,
            )

            if legacy_enabled:
                legacy_windows = eval_windows(
                    legacy_volume,
                    legacy_mask,
                    k=int(
                        LEGACY_ARM[
                            "k_eval"
                        ]
                    ),
                    res=legacy_res,
                    norm=NORM,
                )

                primary_future = executor.submit(
                    infer_probs,
                    primary_model,
                    primary_windows,
                    primary_device,
                )

                legacy_future = executor.submit(
                    infer_probs,
                    legacy_model,
                    legacy_windows,
                    legacy_device,
                )

                primary_prediction = (
                    primary_future.result()
                )

                try:
                    legacy_prediction = (
                        legacy_future.result()
                    )
                    legacy_success[
                        study_index
                    ] = True
                except Exception as legacy_error:
                    legacy_prediction = (
                        primary_prediction.copy()
                    )

                    print(
                        "[DINOsaur V4.5] "
                        f"legacy study "
                        f"{study_index} fallback: "
                        f"{type(legacy_error).__name__}: "
                        f"{legacy_error}",
                        flush=True,
                    )

                del legacy_windows

            else:
                primary_prediction = infer_probs(
                    primary_model,
                    primary_windows,
                    primary_device,
                )
                legacy_prediction = (
                    primary_prediction.copy()
                )

            primary_predictions[
                study_index
            ] = primary_prediction

            legacy_predictions[
                study_index
            ] = legacy_prediction

            del (
                primary_volume,
                primary_mask,
                legacy_volume,
                legacy_mask,
                primary_windows,
                primary_prediction,
                legacy_prediction,
            )

        except Exception as error:
            print(
                "[DINOsaur V4.5] "
                f"study {study_index} "
                f"{study_id[:16]} FALLBACK "
                f"({type(error).__name__}: "
                f"{error})",
                flush=True,
            )

        if (
            (
                study_index + 1
            )
            % 100
            == 0
            or study_index + 1
            == number_studies
        ):
            print(
                "[DINOsaur V4.5] "
                f"{study_index+1}/"
                f"{number_studies} | "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )

    if executor is not None:
        executor.shutdown(
            wait=True
        )

    del primary_model

    if legacy_model is not None:
        del legacy_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    primary_rank = rankpct(
        np.clip(
            primary_predictions,
            0,
            1,
        )
    )

    raptor_rank = (
        primary_rank.copy()
    )

    legacy_fraction = float(
        legacy_success.mean()
    ) if legacy_enabled else 0.0

    if (
        legacy_enabled
        and legacy_fraction >= 0.98
    ):
        legacy_rank = rankpct(
            np.clip(
                legacy_predictions,
                0,
                1,
            )
        )

        # The expanded MaxSpan checkpoint's documented largest gains are ACL,
        # both menisci, Lateral OA and Fracture. Keep it almost pure there.
        # Use the previous 0.934 checkpoint only for the remaining findings,
        # where it may restore complementary ordering.
        complement_weight = {
            "ACL": 0.025,
            "MCL": 0.16,
            "Medial Meniscus": 0.025,
            "Lateral Meniscus": 0.020,
            "Medial OA": 0.10,
            "Lateral OA": 0.025,
            "PF OA": 0.12,
            "Effusion": 0.10,
            "Synovitis": 0.16,
            "Baker's": 0.12,
            "Contusion": 0.12,
            "Fracture": 0.020,
        }

        complement_log = []

        for target_index, target in enumerate(
            LAB
        ):
            weight = float(
                complement_weight.get(
                    target,
                    0.0,
                )
            )

            if weight <= 0:
                continue

            correlation = float(
                np.corrcoef(
                    primary_rank[
                        :,
                        target_index,
                    ],
                    legacy_rank[
                        :,
                        target_index,
                    ],
                )[0, 1]
            )

            if not np.isfinite(
                correlation
            ):
                weight = 0.0
            elif correlation > 0.992:
                weight *= 0.50
            elif correlation < 0.65:
                weight *= 0.40

            if weight <= 0:
                continue

            raptor_rank[
                :,
                target_index,
            ] = (
                (
                    1.0
                    - weight
                )
                * primary_rank[
                    :,
                    target_index,
                ]
                + weight
                * legacy_rank[
                    :,
                    target_index,
                ]
            )

            complement_log.append(
                (
                    target,
                    weight,
                    correlation,
                )
            )

        raptor_rank = rankpct(
            raptor_rank
        )

        print(
            "[DINOsaur V4.5] "
            "legacy complement: "
            + "; ".join(
                f"{target}=w{weight:.3f},"
                f"corr={correlation:.3f}"
                for (
                    target,
                    weight,
                    correlation,
                )
                in complement_log
            ),
            flush=True,
        )

    else:
        print(
            "[DINOsaur V4.5] "
            f"legacy success={legacy_fraction:.3f}; "
            "exact primary Raptor used",
            flush=True,
        )

    ranks = raptor_rank

    # Optional third architecture. It is intentionally only a small residual.
    # If the checkpoint is absent, incompatible, too slow, or incomplete,
    # this block becomes a strict no-op and preserves the 0.936 anchor.
    swin_arm = _d45_find_swin_checkpoint()

    if swin_arm is not None:
        print(
            "[DINOsaur V4.5] optional Swin found: "
            f"{os.path.basename(swin_arm['path'])} | "
            f"{swin_arm['arch']} | res={swin_arm['res']}",
            flush=True,
        )

        swin_prediction, swin_fraction = _d45_run_sparse_swin(
            swin_arm,
            test_ids,
            SER,
            tsdir,
            reader,
        )

        if (
            swin_prediction is not None
            and swin_fraction >= 0.97
        ):
            swin_rank = rankpct(
                np.clip(
                    swin_prediction,
                    0,
                    1,
                )
            )

            diversity_weight = {
                "ACL": 0.035,
                "MCL": 0.075,
                "Medial Meniscus": 0.045,
                "Lateral Meniscus": 0.040,
                "Medial OA": 0.065,
                "Lateral OA": 0.045,
                "PF OA": 0.070,
                "Effusion": 0.060,
                "Synovitis": 0.080,
                "Baker's": 0.070,
                "Contusion": 0.070,
                "Fracture": 0.040,
            }

            swin_log = []

            for target_index, target in enumerate(
                LAB
            ):
                weight = float(
                    diversity_weight[
                        target
                    ]
                )

                correlation = float(
                    np.corrcoef(
                        ranks[
                            :,
                            target_index,
                        ],
                        swin_rank[
                            :,
                            target_index,
                        ],
                    )[0, 1]
                )

                if not np.isfinite(
                    correlation
                ):
                    weight = 0.0
                elif correlation > 0.985:
                    weight *= 0.45
                elif correlation < 0.50:
                    weight *= 0.35

                if weight <= 0:
                    continue

                ranks[
                    :,
                    target_index,
                ] = (
                    (
                        1.0
                        - weight
                    )
                    * ranks[
                        :,
                        target_index,
                    ]
                    + weight
                    * swin_rank[
                        :,
                        target_index,
                    ]
                )

                swin_log.append(
                    (
                        target,
                        weight,
                        correlation,
                    )
                )

            ranks = rankpct(
                ranks
            )

            print(
                "[DINOsaur V4.5] Swin residual: "
                + "; ".join(
                    f"{target}=w{weight:.3f},corr={correlation:.3f}"
                    for (
                        target,
                        weight,
                        correlation,
                    )
                    in swin_log
                ),
                flush=True,
            )

    anchor_ranks = rankpct(np.asarray(ranks, np.float64))
    family_ranks = anchor_ranks.copy()
    aux_ranks = {}
    aux_info = {}

    aux_arms = _d53_discover_aux_arms()
    if aux_arms:
        for arm_index, arm in enumerate(aux_arms):
            device_index = arm_index % max(torch.cuda.device_count(), 1)
            device = torch.device(f'cuda:{device_index}') if torch.cuda.is_available() else torch.device('cpu')
            pred, fraction = _d53_run_aux_arm(
                arm, test_ids, SER, tsdir, reader, device,
            )
            aux_info[arm['family']] = {
                'file': os.path.basename(arm['path']),
                'arch': arm['arch'],
                'res': int(arm['res']),
                'coverage': float(fraction),
            }
            if pred is not None and fraction >= 0.98:
                aux_ranks[arm['family']] = rankpct(np.clip(pred, 0, 1))

    family_diag = {}
    if aux_ranks:
        family_ranks, family_diag = _d53_family_core(raptor_rank, aux_ranks)
        safe_ranks = _d53_move_toward(anchor_ranks, family_ranks, 0.25)
        main_ranks = _d53_move_toward(anchor_ranks, family_ranks, 0.45)
        probe_ranks = _d53_move_toward(anchor_ranks, family_ranks, 0.70)
        ranks = main_ranks
        print(
            '[DINOsaur V5.3] tri-family active: '
            + ', '.join(sorted(aux_ranks))
            + ' | main strength=0.45',
            flush=True,
        )
    else:
        safe_ranks = anchor_ranks.copy()
        main_ranks = anchor_ranks.copy()
        probe_ranks = anchor_ranks.copy()
        ranks = anchor_ranks.copy()
        print('[DINOsaur V5.3] no complete aux arm; exact V4.5 Raptor anchor retained', flush=True)

    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    def _d53_write_raptor(values, filename):
        frame = pd.DataFrame(np.asarray(values, np.float32), columns=LAB)
        frame.insert(0, 'StudyInstanceUID', test_ids)
        frame = frame[sub_cols]
        assert list(frame.columns) == sub_cols, 'column order drift'
        assert frame['StudyInstanceUID'].tolist() == test_ids, 'row identity drift'
        assert np.isfinite(frame[LAB].values).all()
        path = os.path.join('/kaggle/working', filename)
        frame.to_csv(path, index=False)
        return path

    _d53_write_raptor(anchor_ranks, 'submission_raptor_v45_anchor.csv')
    _d53_write_raptor(safe_ranks, 'submission_raptor_v53_safe.csv')
    _d53_write_raptor(main_ranks, 'submission_raptor_v53_main.csv')
    _d53_write_raptor(probe_ranks, 'submission_raptor_v53_probe.csv')
    out = _d53_write_raptor(main_ranks, 'submission_coatnet.csv')

    diag_payload = {
        'version': 'DINOsaur V5.3 RaptorTriFamily',
        'aux_arms': aux_info,
        'family_diagnostics': family_diag,
        'variant_strength': {'safe': 0.25, 'main': 0.45, 'probe': 0.70},
        'default': 'main',
    }
    with open('/kaggle/working/dinosaur_v53_raptor_family.json', 'w') as handle:
        json.dump(diag_payload, handle, indent=2, sort_keys=True)

    print('wrote', out, '|', len(test_ids), 'rows x', len(sub_cols), 'cols', flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _coat_exc:
        import traceback as _coat_traceback
        print(f"CoAtNet branch failed; retaining transformer submission: {type(_coat_exc).__name__}: {_coat_exc}", flush=True)
        _coat_traceback.print_exc()



# Hidden-rerun fail-safe final fusion.
from pathlib import Path as _D42Path
import shutil as _d42_shutil

_d42_primary = _D42Path('/kaggle/working/submission.csv')
_d42_backup = _D42Path('/kaggle/working/.d53_transformer_backup.csv')
if _d42_primary.is_file():
    _d42_shutil.copy2(_d42_primary, _d42_backup)

try:
    from pathlib import Path as _BlendPath
    import numpy as _blend_np
    import pandas as _blend_pd

    _blend_work = _BlendPath('/kaggle/working')
    _blend_transformer_path = _blend_work / 'submission.csv'
    _blend_transformer = _blend_pd.read_csv(
        _blend_transformer_path, dtype={'StudyInstanceUID': str}
    )
    _blend_labels = [c for c in _blend_transformer.columns if c != 'StudyInstanceUID']
    _blend_tr = _blend_transformer[_blend_labels].rank(method='average', pct=True)

    def _d53_fuse(raptor_path):
        raptor = _blend_pd.read_csv(raptor_path, dtype={'StudyInstanceUID': str})
        if raptor.columns.tolist() != _blend_transformer.columns.tolist():
            raise RuntimeError(f'Raptor/transformer schema mismatch: {raptor_path.name}')
        if raptor['StudyInstanceUID'].tolist() != _blend_transformer['StudyInstanceUID'].tolist():
            raise RuntimeError(f'Raptor/transformer study order mismatch: {raptor_path.name}')
        rr = raptor[_blend_labels].rank(method='average', pct=True)
        output = _blend_transformer.copy()
        # H-38 is a conservative residual ablation after H-37's 50/50 blend
        # scored 0.922 versus H-36's 0.928. Keep 80% of the CoAtNet arm and
        # add only a 20% DINOv3 residual. The public V5.3 source has adaptive
        # weights inherited from an older calibrated family; those are
        # intentionally disabled so the three-study smoke cannot silently
        # change the experimental question.
        dino_weight = 0.20
        coatnet_weight = 0.80
        weight = {label: coatnet_weight for label in _blend_labels}
        for label in _blend_labels:
            cw = float(weight[label])
            output[label] = dino_weight * _blend_tr[label] + cw * rr[label]
        output[_blend_labels] = output[_blend_labels].rank(method='average', pct=True)
        values = output[_blend_labels].to_numpy(_blend_np.float64)
        if not _blend_np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise RuntimeError(f'invalid fused values: {raptor_path.name}')
        return output, weight

    variants = {
        'v45_anchor': _blend_work / 'submission_raptor_v45_anchor.csv',
        'v53_safe': _blend_work / 'submission_raptor_v53_safe.csv',
        'v53_main': _blend_work / 'submission_raptor_v53_main.csv',
        'v53_probe': _blend_work / 'submission_raptor_v53_probe.csv',
    }
    fused = {}
    for name, path in variants.items():
        if not path.is_file():
            continue
        frame, weights = _d53_fuse(path)
        fused[name] = frame
        frame.to_csv(_blend_work / f'submission_{name}.csv', index=False)

    if 'v53_main' in fused:
        fused['v53_main'].to_csv(_blend_transformer_path, index=False)
        print(
            f"final submission.csv = DINOsaur V5.3 Raptor Tri-Family; {fused['v53_main'].shape}",
            flush=True,
        )
    elif 'v45_anchor' in fused:
        fused['v45_anchor'].to_csv(_blend_transformer_path, index=False)
        print('[DINOsaur V5.3] aux unavailable; exact V4.5 fused anchor restored', flush=True)
    else:
        print('[DINOsaur V5.3] Raptor outputs unavailable; transformer submission retained', flush=True)

except Exception as _d42_error:
    import traceback as _d42_traceback
    print(
        '[DINOsaur V5.3] final fusion failed; restoring calibrated transformer submission: '
        f'{type(_d42_error).__name__}: {_d42_error}',
        flush=True,
    )
    _d42_traceback.print_exc()
    if _d42_backup.is_file():
        _d42_shutil.copy2(_d42_backup, _d42_primary)
finally:
    try:
        if _d42_backup.is_file():
            _d42_backup.unlink()
    except OSError:
        pass

if not _d42_primary.is_file():
    raise RuntimeError('submission.csv missing after V5.3 fail-safe')
