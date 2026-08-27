from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

EXPECTED_HASHES = {
    'compiler_block0.pt': '78fd7f52ef6f019f6ede72c73b4928b0482d61e7f7914de532ebc084779fce56',
    'compiler_block1.pt': 'd70098af827694a42e7bb31958cf9959ec3aceef0d432941960c15d0cb5091c8',
}
SEED = 20260827
SHAPE = (2, 48, 24)
torch.set_num_threads(1)

class CausalBlock(nn.Module):
    def __init__(self, d=24, heads=4, mlp=24):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        z = self.n1(x)
        a, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))

def tensor_hash(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode() + b'\0')
        h.update(str(t.dtype).encode() + b'\0')
        h.update(str(tuple(t.shape)).encode() + b'\0')
        h.update(t.numpy().tobytes(order='C'))
    return h.hexdigest()

def safe_name(k: str) -> str:
    return k.replace('.', '__') + '.npy'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.out_dir / 'raw'
    raw_root.mkdir(exist_ok=True)

    states = []
    observed_hashes = {}
    block_payloads = []
    block_serialized = []
    key_files = []
    for i in range(2):
        name = f'compiler_block{i}.pt'
        state = torch.load(args.source_dir / name, map_location='cpu', weights_only=True)
        observed_hashes[name] = tensor_hash(state)
        if observed_hashes[name] != EXPECTED_HASHES[name]:
            raise SystemExit(f'tensor hash mismatch for {name}')
        states.append(state)
        bdir = raw_root / f'block{i}'
        bdir.mkdir(exist_ok=True)
        mapping = {}
        payload = 0
        serialized = 0
        for k, v in state.items():
            arr = v.detach().cpu().contiguous().numpy()
            fn = safe_name(k)
            np.save(bdir / fn, arr, allow_pickle=False)
            mapping[k] = fn
            payload += int(arr.nbytes)
            serialized += int((bdir / fn).stat().st_size)
        key_files.append(mapping)
        block_payloads.append(payload)
        block_serialized.append(serialized)

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(SHAPE, dtype=np.float32)
    np.save(args.out_dir / 'input.npy', x, allow_pickle=False)
    xt = torch.from_numpy(x.copy())
    with torch.no_grad():
        for state in states:
            b = CausalBlock()
            b.load_state_dict(state)
            b.eval()
            xt = b(xt)
    reference = xt.detach().cpu().numpy()
    np.save(args.out_dir / 'pytorch_reference.npy', reference, allow_pickle=False)

    manifest = {
        'format': 'canaria-s4-numpy-stream-v1',
        'input_shape': list(SHAPE),
        'input_seed': SEED,
        'dtype': 'float32',
        'block_tensor_content_sha256': observed_hashes,
        'block_tensor_payload_bytes': block_payloads,
        'one_block_max_tensor_payload_bytes': max(block_payloads),
        'block_raw_npy_serialized_bytes': block_serialized,
        'key_files': key_files,
        'reference_output_shape': list(reference.shape),
        'reference_output_sum': float(reference.sum()),
    }
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))

if __name__ == '__main__':
    main()
