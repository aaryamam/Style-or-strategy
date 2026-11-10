"""Provenance helper. This repo is NOT a git checkout (verified: `git rev-parse`
fails), so a git hash is unavailable. We fingerprint the repo state instead:
sha256 over the pipeline source files, plus a manifest (path,size,mtime) of every
input data file. Import `STATE` for the fingerprint dict and `stamp()` for a
per-value provenance stub."""
import os, sys, json, hashlib, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SRC = ["signature.py", "study.py", "seller_exp.py", "game_lib.py",
        "metric_concession.py", "policy_shift.py", "freepass.py",
        "fh_analyze.py", "b_analyze.py", "zephyr_gates.py", "loco_locus.py",
        "sellerD_adaptation_value.py"]

_DATA_GLOBS = ["runs/n500/*.jsonl", "runs/n2000/*.jsonl", "runs/seller_exp/*.jsonl",
               "results/forced_horizon/*.jsonl", "results/base_artifact/*.jsonl",
               "results/realized_robust/*.jsonl", "results/realized_probe/*.jsonl"]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_state():
    src = {}
    combined = hashlib.sha256()
    for name in _SRC:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            d = _sha(p)
            src[name] = d
            combined.update(d.encode())
    data = []
    for g in _DATA_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            st = os.stat(p)
            data.append({"path": os.path.relpath(p, ROOT),
                         "size": st.st_size, "mtime": int(st.st_mtime)})
    return {"git": None, "git_note": "not a git repo; content-hash fingerprint used",
            "source_sha256": src,
            "source_combined_sha256": combined.hexdigest(),
            "n_data_files": len(data), "data_manifest": data}


STATE = build_state()
FINGERPRINT = STATE["source_combined_sha256"]


def stamp(source_script, source_data, seed=None, notes=None):
    d = {"source_script": source_script, "source_data": source_data,
         "repo_fingerprint": FINGERPRINT}
    if seed is not None:
        d["seed"] = seed
    if notes:
        d["notes"] = notes
    return d


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "analysis/revision"), exist_ok=True)
    json.dump(STATE, open(os.path.join(ROOT, "analysis/revision/repo_state.json"), "w"),
              indent=1)
    print("combined source sha256:", FINGERPRINT)
    print("n data files:", STATE["n_data_files"])
