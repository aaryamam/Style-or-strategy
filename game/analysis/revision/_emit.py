"""Shared result collector. Each task module returns a dict; run_all merges them
into analysis/revision/revision_numbers.json. Also common helpers."""
import os, sys, json
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)

OUT = "analysis/revision/revision_numbers.json"


def load_all():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {}


def merge(d):
    cur = load_all()
    cur.update(d)
    json.dump(cur, open(OUT, "w"), indent=1)
    return cur
