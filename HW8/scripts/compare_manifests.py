"""Fail when core output hashes differ between CI operating systems/Python versions."""
import json, sys
from pathlib import Path

root=Path(sys.argv[1] if len(sys.argv)>1 else "ci_artifacts")
paths=sorted(root.glob("**/reproducibility_manifest.json"))
if len(paths)<2: raise SystemExit(f"Expected at least 2 manifests, found {len(paths)}")
items=[(p,json.loads(p.read_text(encoding="utf-8"))) for p in paths]
base_path,base=items[0]
failed=False
for path,item in items[1:]:
    keys=sorted(set(base["files"])|set(item["files"]))
    diffs=[k for k in keys if base["files"].get(k)!=item["files"].get(k)]
    if diffs:
        failed=True; print(f"MISMATCH {base_path} vs {path}: {', '.join(diffs)}")
    else: print(f"MATCH {base_path} vs {path}")
if failed: raise SystemExit(1)
print(f"All {len(items)} cross-platform manifests have identical core file hashes.")
