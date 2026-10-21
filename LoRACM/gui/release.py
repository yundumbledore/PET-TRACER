"""Reopen projects from a new download without changing their scientific inputs."""
import json
from pathlib import Path
from common import file_sha256
from runtime import write_json_atomic

CONFIGS=('protocol.json','cm_protocol.json','train.json','train_cm.json','gui_state.json')
BUNDLED={'data','checkpoints','paper','examples'}


def sync_release(root,workspace):
    root=Path(root).resolve();workspace=Path(workspace)
    marker=workspace/'toolbox_location.json'
    previous=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
    old=Path(previous.get('root',str(root)))
    hashes=dict(previous.get('bundled_sha256',{}));memo={}
    def digest(path):
        name=str(path)
        if name not in memo:memo[name]=file_sha256(path)
        return memo[name]
    def visit(value):
        if isinstance(value,dict):return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        if not isinstance(value,str):return value
        path=Path(value)
        if not path.is_absolute():return value
        path=path.resolve()
        try:relative=path.relative_to(old)
        except ValueError:return value
        if not relative.parts or relative.parts[0] not in BUNDLED:return value
        key=relative.as_posix();candidate=root/relative
        expected=hashes.get(key)
        if expected is None and path.is_file():expected=digest(path);hashes[key]=expected
        # Only relocate to byte-identical shipped data/model files. A modified
        # custom model or new scientific asset never silently replaces an input.
        if root!=old and expected and candidate.is_file() and digest(candidate)==expected:return str(candidate)
        return value
    for name in CONFIGS:
        target=workspace/name
        if not target.exists():continue
        before=json.loads(target.read_text(encoding='utf-8'));after=visit(before)
        if after!=before:write_json_atomic(target,after)
    updated={'root':str(root),'bundled_sha256':hashes}
    if updated!=previous:write_json_atomic(marker,updated)

