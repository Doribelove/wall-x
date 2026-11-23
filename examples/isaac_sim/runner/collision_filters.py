"""Measured same-chain collision exclusions for the EX001 proxy."""
import hashlib
import json
import xml.etree.ElementTree as ET

PAIR_MANIFEST_SHA='a942eea38a09f10f600f3cb610c1c1f0f288334a51b3db33fc6e010aed369a39'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_measured_ancestor_filters(stage, bodies, root_path, source_urdf, evidence, UsdPhysics):
    """Keep all other imported and stage pairs unaltered; never disable self collision."""
    if sha(evidence)!=PAIR_MANIFEST_SHA:
        raise RuntimeError("unexpected collision exclusion evidence bytes")
    audit=json.loads(evidence.read_text())
    if audit.get("exact_proxy_urdf_sha256")!=sha(source_urdf):
        raise RuntimeError("filter provenance does not match exact proxy")
    doc=ET.parse(source_urdf).getroot()
    parent={j.find('child').get('link'):j.find('parent').get('link') for j in doc.findall('joint')}
    def ancestor(a,b):
        while b in parent:
            b=parent[b]
            if b==a:return True
        return False
    approved=set()
    for item in audit["pairs"]:
        a,b=item["a"],item["b"]
        pair=tuple(sorted((a,b)))
        if (a==b or a not in bodies or b not in bodies or pair in approved
            or not (ancestor(a,b) or ancestor(b,a)) or item["static_separation_mm"]>=0
            or (a.startswith("left_") and b.startswith("right_"))
            or (a.startswith("right_") and b.startswith("left_"))
            or not bodies[a].GetPath().HasPrefix(root_path)
            or not bodies[b].GetPath().HasPrefix(root_path)
            or bodies[a].IsInstanceProxy() or bodies[b].IsInstanceProxy()):
            raise RuntimeError(f"not a provenance-bound robot ancestor pair: {a}/{b}")
        approved.add(pair)
    if len(approved)!=18:
        raise RuntimeError("wrong measured pair count")
    # Capture the entire composed world's pre-existing filter relationships and
    # ensure only the listed robot pairs are added; do not erase any exclusions.
    def snapshot():
        out=set()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                prop=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
                for target in prop.GetTargets():out.add((str(prim.GetPath()),str(target)))
        return out
    before=snapshot()
    permitted=set()
    for a,b in sorted(approved):
        for source,target in ((a,b),(b,a)):
            ap=UsdPhysics.FilteredPairsAPI.Apply(bodies[source])
            if not ap.CreateFilteredPairsRel().AddTarget(bodies[target].GetPath()):
                raise RuntimeError(f"could not author pair {source} -> {target}")
            permitted.add((str(bodies[source].GetPath()),str(bodies[target].GetPath())))
    after=snapshot()
    if not before <= after or not (after - before) <= permitted or not permitted <= after:
        raise RuntimeError("composed filtered-pair effects differ from exact allowlist")
    return {"mode":"pre-existing authored filters preserved; explicit 18 measured same-chain pairs only",
            "scope":"not all ancestor pairs; no global self collision override",
            "provenance_sha256":PAIR_MANIFEST_SHA,"source_drive_tune_sha256":audit["frozen_s1_source_sha256"],
            "authored_link_pairs":audit["pairs"],"pairs_added_as_directed_targets":len(after-before)}
