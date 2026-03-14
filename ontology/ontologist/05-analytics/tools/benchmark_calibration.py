#!/usr/bin/env python3
"""Compute lightweight structural calibration metrics for local ontology vs external benchmarks."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
LOCAL_ONTOLOGY_DIR = ROOT / "ontology"
BENCHMARK_DIR = ROOT / "ontologist" / "05-analytics" / "benchmarks"

BENCHMARKS = [
    {
        "name": "SemicONTO 0.2",
        "path": BENCHMARK_DIR / "SemicONTO-0.2.ttl",
        "source": "https://raw.githubusercontent.com/huanyu-li/SemicONTO/main/ontology/0.2/SemicONTO.ttl",
    },
    {
        "name": "Digital Reference 1.2.10",
        "path": BENCHMARK_DIR / "DigitalReference.ttl",
        "source": "https://raw.githubusercontent.com/tibonto/dr/master/DigitalReference.ttl",
    },
    {
        "name": "IOF Core (202502)",
        "path": BENCHMARK_DIR / "IOF-Core.rdf",
        "source": "https://raw.githubusercontent.com/iofoundry/ontology/master/core/Core.rdf",
    },
    {
        "name": "SAREF4INMA v2.1.1",
        "path": BENCHMARK_DIR / "SAREF4INMA-v2.1.1.ttl",
        "source": "https://saref.etsi.org/saref4inma/v2.1.1/saref4inma.ttl",
    },
]


def count(patterns: list[str], text: str) -> int:
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def metrics(text: str) -> dict[str, int]:
    return {
        "ontologies": count([r"\ba\s+owl:Ontology\b", r"rdf:type\s+owl:Ontology\b", r"<owl:Ontology\b"], text),
        "classes": count([r"\ba\s+owl:Class\b", r"rdf:type\s+owl:Class\b", r"<owl:Class\b"], text),
        "obj_props": count([r"\ba\s+owl:ObjectProperty\b", r"rdf:type\s+owl:ObjectProperty\b", r"<owl:ObjectProperty\b"], text),
        "data_props": count([r"\ba\s+owl:DatatypeProperty\b", r"rdf:type\s+owl:DatatypeProperty\b", r"<owl:DatatypeProperty\b"], text),
        "ann_props": count([r"\ba\s+owl:AnnotationProperty\b", r"rdf:type\s+owl:AnnotationProperty\b", r"<owl:AnnotationProperty\b"], text),
        "individuals": count([r"\ba\s+owl:NamedIndividual\b", r"rdf:type\s+owl:NamedIndividual\b", r"<owl:NamedIndividual\b"], text),
        "imports": len(re.findall(r"\bowl:imports\b", text)),
        "versionInfo": len(re.findall(r"\bowl:versionInfo\b", text)),
        "versionIRI": len(re.findall(r"\bowl:versionIRI\b", text)),
        "shapes": count([r"\ba\s+sh:NodeShape\b", r"rdf:type\s+sh:NodeShape\b"], text),
    }


def load_local_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(LOCAL_ONTOLOGY_DIR.rglob("*.ttl"))
    )


def ratio(a: int, b: int) -> str:
    if b == 0:
        return "n/a"
    return f"{a / b:.1f}x"


def main() -> int:
    missing = [b for b in BENCHMARKS if not b["path"].exists()]
    if missing:
        print("ERROR: missing benchmark files:")
        for b in missing:
            print(f"- {b['name']}: {b['path']}")
        return 2

    local_text = load_local_text()
    local = metrics(local_text)

    print("# Ontology Benchmark Calibration")
    print("")
    print("| Benchmark | Ontologies | Classes | ObjProps | DataProps | AnnProps | Individuals | Imports | VersionInfo | VersionIRI | Shapes |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        "| Local (`ontology/`)"
        f" | {local['ontologies']} | {local['classes']} | {local['obj_props']} | {local['data_props']}"
        f" | {local['ann_props']} | {local['individuals']} | {local['imports']} | {local['versionInfo']}"
        f" | {local['versionIRI']} | {local['shapes']} |"
    )

    for b in BENCHMARKS:
        txt = b["path"].read_text(encoding="utf-8", errors="ignore")
        m = metrics(txt)
        print(
            f"| {b['name']}"
            f" | {m['ontologies']} | {m['classes']} | {m['obj_props']} | {m['data_props']}"
            f" | {m['ann_props']} | {m['individuals']} | {m['imports']} | {m['versionInfo']}"
            f" | {m['versionIRI']} | {m['shapes']} |"
        )

    print("\n# Local/Benchmark Ratios (selected)")
    print("| Benchmark | Classes | ObjProps | DataProps | VersionIRI |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for b in BENCHMARKS:
        txt = b["path"].read_text(encoding="utf-8", errors="ignore")
        m = metrics(txt)
        print(
            f"| {b['name']} | {ratio(local['classes'], m['classes'])}"
            f" | {ratio(local['obj_props'], m['obj_props'])} | {ratio(local['data_props'], m['data_props'])}"
            f" | {ratio(local['versionIRI'], m['versionIRI'])} |"
        )

    print("\n# Sources")
    for b in BENCHMARKS:
        print(f"- {b['name']}: {b['source']}")
    print(f"\n- Local TTL modules: {len(list(LOCAL_ONTOLOGY_DIR.rglob('*.ttl')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
