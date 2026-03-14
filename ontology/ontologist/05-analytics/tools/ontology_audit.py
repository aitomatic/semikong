#!/usr/bin/env python3
"""Repository-local audit checks for ontology curation boundaries and legacy drift."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = ROOT / "ontology"
ALLOWED_NON_TTL = {ONTOLOGY_DIR / "README.md"}
LEGACY_TOKENS = [
    "06-packaging-test-equipment",
    "/use-cases/",
    "ontology/05-foundry-idm/use-cases",
]
LEGACY_SCAN_PATHS = [
    ROOT / "ontology",
    ROOT / "ontologist" / "01-docs",
    ROOT / "ontologist" / "04-context",
]


def iter_files(base: Path):
    for p in base.rglob("*"):
        if p.is_file():
            yield p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def check_ontology_boundary(issues: list[str]) -> None:
    for p in iter_files(ONTOLOGY_DIR):
        if p.suffix == ".ttl":
            continue
        if p in ALLOWED_NON_TTL:
            continue
        issues.append(f"boundary: disallowed non-TTL file in ontology/: {rel(p)}")


def check_legacy_tokens(issues: list[str]) -> None:
    for base in LEGACY_SCAN_PATHS:
        if not base.exists():
            continue
        for p in iter_files(base):
            if ".git" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for token in LEGACY_TOKENS:
                if token in text:
                    issues.append(f"legacy: token '{token}' found in {rel(p)}")


def check_absolute_paths_in_ttl(issues: list[str]) -> None:
    for p in iter_files(ONTOLOGY_DIR):
        if p.suffix != ".ttl":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "/Users/" in text or "C:\\\\" in text:
            issues.append(f"ttl: absolute filesystem path token found in {rel(p)}")


def main() -> int:
    if not ONTOLOGY_DIR.exists():
        print("ERROR: ontology/ directory not found")
        return 2

    issues: list[str] = []
    check_ontology_boundary(issues)
    check_legacy_tokens(issues)
    check_absolute_paths_in_ttl(issues)
    check_ttl_provenance_completeness(issues)
    check_class_level_provenance(issues)
    check_ontology_version_baseline(issues)

    if issues:
        print("Ontology audit failed:")
        for issue in issues:
            print(f"- {issue}")
        print(f"\nTotal issues: {len(issues)}")
        return 1

    print("Ontology audit passed: no boundary or legacy issues detected.")
    return 0


def check_ttl_provenance_completeness(issues: list[str]) -> None:
    """Check that TTL modules have required provenance fields."""
    required_props = ["dc:source", "dc:rights"]

    for p in iter_files(ONTOLOGY_DIR):
        if p.suffix != ".ttl":
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")

        # Find the ontology declaration
        if "a owl:Ontology" not in text:
            continue

        # Check for required provenance properties
        missing = []
        for prop in required_props:
            if prop not in text:
                missing.append(prop)

        if missing:
            missing_props = ", ".join(missing)
            issues.append(
                f"ttl_provenance_completeness: {rel(p)} missing required properties: {missing_props}"
            )


def check_class_level_provenance(issues: list[str]) -> None:
    """Check that substantive classes have provenance documentation."""
    # This is an informational check for now - we'll flag but not fail
    for p in iter_files(ONTOLOGY_DIR):
        if p.suffix != ".ttl":
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")

        # Skip placeholder files
        if "PlaceholderClass" in text:
            continue

        # Count classes and those with provenance
        lines = text.split('\n')
        total_classes = 0
        classes_with_prov = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            if "a owl:Class" in line:
                total_classes += 1
                # Check next few lines for provenance
                for j in range(i+1, min(i+10, len(lines))):
                    if "dc:source" in lines[j] or "dc:rights" in lines[j] or "prov:wasInformedBy" in lines[j]:
                        classes_with_prov += 1
                        break
                    if lines[j].strip().endswith(".") and not lines[j].strip().startswith("#"):
                        break
                i += 1
            else:
                i += 1

        if total_classes > 3 and classes_with_prov < total_classes:
            issue = (
                f"class_level_provenance: {rel(p)} has {total_classes} classes but only "
                f"{classes_with_prov} with provenance documentation (informational)"
            )
            issues.append(issue)


def check_ontology_version_baseline(issues: list[str]) -> None:
    """Check that ontology modules use the current repository version baseline."""
    baseline = 'owl:versionInfo "0.1.0"'
    for p in iter_files(ONTOLOGY_DIR):
        if p.suffix != ".ttl":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "a owl:Ontology" in text and baseline not in text:
            issues.append(f"ontology_version_baseline: {rel(p)} missing required baseline {baseline}")


if __name__ == "__main__":
    raise SystemExit(main())
