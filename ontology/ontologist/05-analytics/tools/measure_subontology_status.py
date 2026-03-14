#!/usr/bin/env python3
"""
Measure completeness and status of sub-ontologies against the 10-dimension quality checklist.

This tool provides a quantitative assessment of sub-ontology maturity based on the
criteria defined in ontologist/01-docs/subontology-quality-checklist.md
"""

from pathlib import Path
import re
import sys
from typing import List
from dataclasses import dataclass
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = ROOT / "ontology"

@dataclass
class SubontologyMetrics:
    """Metrics for a sub-ontology module."""
    path: Path
    name: str

    # Core counts
    total_classes: int = 0
    total_properties: int = 0  # object + datatype
    total_individuals: int = 0

    # Quality dimensions (0-1 scores)
    scope_defined: float = 0.0
    core_classes: float = 0.0
    object_properties: float = 0.0
    datatype_properties: float = 0.0
    owl_axioms: float = 0.0
    units_quantities: float = 0.0
    external_mappings: float = 0.0
    provenance: float = 0.0
    validation: float = 0.0
    maturity_declared: float = 0.0

    # Derived metrics
    overall_score: float = 0.0
    maturity_level: str = "unknown"

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            'scope_defined': 0.10,
            'core_classes': 0.15,
            'object_properties': 0.10,
            'datatype_properties': 0.10,
            'owl_axioms': 0.10,
            'units_quantities': 0.10,
            'external_mappings': 0.05,
            'provenance': 0.15,
            'validation': 0.10,
            'maturity_declared': 0.05
        }

        self.overall_score = sum(
            getattr(self, dim) * weight
            for dim, weight in weights.items()
        )

        # Determine maturity level based on score
        if self.overall_score >= 0.9:
            self.maturity_level = "validated"
        elif self.overall_score >= 0.7:
            self.maturity_level = "curated"
        elif self.overall_score >= 0.4:
            self.maturity_level = "baseline"
        else:
            self.maturity_level = "scaffold"

        return self.overall_score

class SubontologyAnalyzer:
    """Analyzes sub-ontology files for quality metrics."""

    def __init__(self, ontology_dir: Path):
        self.ontology_dir = ontology_dir

    def analyze_file(self, file_path: Path) -> SubontologyMetrics:
        """Analyze a single TTL file."""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.split('\n')

        metrics = SubontologyMetrics(
            path=file_path,
            name=str(file_path.relative_to(self.ontology_dir))
        )

        # Count basic elements
        metrics.total_classes = len(re.findall(r'a\s+owl:Class', text))
        metrics.total_properties = len(re.findall(r'a\s+owl:(Object|Datatype)Property', text))
        metrics.total_individuals = len(re.findall(r'a\s+owl:NamedIndividual', text))

        # Analyze each quality dimension
        self._check_scope_defined(text, lines, metrics)
        self._check_core_classes(text, lines, metrics)
        self._check_object_properties(text, lines, metrics)
        self._check_datatype_properties(text, lines, metrics)
        self._check_owl_axioms(text, lines, metrics)
        self._check_units_quantities(text, lines, metrics)
        self._check_external_mappings(text, lines, metrics)
        self._check_provenance(text, lines, metrics)
        self._check_validation(text, lines, metrics)
        self._check_maturity_declared(text, lines, metrics)

        # Calculate final score
        metrics.calculate_overall_score()

        return metrics

    def _check_scope_defined(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check if scope is defined."""
        has_scope_note = "scopeNote" in text or "scope note" in text.lower()
        has_explicit_scope = any("SCOPE:" in line for line in lines)

        if has_scope_note and has_explicit_scope:
            metrics.scope_defined = 1.0
        elif has_scope_note or has_explicit_scope:
            metrics.scope_defined = 0.5
        else:
            metrics.scope_defined = 0.0

    def _check_core_classes(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check core class coverage."""
        if metrics.total_classes == 0:
            metrics.core_classes = 0.0
        elif metrics.total_classes < 3:
            metrics.core_classes = 0.3
        elif metrics.total_classes < 10:
            metrics.core_classes = 0.7
        else:
            metrics.core_classes = 1.0

    def _check_object_properties(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check object property coverage."""
        obj_props = len(re.findall(r'a\s+owl:ObjectProperty', text))
        if obj_props == 0:
            metrics.object_properties = 0.0
        elif obj_props < 3:
            metrics.object_properties = 0.5
        elif obj_props < 8:
            metrics.object_properties = 0.8
        else:
            metrics.object_properties = 1.0

    def _check_datatype_properties(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check datatype property coverage."""
        dt_props = len(re.findall(r'a\s+owl:DatatypeProperty', text))
        if dt_props == 0:
            metrics.datatype_properties = 0.0
        elif dt_props < 3:
            metrics.datatype_properties = 0.5
        elif dt_props < 8:
            metrics.datatype_properties = 0.8
        else:
            metrics.datatype_properties = 1.0

    def _check_owl_axioms(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check OWL axioms presence."""
        has_disjoint = "owl:disjointWith" in text
        has_restriction = "owl:Restriction" in text or "owl:cardinality" in text
        has_subclassof = "rdfs:subClassOf" in text

        count = sum([has_disjoint, has_restriction, has_subclassof])
        metrics.owl_axioms = count / 3.0

    def _check_units_quantities(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check unit/quantity pattern usage."""
        unitized_values = len(re.findall(r':UnitizedValue', text))
        qudt_units = len(re.findall(r'unit:', text))

        if unitized_values > 0 and qudt_units > 0:
            metrics.units_quantities = min(1.0, (unitized_values + qudt_units) / 10.0)
        elif unitized_values > 0:
            metrics.units_quantities = 0.7
        else:
            metrics.units_quantities = 0.0

    def _check_external_mappings(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check external mappings."""
        semis = len(re.findall(r'semiconto-standards:', text))
        pubchems = len(re.findall(r'pubchem\.ncbi\.nlm\.nih\.gov', text))
        astms = len(re.findall(r'astm\.org', text))

        if semis > 0 or pubchems > 0 or astms > 0:
            metrics.external_mappings = min(1.0, (semis + pubchems + astms) / 5.0)
        else:
            metrics.external_mappings = 0.0

    def _check_provenance(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check provenance completeness."""
        module_prov = all(prop in text for prop in ["dc:source", "dc:rights", "prov:hadPrimarySource"])

        # Check class-level provenance
        class_lines = []
        for i, line in enumerate(lines):
            if "a owl:Class" in line:
                # Collect next 10 lines for the class
                class_block = lines[i:i+10]
                class_lines.append('\n'.join(class_block))

        classes_with_prov = 0
        for class_block in class_lines:
            if "prov:wasInformedBy" in class_block or "dc:source" in class_block:
                classes_with_prov += 1

        class_prov_ratio = classes_with_prov / len(class_lines) if class_lines else 0

        if module_prov and class_prov_ratio > 0.5:
            metrics.provenance = 1.0
        elif module_prov or class_prov_ratio > 0.3:
            metrics.provenance = 0.7
        else:
            metrics.provenance = 0.3

    def _check_validation(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check validation (SHACL shapes)."""
        shacl_shapes = len(re.findall(r'a\s+sh:NodeShape', text))
        shacl_props = len(re.findall(r'sh:', text))

        if shacl_shapes > 0 and shacl_props > 5:
            metrics.validation = 1.0
        elif shacl_shapes > 0:
            metrics.validation = 0.7
        else:
            metrics.validation = 0.0

    def _check_maturity_declared(self, text: str, lines: List[str], metrics: SubontologyMetrics) -> None:
        """Check maturity level declaration."""
        has_maturity = "maturityLevel" in text or "maturity level" in text.lower()
        has_explicit = any("scaffold" in line or "baseline" in line or "curated" in line or "validated" in line
                          for line in lines if "maturity" in line.lower())

        if has_maturity and has_explicit:
            metrics.maturity_declared = 1.0
        elif has_maturity:
            metrics.maturity_declared = 0.7
        else:
            metrics.maturity_declared = 0.0


def main():
    """Main function to analyze sub-ontologies."""
    if len(sys.argv) > 1:
        # Analyze specific directory
        target_dir = Path(sys.argv[1])
        if not target_dir.is_absolute():
            target_dir = ONTOLOGY_DIR / target_dir
    else:
        # Analyze entire ontology
        target_dir = ONTOLOGY_DIR

    analyzer = SubontologyAnalyzer(ONTOLOGY_DIR)

    # Collect all TTL files
    ttl_files = []
    for p in target_dir.rglob("*.ttl"):
        if "PlaceholderClass" not in p.read_text(encoding="utf-8", errors="ignore"):
            ttl_files.append(p)

    # Analyze each file
    results = []
    for file_path in sorted(ttl_files):
        try:
            metrics = analyzer.analyze_file(file_path)
            results.append(metrics)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)

    # Generate report
    print("# Sub-Ontology Status Report")
    print(f"\nAnalyzed {len(results)} modules from {target_dir}")
    print("\n## Summary Statistics")

    # Calculate averages
    if results:
        avg_score = sum(r.overall_score for r in results) / len(results)
        maturity_counts = defaultdict(int)
        for r in results:
            maturity_counts[r.maturity_level] += 1

        print(f"\n- Average score: {avg_score:.2f}")
        print("- Modules by maturity:")
        for level, count in sorted(maturity_counts.items()):
            print(f"  - {level}: {count}")

    print("\n## Detailed Results")
    print("\n| Module | Classes | Props | Score | Maturity | Physics | QC | Supply | Provenance | Validation |")
    print("|--------|---------|-------|-------|----------|---------|----|--------|------------|------------|")

    for r in results:
        # Determine feature areas
        physics_score = (r.owl_axioms + r.units_quantities) / 2
        qc_score = r.validation
        supply_score = r.external_mappings

        print(f"| {r.name} | {r.total_classes} | {r.total_properties} | {r.overall_score:.2f} | {r.maturity_level} | "
              f"{physics_score:.1f} | {qc_score:.1f} | {supply_score:.1f} | {r.provenance:.1f} | {r.validation:.1f} |")

    print("\n## Recommendations")
    print("\nModules with score < 0.5 should be reviewed for:")
    print("- Missing scope definitions")
    print("- Insufficient properties or relationships")
    print("- Lack of validation (SHACL shapes)")
    print("- Missing provenance documentation")
    print("\nModules with score < 0.7 should consider:")
    print("- Adding more external mappings")
    print("- Enhancing class-level provenance")
    print("- Adding more OWL axioms for reasoning")

if __name__ == "__main__":
    main()
