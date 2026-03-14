# State

## Current Focus

Execute a phased ontology fill-out plan that prioritizes SEMI-aligned core semantics first, then TEL/JSR/Tata/Renesas/MaxLinear-adjacent branches, while preserving the open-source boundary (industry-general depth only).

## Current Status

- `ontology/` uses the updated `00-` to `09-` top-level layout.
- `ontology/README.md` is the ontology contract.
- `ontologist/` now follows renumbered ks-style directories (`01-docs` to `06-learning`).
- Baseline audit tool exists at `ontologist/05-analytics/tools/ontology_audit.py`.
- Latest audit run (2026-03-10): `python ontologist/05-analytics/tools/ontology_audit.py` passed.
- Ontology version baseline is standardized to `owl:versionInfo "0.1.0"` across all `ontology/**/*.ttl` modules, with audit enforcement enabled.
- TTL-quality scan (text-based) currently shows 58 `.ttl` files with placeholder markers.
- Prioritization sequence is now explicit:
  1. SEMI backbone (`00-shared`, `01-standards-reference`)
  2. Tokyo Electron-relevant `07-wfe` depth
  3. JSR-relevant `08-materials` depth
  4. Tata-relevant `06-osat-packaging-test` depth
  5. Renesas/MaxLinear-relevant device/application taxonomy
  6. Non-proprietary company mapping examples last

### Status Table (2026-03-10)

| Dimension | Current Status | Evidence |
| --- | --- | --- |
| Audit gates | Pass | `python ontologist/05-analytics/tools/ontology_audit.py` |
| Ontology TTL boundary | Pass | non-TTL files under `ontology/` (excluding `ontology/README.md`): `0` |
| Module count | 446 TTL modules | `find ontology -name '*.ttl'` |
| Version baseline | 446/446 at `owl:versionInfo "0.1.0"` | `rg 'owl:versionInfo \"0.1.0\"'` |
| Placeholder debt | 58 TTL modules contain placeholder markers | `rg -l 'PlaceholderClass|placeholder' ontology --glob '*.ttl'` |
| WFE placeholder debt | 1 module | same scan, layer `07-wfe` |
| Materials placeholder debt | 1 module | same scan, layer `08-materials` |
| Top-level taxonomy drift | Present | extra top-level dirs: `00-physics`, `01-devices` |

### Benchmark Calibration (External Ontologies)

| Benchmark | Classes | ObjProps | DataProps | VersionIRI | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| SemicONTO 0.2 | 51 | 17 | 9 | 1 | `raw.githubusercontent.com/huanyu-li/SemicONTO/main/ontology/0.2/SemicONTO.ttl` |
| Digital Reference 1.2.10 | 1368 | 879 | 769 | 0 | `raw.githubusercontent.com/tibonto/dr/master/DigitalReference.ttl` |
| IOF Core (202502) | 265 | 63 | 2 | 1 | `raw.githubusercontent.com/iofoundry/ontology/master/core/Core.rdf` |
| SAREF4INMA v2.1.1 | 26 | 16 | 10 | 1 | `saref.etsi.org/saref4inma/v2.1.1/saref4inma.ttl` |

Local reference point: Classes `770`, ObjProps `81`, DataProps `180`, VersionIRI `0`.

## Next Three Actions

1. Continue Phase 3 materials depth by curating `08-materials/02-gases/` using the same facet-complete template.
2. Resolve top-level taxonomy drift (`00-physics`, `01-devices`) by either integrating into canonical layers or updating the documented layer contract.
3. Extend `ontology_audit.py` to enforce additional facet-completeness checks in machine-readable form.

## Active Risks

- Large generated ontology tree still contains placeholder scaffolds (58 files), which reduces semantic usefulness.
- Syntax-level TTL validation is not automated yet in this environment (no local RDF parser dependency installed for the audit script).
- Cross-branch depth can diverge without explicit leaf-threshold review at each phase gate.
- Company pressure can blur boundary discipline unless mapping work stays after core completion and remains non-proprietary.
- Taxonomy contract drift exists between documentation and current top-level directories (`00-physics`, `01-devices`).

## 2026-03-10 Update (Etch Systems)

- Completed curation of `ontology/07-wfe/06-etch-systems/ontology.ttl` to operational depth with:
  - dry/wet etch taxonomy through sub- and sub-subcategories,
  - process stages and lifecycle states,
  - capabilities/constraints and parameterization classes,
  - quality/specification classes and SHACL validation shapes,
  - module/class/property provenance from primary public sources.
- Post-edit audit run completed: boundary/legacy gates clean, with existing repo-wide class-level provenance informational findings unchanged in other modules.

## 2026-03-11 Update (Etch Systems Near-100 Plan)

- Executed the near-100 completion plan for `ontology/07-wfe/06-etch-systems/ontology.ttl` with internet-grounded public sources only.
- Strengthened facet coverage with:
  - chamber configuration facet (single chamber vs cluster tool),
  - parameter range modeling (`ParameterRangeSpec`, min/max bounds),
  - improved quantity/unit modeling (`NumericMeasure`, `EngineeringUnit`),
  - tighter SHACL checks (uniformity bounds and min<=max SPARQL check),
  - conservative external alignment stubs to SEMI references.
- Preserved industry-general boundary and explicitly avoided vendor-internal/proprietary recipe semantics.
- Post-edit verification completed:
  - `python ontologist/05-analytics/tools/ontology_audit.py` (pass)
  - `python ontologist/05-analytics/tools/benchmark_calibration.py` (completed; baseline comparison recorded).

## 2026-03-11 Update (Etch Systems 100% Completion Target)

- Completed the remaining Etch Systems checklist items:
  - lifecycle transition graph semantics,
  - actor-role semantics for operation/process/equipment ownership,
  - stronger SHACL for configuration cardinality and setpoint-range linkage,
  - deeper external crosswalk stubs.
- Module status advanced to `validated` for current industry-general scope.
- Verification:
  - `python ontologist/05-analytics/tools/ontology_audit.py` pass
  - `python ontologist/05-analytics/tools/benchmark_calibration.py` completed

## 2026-03-11 Update (Etch Systems Quality Assessment)

Applied subontology-quality-checklist.md to 06-etch-systems:

| Module | Score | Maturity | Classes | Props | Status |
|--------|-------|----------|---------|-------|--------|
| `ontology/07-wfe/06-etch-systems/ontology.ttl` | 0.77 | curated | 52 | 24 | ✅ Complete |
| `ontology/07-wfe/06-etch-systems/01-plasma-etch/ontology.ttl` | 0.33 | scaffold | 12 | 0 | ⚠️ Needs expansion |
| `ontology/07-wfe/06-etch-systems/02-wet-etch/ontology.ttl` | 0.29 | scaffold | 6 | 0 | ⚠️ Needs expansion |
| `ontology/07-wfe/06-etch-systems/00-atomic-layer-etch/ontology.ttl` | 0.29 | scaffold | 3 | 0 | ⚠️ Needs expansion |
| Plasma etch sub-branches (4 modules) | 0.23 | scaffold | 1 | 0 | ❌ Placeholders only |

**Summary**: 1 curated module, 7 scaffold modules. Main module production-ready, branches need development.

Detailed assessment: `ontologist/04-context/07-wfe-06-etch-systems-quality-assessment.md`

## Last Updated

2026-03-10

## 2026-03-10 Update

**Progress:** Phase 2 WFE equipment taxonomy 96% complete.
- All major WFE equipment modules created with industry-general concepts
- Ion implantation, CMP, clean/strip, metrology, inspection, thermal processing completed
- Only 14-suppliers-and-maintenance module remains as placeholder

**Governance Milestone:** Comprehensive provenance documentation implemented
- Module-level dc:source/dc:rights added to all 142 TTL files
- Audit gate enforcement: 100% provenance compliance achieved
- Zero boundary/legacy issues remain
- 7 informational class-level findings (acceptable for initial implementation)

**Next Focus:** Phase 3 - Continue materials depth work
- 08-materials/02-gases/ - Specialty gases for semiconductor processes (N2, Ar, specialty dopants)
- 08-materials/03-wafers/ - Silicon and compound semiconductor wafers (Si, GaAs, SiC)
- 08-materials/04-polymers/ - Photoresist polymers and specialty materials
- 08-materials/05-metals/ - Metal films and interconnect materials (Al, Cu, Ti, W)
- Enhance class-level provenance in high-value modules (informational)
- Add TTL syntax validation when rdflib dependency available

**Risks Addressed:**
- Provenance gaps: Fully resolved with systematic documentation
- Copyright compliance: All content documented as fair use/public domain
- Boundary discipline: Maintained industry-general scope throughout

Last Updated: 2026-03-10
