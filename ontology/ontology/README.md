# Semicont Ontology Directory

## Purpose

`ontology/` is the canonical semantic source for this repository.

This directory defines:

1. Industry-layer knowledge structure.
2. Domain vocabulary and relationships in Turtle (`.ttl`).
3. Stable semantic contracts that downstream tools can query and reuse.

## Design Intent

The intent is to keep `ontology/` focused on durable knowledge, not runtime procedure.

`ontology/` should answer:

1. What entities exist?
2. How are they related?
3. What process semantics/states are valid in the domain?

It should not prescribe operational execution details for specific systems.

## Source Of Truth Policy

Rules:

1. Turtle (`.ttl`) files under `ontology/` are the source of truth.
2. `ontology/README.md` is the only non-TTL file in this tree.
3. Derived artifacts (rules YAML, skills markdown, workflow runbooks, etc.) are outside `ontology/`.

## Versioning Policy

Current repository-wide ontology module baseline:

1. Every `ontology/**/*.ttl` module uses `owl:versionInfo "0.1.0"`.
2. This baseline is enforced by ontologist audit and curation rules.
3. Any future version transition must be intentional, documented in `ontologist/01-docs/DECISIONS.md`, and applied consistently.

## Top-Level Taxonomy

```text
ontology/
├── 00-shared/
├── 01-standards-reference/
├── 01-integrators/
├── 02-ip-providers/
├── 03-fabless/
├── 04-eda/
├── 05-foundry-idm/
├── 06-osat-packaging-test/
├── 07-wfe/
├── 08-materials/
└── 09-supply-chain/
```

## Subcategory Plan

- `00-shared`: `standards`, `regulations`, `common_technologies`, `data_models`, `sustainability_metrics`, `intellectual_property_rights`, `industry_associations`, `shared_resources`
- `01-integrators`: `system_integrators`, `end_use_applications`, `module_assembly`, `testing_and_validation`, `customization_services`, `market_segments`
- `02-ip-providers`: `ip_cores`, `interface_ips`, `analog_mixed_signal_ips`, `libraries`, `verification_ips`, `licensing_models`, `providers`
- `03-fabless`: `chip_design_companies`, `design_methodologies`, `verification_processes`, `process_nodes`, `product_types`, `collaboration_tools`
- `04-eda`: `design_tools`, `simulation_software`, `verification_tools`, `physical_design`, `vendors`, `automation_scripts`
- `05-foundry-idm`: `foundry_services`, `idm_operations`, `wafer_fabrication_processes`, `process_technologies`, `capacity_management`, `yield_optimization`
- `06-osat-packaging-test`: `osat_providers`, `packaging_types`, `testing_services`, `assembly_processes`, `reliability_testing`, `failure_analysis`
- `07-wfe`: `lithography_equipment`, `deposition_tools`, `etch_systems`, `metrology_inspection`, `clean_strip_tools`, `ion_implantation`, `thermal_processing`, `chemical_mechanical_planarization`, `photoresist_processing`, `doping_equipment`, `wafer_handling_automation`, `annealing_furnaces`, `oxidation_systems`, `inspection_tools`, `suppliers_and_maintenance`
- `08-materials`: `substrate_materials`, `chemicals`, `gases`, `metals_targets`, `advanced_materials`, `suppliers`, `quality_control`
- `09-supply-chain`: `logistics_providers`, `inventory_management`, `demand_forecasting`, `risk_assessment`, `traceability_systems`, `supplier_relationships`, `sustainability_in_supply`

## Sub-Subcategory Convention

Each subcategory may include finer-grained sub-subcategories.

Example:

```text
ontology/07-wfe/06-etch-systems/
├── ontology.ttl
├── plasma_etch/ontology.ttl
├── wet_etch/ontology.ttl
└── atomic_layer_etch/ontology.ttl
```

## What Belongs In `ontology/`

Add to `ontology/` when content is stable, queryable domain knowledge.

Examples:

1. Classes, properties, relationships, taxonomies.
2. Process semantics: stage types, states, transitions, dependencies.
3. Inputs/outputs/roles as domain concepts.
4. Shared cross-layer concepts in `00-shared`.

## Open-Source Boundary (General vs Company-Specific)

This ontology is intended to be open and reusable across organizations.

Model in core `ontology/`:

1. Industry-general knowledge that is stable across companies.
2. Process/mechanism categories that support common querying and reasoning.
3. Shared terminology and relationships that multiple organizations can adopt.

Do not model in core `ontology/`:

1. Company-specific competitive differentiation (proprietary recipes, internal decision logic, product-line-specific capability claims).
2. Organization-private operational details that are not broadly reusable domain semantics.

Depth guidance:

1. Decompose until the next split would mostly be vendor/company-specific rather than domain-semantic.
2. Keep extension points so downstream users can specialize in private or project-specific ontologies without changing core structure.

### Rules In `ontology/`

Rules belong in `ontology/` only when they are semantic model rules encoded in Turtle.

Examples that belong:

1. Ontology/constraint semantics represented in TTL (including SHACL-in-TTL).
2. Domain rule concepts that should be queryable as knowledge.

Examples that do not belong:

1. Runtime policy engine YAML.
2. Environment-specific threshold/config rules.
3. Agentic decision heuristics used at execution time.

### About Workflows

General workflow knowledge can belong in `ontology/` when modeled semantically.

Examples that belong:

1. Workflow type/stage/state ontology terms.
2. Step dependency semantics (precedes, dependsOn, producesArtifact).
3. Lifecycle/status concepts.

Examples that do not belong:

1. Agent runbooks and procedural instructions.
2. Runtime heuristics/tuning rules.
3. Environment/pipeline execution config.

### Workflow Examples

Example workflows should not live in `ontology/`.

1. Keep workflow examples, walkthroughs, and sample executions in `examples/` or `docs/`.
2. In `ontology/`, keep only workflow semantics (types, stages, states, dependencies, roles).

## What Does Not Belong In `ontology/`

1. `rules.yaml` operational policy files.
2. `skills.md` procedural runbooks.
3. Free-form markdown derived docs (except this file).
4. Product/runtime code or environment-specific configs.

### Borderline Cases (Decision Guidance)

1. Workflow instance examples: outside `ontology/` (`examples/` or `docs/`).
2. Competency queries and query packs: outside `ontology/`.
3. Validation harnesses/scripts: outside `ontology/`; semantic constraints can remain in TTL.
4. Heuristics: semantic intent in TTL is acceptable; executable heuristics stay outside.

## What Contributors Add

Contributors should add or modify:

1. New `.ttl` modules in the correct layer/subcategory.
2. Ontology imports and semantic links to existing modules.
3. Term metadata/provenance fields required by project policy.

## What Users Can Expect To Get Out Of `ontology/`

Consumers can rely on:

1. A deterministic industry-layer semantic map.
2. Machine-readable TTL modules for querying, reasoning, and mapping.
3. Reusable domain and cross-domain vocabulary contracts.

## Change Discipline

When moving or renaming ontology paths:

1. Update IRIs and imports consistently.
2. Update references in repository docs and ontologist tracking files.
3. Preserve semantic intent; avoid replacing curated modules with placeholders.
