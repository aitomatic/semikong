# Ontology Design Spec

## Purpose

Define deterministic design rules so different agents produce compatible ontology artifacts.

## Canonical Serialization

1. Primary format: Turtle (`.ttl`).
2. One module per file.
3. UTF-8 text, Unix newlines.

## Namespace Policy

Use these namespace families:

1. `semicont-core:` shared cross-cutting vocabulary.
2. `semicont-<module>:` domain module vocabulary.
3. `semicont-shape:` validation terms.
4. `prov:` provenance (PROV-O aligned usage where applicable).
5. `qudt:` units/quantities where measurement terms require it.

Rules:

1. Stable base IRIs only; do not rename once released.
2. New term IRIs are lowercase kebab-case.
3. No module may redefine a `semicont-core:` term.

## Module Boundaries

1. `ontology/00-shared/use-cases/`: shared cross-cutting terms only.
2. `ontology/01-*/use-cases/` through `ontology/09-*/use-cases/`: layer-specific modules.
3. If a term is used by 2+ layers, promote it to `00-shared`.
4. If a term is used by 1 layer, keep it in that layer's `use-cases/` directory.

## Class and Property Conventions

1. Class names: singular noun (`Wafer`, `ProcessStep`).
2. Object properties: verb phrase (`hasProcessStep`, `measuredByTool`).
3. Datatype properties: attribute phrase (`hasLotId`, `hasMeasurementValue`).
4. Every class/property requires:
- human-readable label
- definition
- provenance fields (see `docs/source-policy.md`)

## Required Metadata Per Term

Each term must include:

1. Preferred label
2. Definition
3. Module owner
4. Status (`draft`, `reviewed`, `stable`, `deprecated`)
5. Version introduced
6. Provenance link/reference

## Change Policy

1. Additive term/property: minor version.
2. Breaking rename/removal/semantic shift: major version.
3. Typo/comment-only fix: patch version.
4. Deprecated terms must include replacement guidance.

## Minimal File Skeleton

```turtle
@prefix semicont-core: <https://example.org/semicont/shared#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

semicont-core:Wafer a rdfs:Class ;
  rdfs:label "Wafer" ;
  rdfs:comment "Semiconductor wafer unit of work before dicing." .
```
