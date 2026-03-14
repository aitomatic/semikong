# Skill: Modeling Boundary

## Goal

Decide whether information belongs in ontology TTL or in ontologist operational docs.

## Decision Rule

- If it defines domain meaning or relationships: put it in `ontology/**/*.ttl`.
- If it defines how curators/agents perform work: put it in `ontologist/`.

## Examples

- Belongs in ontology: classes for packaging types, test stages, yield concepts.
- Belongs in ontologist: curation checklists, QA gates, review workflows, scripts.
