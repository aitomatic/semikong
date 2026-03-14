# Provenance Documentation Skill

## Purpose
Document source attribution and copyright/IP compliance within ontology TTL files using W3C PROV-O and Dublin Core vocabularies.

## When to Use
- Adding new substantive ontology classes or properties
- Creating new module ontologies
- When fair-use justification is needed

## Required Vocabularies
```turtle
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```

## Module-Level Provenance Pattern
```turtle
<module-iri> a owl:Ontology ;
    rdfs:label "Module Name" ;
    rdfs:comment "Module description" ;
    dc:source "Primary source description or URL" ;
    dc:rights "Fair use: industry-general concepts only" ;
    prov:hadPrimarySource <https://example.com/source> ;
    prov:wasInformedBy "Model synthesis of public domain knowledge" ;
    prov:wasGeneratedBy "semicont-ontology-build" ;
    prov:generatedAtTime "2026-03-10"^^xsd:date .
```

## Class-Level Provenance Pattern
```turtle
example:SubstantiveClass a owl:Class ;
    rdfs:label "Class Name" ;
    rdfs:comment "Class description" ;
    dc:source "Public semiconductor equipment taxonomy" ;
    dc:rights "Fair use: generic equipment category" ;
    prov:wasInformedBy "Industry-standard classification systems" ;
    prov:wasGeneratedBy "semicont-ontology-build" ;
    prov:generatedAtTime "2026-03-10"^^xsd:date .
```

## Fair Use Documentation
When using fair-use excerpts:
```turtle
dc:rights "Fair use: short excerpt for semantic classification purposes" ;
dc:source "Copyrighted technical specification, paraphrased for ontology" .
```

## Checklist
- [ ] Module metadata includes dc:source and dc:rights
- [ ] Substantive classes have prov:wasInformedBy
- [ ] Fair-use justification documented in dc:rights
- [ ] External URLs use prov:hadPrimarySource
- [ ] No proprietary content marked as "industry-general"