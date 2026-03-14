# Provenance Documentation Workflow

## Purpose
Ensure all ontology content has complete provenance documentation meeting IP/copyright requirements.

## Trigger
This workflow runs automatically as part of the execution loop when making substantive ontology edits.

## Steps

### 1. Pre-Edit Assessment
- Identify if edit is substantive (new classes/properties vs. minor fixes)
- Determine if sources are public domain, fair use, or require permission
- Check if content is industry-general vs. company-specific

### 2. Source Documentation
For each source used:
- Record URL or citation in curation notes
- Note whether it's model reasoning, public reference, or fair use
- Document any paraphrasing performed

### 3. TTL Provenance Addition
Add to module ontology:
```turtle
@prefix dc: <http://purl.org/dc/elements/1.1/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
```

Add to module metadata:
```turtle
<module> dc:source "Description of source" ;
         dc:rights "Fair use: industry-general concepts" ;
         prov:hadPrimarySource <url> .
```

For substantive classes:
```turtle
<Class> dc:source "Source description" ;
        dc:rights "Rights statement" ;
        prov:wasInformedBy "Description of reasoning" .
```

### 4. Context Documentation
Record in WORKLOG.md:
- Sources consulted
- Fair-use justifications
- Any copyright concerns

### 5. Validation Check
Run audit script to verify:
- No absolute paths in TTL
- Proper IRI formatting
- Provenance properties present

## Quality Gates
- All substantive classes have dc:source
- Module has dc:rights statement
- Fair use is justified in documentation
- No proprietary content without clearance

## Output
- Updated TTL files with provenance
- WORKLOG.md entry with detailed source notes
- Audit passing with zero issues