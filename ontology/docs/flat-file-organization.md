# Flat-File Ontology Organization for AI Agent Navigation

## Design Principles

1. **Discoverability First** - Agents should find relevant files without deep directory traversal
2. **Semantic Locality** - Related concepts cluster together physically
3. **Progressive Disclosure** - Start broad, refine through file structure
4. **Machine-Readable Metadata** - Every file self-describes for agent consumption

## Proposed Structure

```
semicont/
├── 00-INDEX.yaml              # Master index with file relationships
├── ONTOLOGY-MAP.md           # Visual map for humans
├──
├── 01-core-concepts/         # Highest-level abstractions
│   ├── concepts.index.yaml   # Lists all concepts in this directory
│   ├── Material.md
│   ├── Process.md
│   ├── Equipment.md
│   ├── Product.md
│   └── relations.yaml        # Defines relationships between core concepts
│
├── 02-industry-layers/       # Organized by who/where
│   ├── layers.index.yaml
│   ├── 02-01-system-companies.md
│   ├── 02-02-fabless.md
│   ├── 02-03-eda-tools.md
│   └── ...
│
├── 03-domain-modules/        # Organized by what
│   ├── modules.index.yaml
│   ├── 03-01-lot-genealogy/
│   │   ├── module.meta.yaml  # Module metadata, imports, exports
│   │   ├── concepts/         # Concept definitions
│   │   │   ├── WaferLot.md
│   │   │   ├── LotSplit.md
│   │   │   └── relations.yaml
│   │   ├── constraints/      # Validation rules
│   │   │   ├── cardinality.yaml
│   │   │   └── business-rules.yaml
│   │   └── examples/         # Usage examples
│   │       ├── split-example.md
│   │       └── merge-example.md
│   │
│   └── 03-02-process-control/
│       └── ...
│
├── 04-cross-cutting/         # Perspectives that span domains
│   ├── quality-perspective/
│   ├── cost-perspective/
│   └── time-perspective/
│
└── 99-generated/             # Machine-generated artifacts
    ├── ttl-export/          # Turtle files (if needed)
    ├── schema.json          # Unified schema
    └── search-index.json    # Full-text search index
```

## Key Features for AI Navigation

### 1. Numbered Prefix System
- Enables deterministic traversal order
- Agents can process sequentially or jump to specific sections
- Maintains hierarchical relationships (01-, 02-, 03-)

### 2. Index Files at Every Level
```yaml
# concepts.index.yaml
concepts:
  - name: Material
    file: Material.md
    type: class
    relationships:
      - type: subclass-of
        target: Thing
      - type: related-to
        target: Process
        via: used-in

  - name: Process
    file: Process.md
    type: class
    # ...

relationships:
  - type: material-process
    domain: Material
    range: Process
    cardinality: many-to-many
```

### 3. Standardized File Headers
```markdown
<!-- Material.md -->
---
concept_type: class
namespace: semicont-core
version: 0.1.0
provenance:
  source: industry-standards
  confidence: high
relationships:
  - used-in: Process
  - measured-by: Measurement
definition_source: SEMI standards
---

# Material

Physical substance used in semiconductor manufacturing...
```

### 4. Semantic Clustering
- All related concepts in same directory
- Relations defined locally first, then cross-referenced globally
- Examples co-located with concepts they exemplify

### 5. Machine-Readable Relations
```yaml
# relations.yaml
relationships:
  - id: material-used-in-process
    type: object-property
    domain: Material
    range: Process
    cardinality: [0, *]
    inverse: process-uses-material

  - id: process-precedes-process
    type: transitive
    domain: Process
    range: Process
    temporal: true
```

## Navigation Strategies for Agents

### 1. Index-First Traversal
```python
def explore_ontology():
    master_index = load_yaml("00-INDEX.yaml")
    for section in master_index.sections:
        section_index = load_yaml(f"{section}/index.yaml")
        process_section(section_index)
```

### 2. Relationship Following
```python
def follow_relationships(concept_name):
    concept_file = find_concept_by_name(concept_name)
    relations = extract_relations(concept_file)
    for relation in relations:
        target_concepts = find_concepts_by_type(relation.range)
        analyze_relationship(concept_name, target_concepts, relation)
```

### 3. Constraint-Driven Validation
```python
def validate_constraints():
    constraints = load_all_yaml("**/constraints/*.yaml")
    for constraint in constraints:
        concepts = find_concepts_in_scope(constraint.scope)
        check_constraint(constraint, concepts)
```

## Benefits for AI Agents

1. **Predictable Structure** - No surprises in file organization
2. **Self-Describing** - Every file contains its own metadata
3. **Relationship Explicit** - Connections are machine-readable, not implicit
4. **Version Coherent** - All related content moves together
5. **Search Optimized** - Multiple entry points (by layer, domain, concept type)

## Export Strategy

From this flat-file structure, we can generate:
1. **Turtle files** - For semantic web compatibility
2. **JSON schemas** - For API validation
3. **Search indices** - For fast concept lookup
4. **Dependency graphs** - For impact analysis
5. **Documentation sites** - For human consumption

The flat files remain the source of truth, with exports being disposable artifacts. This maintains your progressive formalism principle while optimizing for AI agent consumption.