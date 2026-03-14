<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../model/figures/teaser.png" width="200px">
  <source media="(prefers-color-scheme: light)" srcset="../model/figures/teaser.png" width="200px">
  <img alt="SEMIKONG teaser" src="../model/figures/teaser.png" width="200px">
</picture>

</div>

# Ontology

This subtree contains SemiKong's semiconductor ontology and knowledge-graph work, imported from `semicont` and now housed inside the broader SemiKong repository.

Its purpose is to provide an open semantic foundation for semiconductor knowledge that can support interoperability, provenance, validation, and downstream model- or workflow-level applications.

We intend to contribute the SemiKong ontology work into the broader SEMI standards and interoperability effort where that alignment is useful and appropriate.

## What Is Here

- ontology modules in Turtle under [ontology/ontology/](ontology/)
- architecture and methodology docs under [ontology/docs/](docs/)
- SHACL shapes under [ontology/shapes/](shapes/)
- examples under [ontology/examples/](examples/)
- curation and ontologist workflow materials under [ontology/ontologist/](ontologist/)

## Design Intent

The ontology work is intended to make semiconductor knowledge:

- interoperable across organizations and systems
- auditable through explicit provenance
- reusable across practical workflows
- modular enough for public core plus private extensions

This aligns with the original `semicont` direction and now serves as the ontology side of SemiKong's combined model-plus-ontology repository.

## Start Here

- manifesto: [MANIFESTO.md](MANIFESTO.md)
- commands: [Makefile](Makefile)
- ontology source overview: [ontology/ontology/README.md](ontology/README.md)
- architecture: [ontology/docs/architecture.md](docs/architecture.md)
- industry hierarchy: [ontology/docs/semiconductor-industry-ontology-hierarchy.md](docs/semiconductor-industry-ontology-hierarchy.md)
- ontologist workflow: [ontology/ontologist/README.md](ontologist/README.md)

## How To Use

From the repository root:

```bash
make -C ontology audit
make -C ontology benchmark
make -C ontology export-agentic
```

## Current Shape

The imported ontology currently includes:

- shared and physics foundations
- industry-layer modules such as integrators, EDA, foundry/IDM, OSAT, WFE, materials, and supply chain
- validation shapes and curation assets

The canonical semantic source remains the Turtle content under [ontology/ontology/](ontology/).

## Contributor Signal

Initial contributor called out in the imported ontology materials:

- [Christopher Nguyen](https://github.com/ctn) (`ctn@aitomatic.com`)

## License Note

The imported ontology subtree includes its own [ontology/LICENSE](LICENSE). The repository as a whole is MIT-licensed at the top level, but ontology assets may also carry their own preserved licensing and provenance context from the imported source.
