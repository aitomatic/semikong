# SemiKong

SemiKong is being organized as a combined semiconductor AI repository with two primary areas:

- `model/` for the semiconductor language model, training, inference, configs, and model-specific documentation
- `ontology/` for semiconductor ontology and knowledge-graph assets that will be merged in from `semicont`

## Repository Layout

```text
.
|-- model/
|   |-- README.md
|   |-- INSTALL.md
|   |-- common/
|   |-- configs/
|   |-- figures/
|   |-- inference/
|   |-- references/
|   |-- training/
|   |-- requirements.txt
|   `-- requirements-dev.txt
|-- ontology/
|   `-- README.md
|-- shared/
`-- tests/
```

## Current Status

- The existing SEMIKONG model assets and scripts now live under `model/`.
- `ontology/` is reserved for the semiconductor ontology merge.
- `shared/` is reserved for code or assets used by both areas.

## Where To Start

- Model documentation: [model/README.md](/Users/ctn/src/aitomatic/semikong/model/README.md)
- Model setup guide: [model/INSTALL.md](/Users/ctn/src/aitomatic/semikong/model/INSTALL.md)
- Ontology landing page: [ontology/README.md](/Users/ctn/src/aitomatic/semikong/ontology/README.md)
