# SemiKong

SemiKong is an open-source semiconductor AI repository being organized around two complementary assets:

- a semiconductor-focused language model
- a semiconductor ontology and knowledge graph

The intent is to make this repository the shared home for both the model work and the structured domain knowledge that can support it.

SemiKong was developed in connection with the [AI Alliance](https://aialliance.org), an open community focused on open and responsible AI. The project reflects that open collaboration model while focusing specifically on semiconductor models and ontology assets.

## Repository Areas

- [model/](/Users/ctn/src/aitomatic/semikong/model): the semiconductor model work
- [ontology/](/Users/ctn/src/aitomatic/semikong/ontology): the semiconductor ontology and knowledge-graph work

## Principal

- [Christopher Nguyen](https://github.com/ctn) (`ctn@aitomatic.com`)

## Current Status

- The original SEMIKONG model project now lives under [model/](/Users/ctn/src/aitomatic/semikong/model).
- The ontology area is reserved for the planned merge from `aitomatic/semicont`.
- The repository is in a transitional phase from a single-project layout to a multi-area layout.

## Start Here

- model overview: [model/README.md](/Users/ctn/src/aitomatic/semikong/model/README.md)
- ontology overview: [ontology/README.md](/Users/ctn/src/aitomatic/semikong/ontology/README.md)

## Working Model

This repository currently uses:

- `develop` for active development work
- `main` as the promotion branch from `develop`
- `stable` as the protected default branch

Promotion flow:

- feature branches should merge into `develop`
- `develop -> main` happens by pull request
- `main -> stable` happens by pull request

## Roadmap

Near-term repository goals:

- merge the ontology assets from `aitomatic/semicont` into `ontology/`
- define what belongs in `shared/` versus remaining local to `model/` or `ontology/`
- add integration tests for workflows that combine the model and ontology
- improve the root-level docs as the multi-area architecture becomes more concrete

## License

The repository code and checked-in contents are distributed under the [MIT License](/Users/ctn/src/aitomatic/semikong/LICENSE).

Referenced model weights, datasets, and third-party assets may be governed by separate upstream licenses.
