![SEMIKONG teaser](figures/teaser.png)

# SEMIKONG Model

SEMIKONG is a semiconductor-focused language model project. This `model/` subtree contains the current model assets, training code, inference scripts, configs, references, and model-specific documentation.

The top-level repository is being organized around two major areas:

- `model/` for the semiconductor model
- `ontology/` for the semiconductor ontology and knowledge-graph assets

## Principals

- [Christopher Nguyen](https://github.com/ctn) (`ctn@aitomatic.com`)
- [William Nguyen](https://github.com/nguyennm1024) (`william@aitomatic.com`)

## Quick Links

- Dataset and benchmarks: <https://drive.google.com/drive/u/0/folders/1IjuVyP35-xBEe_i_KkG9MnE-4o7Eb7tq>
- Model weights: <https://huggingface.co/pentagoniac/SEMIKONG-70B>, <https://huggingface.co/pentagoniac/SEMIKONG-8b-GPTQ>
- Instruct chat API: launch with `python -m vllm.entrypoints.openai.api_server ...` as shown in [USAGE.md](/Users/ctn/src/aitomatic/semikong/model/USAGE.md)
- Paper: <https://arxiv.org/abs/2411.13802>

## Papers

- [SemiKong: Curating, Training, and Evaluating A Semiconductor Industry-Specific Large Language Model](https://arxiv.org/abs/2411.13802)
  Christopher Nguyen, William Nguyen, Atsushi Suzuki, Daisuke Oku, Hong An Phan, Sang Dinh, Zooey Nguyen, Anh Ha, Shruti Raghavan, Huy Vo, Thang Nguyen, Lan Nguyen, and Yoshikuni Hirayama. arXiv:2411.13802, 2024.

```bibtex
@article{semikong2024,
  title={SemiKong: Curating, Training, and Evaluating A Semiconductor Industry-Specific Large Language Model},
  author={Nguyen, Christopher and Nguyen, William and Suzuki, Atsushi and Oku, Daisuke and Phan, Hong An and Dinh, Sang and Nguyen, Zooey and Ha, Anh and Raghavan, Shruti and Vo, Huy and Nguyen, Thang and Nguyen, Lan and Hirayama, Yoshikuni},
  journal={arXiv preprint arXiv:2411.13802},
  year={2024}
}
```

## Start Here

- Setup and environment: [INSTALL.md](/Users/ctn/src/aitomatic/semikong/model/INSTALL.md)
- Usage and serving: [USAGE.md](/Users/ctn/src/aitomatic/semikong/model/USAGE.md)
- Commands: [Makefile](/Users/ctn/src/aitomatic/semikong/model/Makefile)
- Training config: [configs/training-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/training-config.yaml)
- Inference config: [configs/inference-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/inference-config.yaml)

## How To Use

From the repository root:

```bash
make -C model install
make -C model train
make -C model infer
```

If you need to change paths or parameters first, edit:

- [configs/training-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/training-config.yaml)
- [configs/inference-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/inference-config.yaml)

## Documentation

- Project overview and model summary: [docs/overview.md](/Users/ctn/src/aitomatic/semikong/model/docs/overview.md)
- Ecosystem, deployment, and references: [docs/ecosystem.md](/Users/ctn/src/aitomatic/semikong/model/docs/ecosystem.md)
- Governance, contributions, disclaimer, and license notes: [docs/governance.md](/Users/ctn/src/aitomatic/semikong/model/docs/governance.md)

## License

The code and repository contents in this project are distributed under the [MIT License](../LICENSE). Model weights, datasets, and third-party assets may carry separate upstream license terms.
