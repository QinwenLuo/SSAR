# SSAR
The official code of our __ICML 2025__ paper [Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](https://www.arxiv.org/abs/2505.19923).

## Acknowledgments

This project makes use of the following open-source projects:

- **[CORL](https://github.com/tinkoff-ai/CORL)**: Implementation of the offline training process.

***
## Install
For installation instructions, please refer to the [CORL](https://github.com/tinkoff-ai/CORL) repository for detailed guidance.
***

## Run
### cql as the backbone algorithm
```
python cql.py --env hopper-medium-v2 --seed 0
```

### td3+bc as the backbone algorithm
```
python td3_bc.py --env hopper-medium-v2 --seed 0
```
