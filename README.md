# Lipreading-DenseNet3D
DenseNet3D Model In https://arxiv.org/abs/1810.06990
## Introduction   

This respository is implementation of the proposed method in [LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild](). Our paper can be found [here](https://arxiv.org/pdf/1810.06990.pdf).

## Dependencies
* python 3.6.7   
* pytorch 1.0.0.dev20181103
* Others
## Dataset
This model is pretrained on LRW with RGB lip images(112×112), and then tranfer to LRW-1000 with the same size. We train the model end-to-end.   
## Training And Testing
You can train or test the model as follow:
```
python main.py options_lip.toml
```
Model architecture details and data annotation items are configured in `options_lip.toml`. Please pay attention that you may need modify the code in `options_lip.toml` and change the parameters `data_root` and `index_root` to make the scripts work just as expected. 

Another implmentation: https://github.com/NirHeaven/D3D

## Reference

If this repository was useful for your research, please cite our work:

```
@article{shuang18LRW1000,
  title={LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild},
  author={Shuang Yang, Yuanhang Zhang, Dalu Feng, Mingmin Yang, Chenhao Wang, Jingyun Xiao, Keyu Long, Shiguang Shan, Xilin Chen},
  booktitle={arXiv},
  year={2018}
}
```
