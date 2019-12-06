# Lipreading-DenseNet3D
DenseNet3D Model In "DenseNet3D Model In "LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild", https://arxiv.org/abs/1810.06990

![Sample of the proposed LRW-1000](banner.png)

## Introduction   

This respository is implementation of the proposed method in [LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild](). Our paper can be found [here](https://arxiv.org/pdf/1810.06990.pdf).

## Dependencies
* Python 3.6.7   
* PyTorch 1.0+
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
@inproceedings{yang2019lrw,
  title={LRW-1000: A Naturally-Distributed Large-Scale Benchmark for Lip Reading in the Wild},
  author={Yang, Shuang and Zhang, Yuanhang and Feng, Dalu and Yang, Mingmin and Wang, Chenhao and Xiao, Jingyun and Long, Keyu and Shan, Shiguang and Chen, Xilin},
  booktitle={2019 14th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2019)},
  pages={1--8},
  year={2019},
  organization={IEEE}
}
```
