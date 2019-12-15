Downloaded from  “https://github.com/galsang/CNN-sentence-classification-pytorch”,trying to understand every row of this project.

代码主要组成部分：
1.加载数据
2.非rand模型加载预训练词向量
3.textCNN模型

1.加载数据
TREC数据集是由可分为6类问题类型的问题数据集构成的，格式为：category：question/n。数据集分为train、test。train中又可以读取出train、dev，dev可以用于选择最佳模型。
这份代码中并没有用torchtext这类模块，或者写好vocab类似的辅助函数，而是用简洁有力的几条代码就获得textCNN可以接受的输入格式。
read_TREC()返回data，包含train、dev、test的x，y。train_x是list of list，形如[['Do'...],...[]],train_y是['DSEC'...]
可以很简单地获得最长的句子长度max_sent_len，也可以建立起word和label的vocab。
在train函数中将x，y数字化，并对x进行pad。

2.加载pretrained word embedding
用genism中的keyedwordvectors加载，并用vocab的顺序形成这个数据集中可以用到的embedding，预训练中没有的用uniform均匀性分布生成。最后，添加unk、pad对应的embedding。

3.textCNN由conv1d、relu、maxpool1d、fc构成。
conv1d（input_channel,output_channel,kernel_size,stride = 1,padding=0...）
input_channel在nlp中可以认为是词向量维度，output_channel是filter个数，kernel_size可以理解为每次卷积处理的词数，在input_channel不为1时，实际kernel_size = （参数kernel_size,input_channel）。下图说明的很清楚：

# Convolutional Neural Networks for Sentence Classification

This is the implementation of [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181) on **Pytorch**.


## Results

Below are results corresponding to all 4 models proposed in the paper for each dataset.
Experiments have been done with a learning rate = ~~0.1~~ 1 up to ~~300~~ 100 epochs and all details are tuned to follow the settings defined in the paper. 

(17.02.06) There was a bug that the best model is selected based on test acc, not on dev acc. The error has been corrected and so do results. Also note that the results can be somewhat different whenever the code is executed. Thanks!
(WARNING: CV should be performed in case of MR, but only the random selection of dev(10%) and test(10%) set is done for simplicity.)

(Measure: Accuracy)

| Model        | Dataset  | MR   | TREC |
|--------------|:----------:|:------:|:----:|
| Rand         | Results  | 67.7 | 88.2 |
|              | Baseline | 76.1 | 91.2 |
| Static       | Results  | 79.7 | **93.2** |
|              | Baseline | 81.0 | 92.8 |
| Non-static   | Results  | 80.1 | **94.4** |
|              | Baseline | 81.5 | 93.6 |
| Multichannel | Results  | 79.8 | **93.6** |
|              | Baseline | 81.1 | 92.2 |     92.2验证正确


## Specification
- **model.py**: CNN sentnece classifier implementation proposed by Y. Kim.
- **run.py**: train and test a model with configs. 
 

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- GPU: GTX 1080


## Requirements

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.
You should download this file and place it in the root folder.

Also you should follow library requirements specified in the **requirements.txt**.

    numpy==1.12.1
    gensim==2.3.0
    scikit_learn==0.19.0


## Execution

> python run.py --help

	usage: run.py [-h] [--mode MODE] [--model MODEL] [--dataset DATASET]
				  [--save_model] [--early_stopping] [--epoch EPOCH]
				  [--learning_rate LEARNING_RATE] [--gpu GPU]

	-----[CNN-classifier]-----

	optional arguments:
	  -h, --help            show this help message and exit
	  --mode MODE           train: train (with test) a model / test: test saved
							models
	  --model MODEL         available models: rand, static, non-static,
							multichannel
	  --dataset DATASET     available datasets: MR, TREC
	  --save_model          whether saving model or not
	  --early_stopping      whether to apply early stopping
	  --epoch EPOCH         number of max epoch
	  --learning_rate LEARNING_RATE
							learning rate
	  --gpu GPU             the number of gpu to be used
 
