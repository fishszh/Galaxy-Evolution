## 星系并合过程再现

利用LSTM学习星系并合过程，尝试再现星系并合过程。



- 由于数据源较少，只是学到了几个固定的模式的并合过程
- 这里只是根据图像信息，学习并合过程，学习结果并不一定受物理机制的限制。
- 如果我们能够获得具体的物理参数，整个结果可能会得到一定的提升。

## 数据来源

数据主要来自[Gadget 2](https://wwwmpa.mpa-garching.mpg.de/gadget/)，[Flash](http://flash.uchicago.edu/site/research/)和[ILLUSTRIS](https://www.illustris-project.org/data/)模拟的结果。目前数据源并不多，可以从相关论文作者的论文中找到相关模型的视频文件。

在此感谢，师兄提供的数据。

## 参考文献

- 

## 模型和技术

### Simple ConvLSTM

采用多层`ConvLSTM`，简单的堆叠来实现星系并合过程的短期预测。

`Loss function`采用`MSE`.

## 结果

### Simple ConvLSTM

- 短期预测

$$
\hat{v}_{n+1} = \argmax p(v_{n+1}|\tilde{v}_{n-j+1}, \tilde{v}_{n-j++2},\ldots,\tilde{v}_{n})
$$

- 长期预测

$$
\hat{v}_{n+k} = \argmax p(v_{n+k}|\tilde{v}_{n-j+1}, \tilde{v}_{n-j++2},\ldots,\tilde{v}_{n},\hat{v}_{n+1}\ldots,\hat{v}_{n+k-1})
$$

#### MSE效果

|训练集|测试集|测试|
|:----:|:----:|:---:|
|short term|short term|long term(k=10)|
|![](./imgs/SimpleConvLSTM/convlstm_com_mse_0250.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_mse_1200.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_mse_1200_10.gif)|
|![](./imgs/SimpleConvLSTM/convlstm_com_mse_0300.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_mse_1500.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_mse_1500_10.gif)|
|![](./imgs/SimpleConvLSTM/convlstm_com_mse_0500.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_mse_2000.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_mse_0500_10.gif)|

#### SSIM效果

|训练集|测试集|测试|
|:----:|:----:|:---:|
|short term|short term|long term(k=20)|
|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_0250.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_1200.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_ssim_1200_10.gif)|
|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_0300.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_1500.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_ssim_1500_10.gif)|
|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_0500.gif)|![](./imgs/SimpleConvLSTM/convlstm_com_ssim_2000.gif)|![](./imgs/SimpleConvLSTM/convlstm_gen_ssim_0500_10.gif)|
|||![](./imgs/SimpleConvLSTM/convlstm_gen_ssim_0500_50.gif)|


