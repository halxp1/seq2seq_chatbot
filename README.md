# seq2seq_chatbot

## 环境依赖

| 程序         | 版本      |
| ---------- | ------- |
| python     | 3.52    |
| Tensorflow | 1.10.0  |
| CUDA       | 9.0.103 |
| cuDNN      | 7.0     |



## Run

- **预料获取**

  1. `wget https://lvzhe.oss-cn-beijing.aliyuncs.com/dgk_shooter_min.conv.zip`

     输出 ：dgk_shooter_min.conv.zip

  2. `unzip dgk_shooter_min.conv.zip`

     输出 ： dgk_shooter_min.conv

-  **文本处理**

  1. `python data_process.py`

     输出： chatbot.pkl

- **训练数据**

  1. `python train_anti.py`

     在./model 得到训练好的模型

- chatbot测试

  1. `python test_anti.py`

- ![](https://ws3.sinaimg.cn/large/006tNbRwgy1fxo7n94nbfj30id0biacl.jpg)

  [**个人博客详解**](https://blog.csdn.net/hl791026701/article/details/84404901)

  有问题欢迎在Issues留言 ，如果觉得不错请给个star !
