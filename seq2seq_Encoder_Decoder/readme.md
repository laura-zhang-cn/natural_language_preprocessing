# 核心思路
**1. Encoder-Decoder机制**  
**2. Encoder 编码X输入向量，将最后时刻的具体输出和记忆状态 作为思想向量states输出**  
**3. Decoder 将添加了起始符的Y和encoder-states同时输入，将添加了结尾符的Y作为目标输出**  
**4. 通过Decoder多次predict,循环的更新和调用states,直到生成结尾符或达到输出的最大长度，停止生成序列（句子）。**  


# 效果
max sentence length = 10
neurons number= 128
batch size= 64
epoch = 100

测试集序列的**完全准确率: 97.3%**
（测试集序列 和 训练集序列 是独立的）
