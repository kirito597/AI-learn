"""
第一步：构建实验预料库
"""
# 定义一个句子列表，后面会使用这些句子来训练CBOW和Skip-Gram模型
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student"]
# 将所有句子连接在一起，然后用空格分隔成词汇
words = ' '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词汇映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词汇
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)  # 计算词汇表的大小
print("词汇表：", word_list)  # 打印词汇表
print("词汇到索引的字典：", word_to_idx)  # 打印词汇到索引的字典
print("索引到词汇的字典：", idx_to_word)  # 打印索引到词汇的字典
print("词汇表大小：", voc_size)  # 打印词汇表大小

"""
第二步：生成Skip-Gram训练数据
"""
# 生成Skip-Gram模型的训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将单词前后各N个单词作为相邻单词
            for neighbor in sentence[max(idx-window_size, 0):
                        min(idx+window_size+1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词和当前单词作为一组训练数据
                    data.append((neighbor, word))
    return data

# 使用函数创建Skip-Gram模型的训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印为编码的Skip-Gram数据序列
print("Skip-Gram数据序列(未编码)：", skipgram_data)

"""
第三步：对Skip-Gram数据进行One-Hot编码
"""
# 定义One-Hot编码函数
import torch  # 导入torch库

def one_hot_encoding(word, word_to_idx):
    # 创建一个全为0的向量，长度与词汇表大小相同
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1  # 将对应词汇的索引位置置为1
    return tensor  # 返回生成的One-Hot编码后的向量

# 展示One-Hot编码前后的数据
word_example = "Teacher"
print("One-Hot编码前的单词：", word_example)
print("One-Hot编码后的向量：", one_hot_encoding(word_example, word_to_idx))

# 展示编码后的Skip-Gram数据序列
print("Skip-Gram数据序列(已编码)：", [(one_hot_encoding(context, word_to_idx),
        word_to_idx[target]) for context, target in skipgram_data[:3]])

"""
第四步：定义Skip-Gram模型
"""
# 定义Skip-Gram模型
import torch.nn as nn  # 导入neural network

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表大小到嵌入大小的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(vocab_size, embedding_size, bias=False)
        # 从嵌入大小到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, X):  # X:[batch_size, voc_size]
        # 生成隐藏层：[batch_size, embedding_size]
        hidden_layer = self.input_to_hidden(X)
        # 生成输出层：[batch_size, voc_size]
        output_layer = self.hidden_to_output(hidden_layer)
        return output_layer

embedding_size = 2  # 设置嵌入层的大小，这里设置为2方便展示
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化Skip-Gram模型
print("Skip-Gram模型：", skipgram_model)

"""
第五步：训练Skip-Gram模型
"""
# 训练Skip-Gram模型
learning_rate = 0.001  # 设置学习率
epochs = 1000  # 设置训练轮数
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim  # 导入随机梯度下降优化器
optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)

# 开始训练循环
loss_values = []  # 用于记录损失值
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失值
    for context, target in skipgram_data:
        # 将中心词转换为One-Hot向量
        X = one_hot_encoding(target, word_to_idx).float().unsqueeze(0)
        # 将周围词转换为索引值
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)
        y_pred = skipgram_model(X)  # 计算预测值
        loss = criterion(y_pred, y_true)  # 计算损失值
        loss_sum += loss.item()  # 累计损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch+1) % 100 == 0:  # 每100轮打印一次损失值
        avg_loss = loss_sum/len(skipgram_data)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
        loss_values.append(avg_loss)  # 记录损失值

# 绘制训练损失曲线
import matplotlib.pyplot as plt  # 导入matplotlib库
plt.plot(range(1, epochs//100 + 1), loss_values)  # 绘图
plt.title("Training Loss")
plt.xlabel("Epoch (x100)")
plt.ylabel("Loss")
plt.show()

"""
第六步：显示习得的词向量
"""
# 输出Skip-Gram习得的词嵌入
print("\nSkip-Gram模型的词嵌入：")
for word, index in word_to_idx.items():
    # 注意：input_to_hidden.weight 的形状是 [embedding_size, vocab_size]
    # 所以需要取第 index 列，而不是第 index 行
    embedding = skipgram_model.input_to_hidden.weight[:, index].detach().numpy()
    print(f"{word}:\t{embedding}")

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制二维词向量图
fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    # 获取词向量（注意索引方式）
    vec = skipgram_model.input_to_hidden.weight[:, idx].detach().numpy()
    # 绘制词向量点
    ax.scatter(vec[0], vec[1])
    # 添加文本标签
    ax.text(vec[0], vec[1], word, fontsize=12)

plt.title("2维词嵌入")
plt.xlabel("向量维度1")
plt.ylabel("向量维度2")
plt.grid(True)
plt.show()