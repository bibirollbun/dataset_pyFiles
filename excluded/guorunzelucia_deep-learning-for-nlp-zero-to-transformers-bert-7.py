import numpy as np  # 数值计算基础库，用于处理多维数组和矩阵运算
import pandas as pd  # 数据分析库，提供DataFrame结构处理表格数据 也可以读取CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm  # 进度条显示工具，用于可视化循环进度
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Kaggle服务器已经装好环境，本地运行需要下载GPU版本的Tensorflow，配置环境
# tensorflow是一个深度学习框架

from keras.models import Sequential  # Sequential 是最简单的模型容器，按顺序堆叠各层，用于快速搭建线性堆叠型网络。
from keras.layers import LSTM, GRU, SimpleRNN  # 这就是我们要训练的神经元，用于处理序列数据的神经元
# SimpleRNN：基础循环层，存在梯度消失问题 
# LSTM：长短期记忆网络，通过门控机制解决长期依赖问题 
# GRU：门控循环单元，LSTM的简化版（合并了遗忘门和输入门）

# Dense、Activation、Dropout 现在都在 keras.layers 里，改成这样：
from keras.layers import Dense, Activation, Dropout
# Dense：全连接层，用于序列模型的最后几层，将 RNN 输出映射到目标维度。
# Activation：激活函数层，可单独指定如 Activation('relu')、Activation('softmax')。
# Dropout：在训练时随机丢弃一定比例的神经元，防止过拟合。

from keras.layers import Embedding  # 词嵌入向量，词嵌入层（将离散符号映射为稠密向量）
from keras.layers import BatchNormalization  # 批标准化层，加速训练并稳定收敛
from tensorflow.keras.utils import to_categorical   # np_utils 提供一些工具函数，比如 to_categorical 将整数标签转成 one-hot 编码。
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
# Conv1D + MaxPooling1D：一维卷积和池化层，可提取局部 n-gram 特征
# GlobalMaxPooling1D：对整个时间步取最大值，常用于提炼最显著特征
# Flatten：把多维张量展开为一维
# Bidirectional：将 RNN 包装为双向 RNN，同时考虑正向和反向上下文
# SpatialDropout1D：在时间步维度上丢弃整条特征通道，防过拟合

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer # keras.preprocessing是处理文本的包，sequence, text是处理具体功能的模块
from keras.callbacks import EarlyStopping  # 调用于在验证集指标（如 loss）不再改善时提前终止训练，防止过拟合并节省计算资源
from sklearn.metrics import accuracy_score

# 可视化工具链
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff



#检测硬件环境，设置分布式策略 
try:
    # 尝试检测是否存在 TPU（张量处理单元），适用于 Google Cloud / Kaggle 等平台
    # 如果 TPU 存在，将返回一个 TPUClusterResolver 对象（自动识别环境变量）
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    # 如果没有 TPU（本地运行或无 TPU 支持），则设为 None
    tpu = None

# 如果检测到 TPU，进行初始化并创建适用于 TPU 的分布式策略
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)  # 连接到 TPU 集群
    tf.tpu.experimental.initialize_tpu_system(tpu)  # 初始化 TPU 系统（分配资源）
    strategy = tf.distribute.experimental.TPUStrategy(tpu)  # 创建基于 TPU 的分布式训练策略
else:
    # 如果没有 TPU，则默认使用 CPU 或单 GPU 的策略
    strategy = tf.distribute.get_strategy()  # 适用于单机多 GPU 或 CPU 的默认策略

# 打印当前同步训练的副本数（例如 TPU 上通常是 8 个）
print("REPLICAS: ", strategy.num_replicas_in_sync)




train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv') #读取训练集，使用的competitions下的一个数据集
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')#读取验证集
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv') #读取测试集
#在右边的input模块 Add Input可以添加本地数据集
#调用 Pandas 的 read_csv() 函数，把位于指定路径的 CSV 文件读入成一个 DataFrame 对象，赋值给变量 train，validation，test


train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
#去除一些不需要的列并在原数据集上修改，axis=1代表在列方向上删除


train = train.loc[:12000,:] #使用loc索引，只使用前12000行的数据
train.shape


#算例：
df = pd.DataFrame({
    'comment_text': ['a','b','c'],
    'toxic': [0,1,0],
    'severe_toxic':[0,0,0],
    'obscene':[1,0,1]
})
# 丢掉不需要的列
df.drop(['severe_toxic','obscene'], axis=1, inplace=True)
# 只取前 2 行（示例中原来有 3 行）
df = df.loc[:1, :]


output：
comment_text	toxic
a	              0
b	              1


train.head() # 查看前几行


train['comment_text'].apply(lambda x:len(str(x).split())).max()
#取出 train DataFrame 中名为 comment_text 的这一列，得到一个 Series，其中每个元素都是一条评论（或空值／NaN）
#对于这一列的每一行作为一个x变成字符串，然后依据空格进行切分，并对每一行进行计数，并在所有返回的词数中取最大值，得到训练集中那条最长评论的词数
#将当前元素 x 强制转成字符串的好处是如果有缺失值（NaN），str(x) 会得到 'nan'，避免报错


#算例：
s = pd.Series(["hello world", "", None])
# len(str(x).split()) 依次为：2, 0, 1 (‘None’→'None'→['None'])
max_len = s.apply(lambda x: len(str(x).split())).max()
# max_len == 2


def roc_auc(predictions,target): 
    #roc_auc：函数名，意指“Receiver Operating Characteristic Area Under Curve”
    #predictions是模型对每个样本输出的“正类”概率或得分（浮点数数组）。target是对应的真实二分类标签（0 或 1 的整数或布尔数组）。
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    #计算 ROC 曲线点
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)#False Positive Rate（假正例率） True Positive Rate（召回率）
    return roc_auc
    #可以试一试也可以用后面的准确率


#算例：
y_true = [0,1,1,0]
y_score = [0.2,0.8,0.6,0.4]
fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
# fpr=[0. ,0.5,1. ], tpr=[0.,1.,1.]
auc = metrics.auc(fpr, tpr)
# auc == 1.0 AUC = 面积 = 1.0（完美分类）


#将训练集中的评论文本和标签按比例随机拆分成训练子集和验证子集
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, #.values 把 Series 转成 NumPy 数组
                                                  stratify=train.toxic.values, 
                                                  random_state=42, #随机种子，保证每次拆分结果一致，方便复现实验。
                                                  test_size=0.2, shuffle=True)#在拆分前先打乱样本顺序，默认即为 True

#stratify=train.toxic.values表示分层抽样 保证训练集与验证集里面的标签比例分布一致，也就是说数据集不能完全随机，避免类别失衡影响模型评估。
#数据集中标签分布比例是2比8则分割得到的训练集和测试集中标签比例也是2比8


# using keras tokenizer here
token = Tokenizer(num_words=None) #使用词嵌入模块Tokenizer，初始化一个分词器
#text 对应 keras.preprocessing.text，提供了 Tokenizer、text_to_word_sequence、one_hot 等文本处理工具  注意是一种工具而不是文本
#num_words=None 表示不过滤词频，Tokenizer 会收录训练语料里出现的所有不同单词。
#如果你希望只保留出现最频繁的前 N 个词，可将 num_words=N，这样在后续把词转成索引时，会忽略索引 ≥ N 的词
max_len = 1500 
#后面 pad_sequences 会把所有序列统一做或截到长度 max_len前面那句 train['comment_text'].apply(...).max() 得到最大词数）
#！！训练深度机器学习模型时硬性要求输入的数据长度一样，如果不一样的话会自行截断影响数据的完整性

token.fit_on_texts(list(xtrain) + list(xvalid)) #将训练集 xtrain 和验证集 xvalid 的文本合并，统计所有单词的频率并构建词典
xtrain_seq = token.texts_to_sequences(xtrain) #将每条文本（每条样本）转换为对应的整数序列（根据 word_index 映射）。
#若 xtrain = ["cat sat", "the cat"]，word_index = {"cat":1, "sat":2, "the":3}，则输出：xtrain_seq = [[1, 2], [3, 1]]
xvalid_seq = token.texts_to_sequences(xvalid)

#RNN/LSTM/GRU 模型要求每个批次的输入长度一致，必须把所有序列填充到相同长度！！！
#zero pad the sequences 填补长度为1500 将所有需要序列用0填补到1500长度
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
#序列长度小于 maxlen 时，默认在前面（padding='pre'）用 0 填充；
#序列长度超出 maxlen 时，默认在前面（truncating='pre'）截掉多余部分。
#可以设置 padding='post'、truncating='post'，让填充/截断都在末尾进行

word_index = token.word_index 
#把 Tokenizer 生成的词典映射保存到变量 word_index，方便后续要：
 #查看某个词的索引：word_index.get("toxic")
 #构建 Embedding 层时指定 vocab_size = len(word_index) + 1（加 1 是因为索引从 1 开始，0 保留给填充用）
 #导出词典做分析或反向映射（索引→词）时，也需要它


#算例：Tokenizer 构建词典 → 序列化
texts = ["cat sat", "the cat sat"]
token = Tokenizer()
token.fit_on_texts(texts)
# word_index = {'sat':1, 'cat':2, 'the':3}
seqs = token.texts_to_sequences(texts)
# => [[2,1], [3,2,1]]


#算例：pad_sequences 填充/截断
seqs = [[2,1], [3,2,1]]
padded = pad_sequences(seqs, maxlen=4)
# 默认 pre-padding → [[0,0,2,1], [0,3,2,1]]


len(word_index) #字典里不同的词的个数


# A simpleRNN without any pretrained embeddings and one dense layer
model = Sequential()#初始化一个线性堆叠的神经网络容器，它将各层按照添加顺序线性堆叠，刚创建时 model 为空，尚未配置输入／输出
model.add(Embedding(len(word_index) + 1,300,input_length=max_len))
#第一层 嵌入层 用于把词索引映射到稠密向量空间，是 NLP 中常用的“词嵌入”组件，在字典长度上加1（因为还用0填补），每个单词的维度是300（自己设置），第三个参数设置每个样本里面序列的长度是多少（1500）
model.add(SimpleRNN(100))
#第二层 简单RNN层 100个RNN（有记忆功能的普通神经元）100为隐藏状态维度即每个时间步输出100维向量
model.add(Dense(1, activation='sigmoid')) 
#第三层 全连接输出层  每一个RNN有一条边连接到 Dense里面的一个神经元 输出维度为1即一个神经元的输出并且将输出使用Sigmoid函数压缩到（0,1）
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #对模型进行其他设置
# 损失函数binary_crossentropy是二分类交叉熵损失，适用于正负样本的二分类任务
# 优化器adam自适应矩估计优化器，结合了RMSProp和动量法的优点     上课讲过！
#评估指标accuracy是分类准确率（正确预测数/总样本数）
model.summary()
#第一层 none 还没有数据这是一个空模型 数据长度 数据个数  最后是本层有多少个参数


#对模型进行训练
model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64) #前12000条数据然后按比例切割，Multiplying by Strategy to run on TPU's
#xtrain_pad：形状 (N_train, max_len) 的整数序列矩阵，表示训练集的所有样本（已做填充）
#ytrain：形状 (N_train,) 的标签向量（0 或 1）
#每个 epoch 会把训练集拆成若干个 batch，依次喂入模型，计算损失并通过 Adam 优化器更新所有可训练参数，也就是说每进行一个epoch，就会重新划分batch
#每一个深度机器学习都需要换分batch后再输入（有点类似于K折交叉验证的划分，K折交叉验证只适用于简单的运算）


###在验证集上的效果
scores = model.predict(xvalid_pad)# 输出的是概率返回一个形状 (N_valid, 1) 的数组 scores，每个值在 [0,1] 之间，表示是类别1的概率
predicted_labels = (scores > 0.5).astype(int)  # 转为0/1标签，固定阈值0.5，将概率转为类别标签；适用于类别平衡数据，不平衡时需调整阈值
print(predicted_labels[:10])  # 输出前10个预测标签


accuracy = accuracy_score(yvalid, predicted_labels)#验证准确率，来自 sklearn.metrics，比较真实标签 yvalid 与预测标签 predicted_labels，返回准确率（正确预测占总样本比例）
print(f"Accuracy: {accuracy:.4f}")  # 输出格式化为4位小数
#print("Auc: %.2f%%" % (roc_auc(scores,yvalid))) 
#可选 也可以用前面定义的 roc_auc 函数计算 ROC AUC，AUC 能反映模型在不同阈值下的整体区分能力，更适合不均衡数据集的评估


scores_model = []#创建一个空列表 scores_model，用于存储不同模型的评估结果
scores_model.append({'Model': 'SimpleRNN','AUC_Score': accuracy})#将当前 SimpleRNN 的“准确率”或“AUC”以字典形式添加进去方便比较不同模型的性能


xtrain_seq[:1]


# load the GloVe vectors in a dictionary:Glove是一个词向量的方法
embeddings_index = {} #右边的文件是一个模型训练好了训练好的词向量从里面调出来我自己需要的单词的词嵌入相当于自己可以不用做词向量化
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8') # 打开包含 GloVe 向量的文件（已提前上传到 Kaggle），设置编码为 utf-8
#glove.840B.300d.txt 是一个 大规模预训练词向量文件：包含约 21 亿个词的训练数据（Common Crawl 语料库）；词向量维度是 300（即每个词映射成一个 300 维实数向量）；文件大小接近 2.1GB，里面大约有 220万个单词的词向量。
for line in tqdm(f): ## tqdm 用于显示加载进度条
    #逐行读取文件，每行格式如下： word val1 val2 val3 ... val300   nginx apple 0.123 -0.456 0.678 ... -0.999
    values = line.split(' ') ## 用空格分割这一行，得到一个列表
    word = values[0] # 提取词
    coefs = np.asarray([float(val) for val in values[1:]])  # 把词向量转换为 numpy 的浮点型数组
    embeddings_index[word] = coefs  # 存入字典，key 为词，value 为对应的 300 维向量
f.close()

print('Found %s word vectors.' % len(embeddings_index))


print(embeddings_index.get("toxic"))  # 应该是一个 300 长度的 array
print(embeddings_index.get("unseenword"))  # 如果返回 None，说明词向量中没有该词


# 创建一个空字典，用于存储每个单词及其对应的词向量（即 word -> vector 的映射）把调出来的放到一个矩阵中
embedding_matrix = np.zeros((len(word_index) + 1, 300))
#word_index 是前面用 Tokenizer 建好的词→索引映射，键是词，值是索引（从 1 开始）；加 1 是为了给索引 0（用于填充 pad_sequences 时产生的 0）也预留一行 ；矩阵的列数等于 GloVe 词向量的维度
#这就是要把预训练好的 300 维向量放入的“容器”，后面会传给 Keras 的 Embedding 层作为初始权重
for word, i in tqdm(word_index.items()): #遍历词典中的每个词及其索引
    #包裹后可以看到遍历进度条，尤其当词典几万甚至十几万条时，能直观了解加载进度。
    embedding_vector =embeddings_index.get(word) #从预训练向量字典中取出该词的向量
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    #如果该词在 GloVe 词表里有预训练向量，就返回长度为 300 的 ndarray
#重新创建一个矩阵的目的：1. Keras的Embedding层接口要求 2. 索引对齐 3.效率与优化：每个 batch 上千、上万个词，每步都要并行地拿出它们的向量。一次性把所有词的向量预载到一个大矩阵里，让底层高效地做 GPU/CPU 的并行查表。


print(embedding_matrix.shape)
print(np.sum(embedding_matrix))


#算例：
词汇索引	词	原始词向量 (embeddings_index)	   填入 embedding_matrix
0	   pad  	  —	                        [0.0, 0.0, 0.0] (预留给填充)
1	   cat  [0.1, 0.2, 0.3]              	[0.1, 0.2, 0.3]
2	   dog	[0.4, 0.5, 0.6]               	[0.4, 0.5, 0.6]
3	 elephant 未在 embeddings_index 中	    [0.0, 0.0, 0.0] (保持默认)

结果矩阵：
     dim1  dim2  dim3
pad   0.0   0.0   0.0
cat   0.1   0.2   0.3
dog   0.4   0.5   0.6
elephant 0.0 0.0 0.0



#构建一个使用 预训练 GloVe 词向量的LSTM模型来处理文本分类任务
with strategy.scope(): #建立LSTM模型
    #分布式训练策略下创建模型，如使用 Kaggle 或 Google Colab 上的 TPU 或多个 GPU 时常用
    
    # A simple LSTM with glove embeddings and one dense layer
    model = Sequential ()#使用 Keras 的 Sequential API 按顺序堆叠模型层。
#第1层：词嵌入层（Embedding）
    model.add(Embedding(len(word_index) + 1,  # 总词数（包括 index=0 的 pad token）
                     300,# 每个词的向量维度（GloVe 300d）
                     weights=[embedding_matrix],#权重是单词的向量直接用不需要训练 使用准备好的 GloVe 词向量矩阵
                     input_shape=(max_len,), # 每条输入文本被填充/截断后的长度（如1500）
                     trainable=False))#不需要训练，不更新这些词向量
    #算法示例：将输入的整数序列（如 [3, 5, 1, 99]）转换成向量序列（如 [[v3], [v5], [v1], [v99]]），其中每个 v 是一个 300维的向量。
#第2层：LSTM 层
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))#随机去除百分之30防止过拟合
    #100为 LSTM 的隐藏单元数（每个时间步输出一个100维向量） #对输入的 dropout（防止过拟合）#对隐藏状态的 dropout（LSTM 的“记忆”部分也加点随机性）
    # dropout表示每一步输入和记忆有 30% 的概率被丢弃，这有助于防止过拟合。
#第3层：输出层   
    model.add(Dense(1, activation='sigmoid'))
    #Dense(1)：输出一个标量值（通常是概率）,使用·sigmoid函数，将输出变成 [0, 1] 之间的概率，代表为toxic=1的概率
  
    model.compile(loss='binary_crossentropy',  # 二分类交叉熵损失函数，适用于 toxic / non-toxic、positive / negative 这类任务
                  optimizer='adam', # Adam 是一种自适应学习率的优化器
                  metrics=['accuracy']) # 训练时输出准确率
    
model.summary()
#打印模型的结构，包括每一层的名称、输出形状、参数数量等，便于检查网络结构是否搭建正确。


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
#训练集中每64个数据为一组共150batch，每一个batch送入模型就更新一次参数然后再重新打包（epoch） 所有的神经网络都是要分batch


scores = model.predict(xvalid_pad) #使用你刚刚训练好的 LSTM 模型，对验证集 xvalid_pad 进行预测,返回的是一个概率值（Sigmoid 函数输出的结果，范围在 [0, 1] 之间），表示每个样本属于“toxic”（有毒）的置信度
predicted_labels = (scores > 0.5).astype(int)  # 转为0/1标签,逻辑判断，若概率大于0.5，返回 True，否则 False  #将布尔值转为整数，True → 1, False → 0
# 因为模型最后一层用了 sigmoid，表示“为 toxic 的概率”，默认二分类的判断阈值是 0.5
accuracy = accuracy_score(yvalid, predicted_labels) #计算预测标签 predicted_labels 与验证集真实标签 yvalid 之间的准确率
print(f"Accuracy: {accuracy:.4f}")  # 输出格式化为4位小数


scores_model.append({'Model': 'LSTM','AUC_Score': accuracy}) 
#将当前 LSTM 的“准确率”或“AUC”以字典形式添加进去方便比较不同模型的性能


with strategy.scope(): #表示在指定的分布式策略（如 TPU、多个 GPU）作用域内创建并编译模型，才能正确在多设备上运行
    # GRU with glove embeddings and two dense layers 只不过把LSTM换成了GRU
     model = Sequential() #使用 Keras 的 Sequential API，按顺序堆叠各层，适合单输入单输出的线性模型
#第一层：词嵌入层 Embedding层
     model.add(Embedding(len(word_index) + 1, #词表大小（包括 0 填充索引）
                     300, #GloVe 预训练向量的维度
                     weights=[embedding_matrix], #初始化为事先构建的 (V+1,300) 矩阵
                     input_length=max_len, #每条序列长度，为1500
                     trainable=False))
     model.add(SpatialDropout1D(0.3))
    #类似普通的 Dropout，但在时间步维度上丢弃整个特征通道，即「同一通道」在所有时间步都丢弃或保留。0.3 表示以 30% 的概率将每个通道丢弃，有助防止过拟合，并对序列模型更有效。
#第二层：GRU 层
     model.add(GRU(300)) #设置隐藏单元数为 300；GRU 会在每个时间步上维护 300 维的隐藏状态只返回序列最后一步的隐藏状态，输出张量形状为 (batch_size, 300)
#第三层：Dense输出层
     model.add(Dense(1, activation='sigmoid'))#Dense(1)：一个神经元，输出一个标量。activation='sigmoid'：Sigmoid 激活函数，将输出映射到 [0,1]，作为“toxic”概率。
#编译模型
     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    #loss='binary_crossentropy'：二分类交叉熵损失函数。
    #optimizer='adam'：自适应学习率优化器。
    #metrics=['accuracy']：训练和评估时计算并显示准确率
model.summary()
#会展示每一层的名称、输出形状和参数量


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
#在 TensorFlow 的分布式策略（如 MirroredStrategy 或 TPUStrategy）下，模型副本会被“同构地”复制到多张 GPU / 多个 TPU 核心上。
#num_replicas_in_sync 返回正在并行训练的副本（replica）数目 # 64代表的是“每个副本”上的 局部 batch size
#在训练的过程中主要的流程：1.数据分批（划分为若干个全局batch，每个全局batch又在后台拆分到副本）2.前向传播（Forward Pass） 3.计算损失（Loss） 4.反向传播 5.梯度聚合与参数更新 6.指标记录


scores = model.predict(xvalid_pad) #对验证集输入做前向推断，得到每条样本属于正类（toxic）的概率
predicted_labels = (scores > 0.5).astype(int)  # 以 0.5 为阈值，把概率转换成 0 / 1 的离散标签。
accuracy = accuracy_score(yvalid, predicted_labels) #计算预测标签与真实标签之间的分类准确率
print(f"Accuracy: {accuracy:.4f}")  # 输出格式化为4位小数


scores_model.append({'Model': 'GRU','AUC_Score': accuracy})
#将当前 GRU 模型在验证集上的性能记录到 scores_model 列表中，字典包含两对键值：标明模型名称， 存储刚计算出的准确率


with strategy.scope(): #在指定的分布式策略（如多 GPU/TPU）下构建并编译模型，确保并行训练时各副本同步
    # A simple bidirectional LSTM with glove embeddings and one dense layer
    model = Sequential() #用 Keras 的线性堆叠容器，按添加顺序逐层构建网络
#词嵌入层
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix], #用预训练好的 embedding_matrix 初始化
                     input_shape=(max_len,),
                     trainable=False))
#双向LSTM
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))
   #dropout=0.3：输入层丢弃 30% recurrent_dropout=0.3：循环状态丢弃 30%
#输出层
    model.add(Dense(1,activation='sigmoid')) #单神经元 + Sigmoid，将前面提取的特征映射为 [0,1] 区间概率，用于二分类。
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    
model.summary()


#model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)#启动模型训练
#训练轮数为 5，即完整扫过训练集 5 次。（等同于新版本中的 epochs=5）
#全局 batch size = 64 × 副本数，确保每个设备各自处理 64 条样本，随后自动聚合梯度进行同步更新
####报错：典型的显存（或内存）不足导致的。双向 LSTM 模型太大、输入序列太长或 batch 太大，以至于一次前向/反向传播无法在可用设备上分配足够的内存必须逐步调小模型规模或 batch size，就能避免 ResourceExhaustedError ，并找到性能与资源使用的平衡点
model.fit(xtrain_pad, ytrain, epochs=5, batch_size=16*strategy.num_replicas_in_sync)#启动模型训练


scores = model.predict(xvalid_pad) #启动模型训练对验证集输入 xvalid_pad 做前向推断，得到每条样本为正类（toxic）的概率数组
predicted_labels = (scores > 0.5).astype(int)  # 以 0.5 为阈值，将概率转换成 0/1 的离散标签。
accuracy = accuracy_score(yvalid, predicted_labels) #用 sklearn 的 accuracy_score 计算预测标签与真实标签 yvalid 之间的准确率
print(f"Accuracy: {accuracy:.4f}")  # 输出格式化为4位小数


scores_model.append({'Model': 'Bi-directional LSTM','AUC_Score': accuracy})
#把当前模型的名称和准确率封装成字典，添加到 scores_model 列表中


scores_model


'''from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Attention
from keras.models import Model

# 1. 定义超参
EMBED_DIM   = 300   # GloVe 维度
ENC_UNITS   = 128   # Encoder 隐藏单元
DEC_UNITS   = 128   # Decoder 隐藏单元
MAX_LEN     = max_len  # 500 或你前面设定的长度
VOCAB_SIZE  = len(word_index) + 1

# 2. 构建输入层
encoder_inputs = Input(shape=(MAX_LEN,), name="encoder_inputs")   # (batch, timesteps)
decoder_inputs = Input(shape=(MAX_LEN,), name="decoder_inputs")   # 同长度，分类任务可全 0

# 3. 词嵌入层（共享）
embedding_layer = Embedding(
    input_dim   = VOCAB_SIZE,
    output_dim  = EMBED_DIM,
    weights     = [embedding_matrix],
    input_shape = (MAX_LEN,),
    trainable   = False,
    name        = "shared_embedding"
)

# 4. Encoder
enc_embed = embedding_layer(encoder_inputs)                        # (batch, MAX_LEN, EMBED_DIM)
encoder_lstm = LSTM(ENC_UNITS, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_embed)                     # 只保留最后时刻的状态

# 5. Decoder
# 5.1 先用 RepeatVector 把 context 向量广播到每个时间步
context = RepeatVector(MAX_LEN, name="repeat_vector")([state_h, state_c])
dec_embed = embedding_layer(decoder_inputs)                        # (batch, MAX_LEN, EMBED_DIM)

# 5.2 拼接 context 与 decoder embed
from keras.layers import Concatenate
dec_concat = Concatenate(name="encoder_decoder_concat")([context, dec_embed])

# 5.3 Decoder LSTM 返回全部时刻的输出
decoder_lstm = LSTM(DEC_UNITS, return_sequences=True, name="decoder_lstm")
dec_outputs = decoder_lstm(dec_concat)                            # (batch, MAX_LEN, DEC_UNITS)

# 6. Attention（可选）
# 6.1 计算 attention 权重
attention_layer = Attention(name="attention_layer")
attn_out = attention_layer([dec_outputs, dec_outputs])            # 自注意力示例

# 7. 时间分布的全连接 + 池化得到分类向量
td_dense = TimeDistributed(Dense(64, activation="relu"), name="td_dense")(attn_out)
from keras.layers import GlobalMaxPooling1D
pooled = GlobalMaxPooling1D(name="global_max_pool")(td_dense)     # (batch, 64)

# 8. 输出层
output = Dense(1, activation="sigmoid", name="classifier")(pooled)

# 9. 定义并编译模型
model_seq2seq = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output, name="seq2seq_classifier")
model_seq2seq.compile(
    loss      = "binary_crossentropy",
    optimizer = "adam",
    metrics   = ["accuracy"]
)

# 10. 查看结构
model_seq2seq.summary()

# 11. 训练
# 对 decoder_inputs，我们这里简单用全零输入（或可同 X_train_pad）
zero_decoder_input = np.zeros_like(X_train_pad)
history_seq2seq = model_seq2seq.fit(
    [X_train_pad, zero_decoder_input], y_train,
    epochs          = 5,
    batch_size      = 64,
    validation_data = ([X_valid_pad, np.zeros_like(X_valid_pad)], y_valid),
    callbacks       = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
)

# 12. 评估并记录
scores = model_seq2seq.predict([X_valid_pad, np.zeros_like(X_valid_pad)])
preds  = (scores > 0.5).astype(int)
acc = accuracy_score(y_valid, preds)
print(f"Seq2Seq 模型验证集准确率: {acc:.4f}")
scores_model.append({'Model': 'Seq2Seq', 'AUC_Score': acc})'''


# Visualization of Results obtained from various Deep learning models
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
#将之前收集的 scores_model（列表里每项是 {'Model':…, 'AUC_Score':…}）转成 Pandas DataFrame
#然后按 'AUC_Score' 从大到小排序，得到各模型性能的排行榜
results.style.background_gradient(cmap='Blues')
#给这个表格添加“蓝色渐变”背景，高分单元格颜色更深，直观突出性能更好的模型


import plotly.io as pio
# 在 Kaggle 中常用的渲染器
pio.renderers.default = "iframe_connected"
#用 Plotly 的 Funnelarea（漏斗图）将模型名称和对应分数映射到面积大小
fig = go.Figure(go.Funnelarea(
    text =results.Model,
    values = results.AUC_Score,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show(renderer="kaggle")      # 在 Kaggle Notebook 上
#text 表示每个扇区的标签（模型名），values 表示面积大小（AUC 得分）；title 设置图表标题并居中显示


#1️⃣ 加载依赖与工具包
import os #Python 标准库，用于文件路径、环境变量等操作
import tensorflow as tf #加载 TensorFlow，NLP 模型训练与数据管道构建的核心框架
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam #加载 Adam 优化器，用于参数更新
from tensorflow.keras.models import Model #Keras 函数式 API 的模型基类
from tensorflow.keras.callbacks import ModelCheckpoint #训练时保存最佳模型的回调
from kaggle_datasets import KaggleDatasets #Kaggle 提供的工具，用于获取竞赛数据集的 GCS 路径
import transformers #Hugging Face 的 Transformers 库，用于加载预训练 Transformer 模型和 Tokenizer
from tokenizers import BertWordPieceTokenizer #Hugging Face 的快速分词工具，基于 WordPiece 算法


# 2️⃣ 加载数据

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
#train1、valid、test、sub 分别对应训练集、验证集、测试集和提交模板


# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
#从 Hugging Face Hub 加载与 distilbert-base-multilingual-cased 模型一致的 Python Tokenizer，实现词到 ID 的映射
# Save the loaded tokenizer locally 将词表文件（vocab.txt）保存到当前工作目录，供快速分词器加载
tokenizer.save_pretrained('.') 
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer
#用 Hugging Face Tokenizers 库的 C++ 实现，加载同一份词表，性能更优；

lowercase=False 保持多语言大小写敏感


#3️⃣ 文本预处理：Tokenizer 编码函数
#定义批量编码函数
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    将输入的文本列表（texts）通过指定的 tokenizer 批量编码为固定长度的整数序列（Token IDs），
    并支持分块处理以优化内存使用。
    """
    #将文本列表批量分块（chunk_size），高效地转换为固定长度的 Token ID 矩阵 (N, maxlen)
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen) #保证每条序列长度固定，超长截断、补短填 0
    all_ids = []
    #一次性对一个文本块进行编码，返回多个 Encoding 对象，每个 .ids 属性即 ID 列表
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


#算例：
假设 texts = ["hi world", "bert model", "foo"]
chunk_size=2, maxlen=5
词表（简化）映射： {"hi":10,"world":11,"bert":20,"model":21,"foo":30}
tokenizer.encode_batch → 对每条文本生成 ids 列表，并做截断/填充：
文本	  原始 Token IDs 	截断/填充到长度 5 → enc.ids
"hi world"	[10,11] 	[0,0,10,11,0]
"bert model"[20,21]	    [0,0,20,21,0]
"foo"	     [30]	     [0,0,0,30,0] 

返回：一个 NumPy 数组，形状 (3,5)：
[[ 0,  0, 10, 11,  0],
 [ 0,  0, 20, 21,  0],
 [ 0,  0,  0, 30,  0]]


#4️⃣ 执行分词编码 + 标签准备
from tokenizers import BertWordPieceTokenizer
# 然后再执行编码
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values
#将训练、验证、测试文本分别转为 Token ID 数组 x_train, x_valid, x_test。y_trainy_valid 为对应的二分类标签（0/1）NumPy 数组


#5️⃣ 配置训练参数和 TF 数据管道
AUTO = tf.data.experimental.AUTOTUNE
#让 tf.data 在执行 dataset.prefetch() 或 dataset.map() 等操作时，自动并行化和调度输入数据的读取与预处理，以最大化 GPU/CPU 利用率、减少 I/O 瓶颈
# Configuration
EPOCHS = 3 #训练时将完整遍历训练集的次数设为 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync #在分布式训练下的全局 batch size。16 是每个设备（GPU/TPU 副本）上的本地 batch
MAX_LEN = 192 #Transformer 模型输入序列的最大长度


#训练集
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train)) #从 NumPy 数组创建 Dataset，每个元素是 (x_i, y_i) 或 x_i
    .repeat() #训练集无限重复，配合 steps_per_epoch 控制结束
    .shuffle(2048) #随机打乱 2048 条样本，提升泛化！
    .batch(BATCH_SIZE) #按全局 batch size 批量化，配合分布式策略
    .prefetch(AUTO)
)
#验证集
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache() #对验证集做缓存，加速多轮评估
    .prefetch(AUTO)
)
#测试集
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


#6️⃣ 构建 Transformer 模型
def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids") 
    ###输入 (input_word_ids)为 (batch_size, max_len) 的整数张量，表示批量文本的Token ID序列。

    sequence_output = transformer(input_word_ids)[0] ###对于大多数HuggingFace的Transformer模型（如BertModel、TFDistilBertModel），
    ###输出是一个元组：outputs[0]: 取每条序列第 0 个位置（[CLS]）的向量，作为序列表示，所有Token的上下文向量（形状 (batch_size, max_len, hidden_dim)）

    cls_token = sequence_output[:, 0, :] ### # 取每个样本的第0个Token（[CLS]）
    ###在BERT预训练中，[CLS] Token的向量被设计为聚合整个序列的信息。分类任务中，通常仅用 [CLS] 向量作为分类头的输入。
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    #使用学习率 1×10−5 的 Adam 微调预训练模型
    return model


#算例：
假设 max_len=5，batch size = 2
input_word_ids 张量示例：
[[101, 2003, 1037, 102,   0],   # [CLS] this a [SEP] [PAD]
 [101, 7592, 2088, 102,   0]]   # [CLS] hello world [SEP] [PAD]
transformer(input_word_ids)[0]
输出 sequence_output 形状 (2, 5, H)，其中 H 是隐藏维度（比如 768）。
例如：
sequence_output = tf.random.normal((2,5,768))
cls_token = sequence_output[:,0,:]
取第 0 位置 [CLS] 的向量，形状 (2,768)：
[[…768-dim…],  # 样本1的CLS向量
 […768-dim…]]  # 样本2的CLS向量
out = Dense(1, activation='sigmoid')(cls_token)
对 (2,768) 的输入做全连接 + Sigmoid，输出 (2,1) 的概率数组：
[[0.12],  # 样本1属于正类的概率
 [0.87]]  # 样本2的概率



#7️⃣ 加载预训练 Transformer 并构建模型
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel #在线下载并加载预训练权重
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()


n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps, #每轮训练需执行的批次数，使一次 epoch 刚好消费完整训练集
    validation_data=valid_dataset, #指定验证集进行每轮后评估
    epochs=EPOCHS
)


#8️⃣ 模型训练（train + fine-tune）
#额外在验证集上进一步 fine‑tune，通常用于领域自适应
n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)


#9️⃣ 预测并提交
sub['toxic'] = model.predict(test_dataset, verbose=1) #对测试集做分布式推断，得到概率向量
sub.to_csv('submission.csv', index=False) #将提交模板中 toxic 列填入预测结果并导出 CSV

