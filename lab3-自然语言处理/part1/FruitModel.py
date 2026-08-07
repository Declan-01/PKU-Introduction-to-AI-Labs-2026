import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
import functools
import nltk
nltk.corpus.stopwords.words = functools.cache(nltk.corpus.stopwords.words)


class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        #两个字典，第一个用来统计负样本（直接 token_num[label][word]进行调用 ）
        
        self.V = 0 #语料库token数量；即词表大小
        self.tokens=set() #用来统计词表里面有什么单词，辅助V的计算
        
        self.pos_neg_num_word = [0, 0] 
        # 正负样本中所有出现单词的总次数 , 0负1正

        self.pos_neg_num=[0,0]
        #正负样本的总数量

        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        data_len=self.dataset.len
        for i in range(data_len):
            cleaned_words,label=self.dataset[i]
            self.pos_neg_num[label]+=1
            token_dict = self.token_num[label]

            for word in cleaned_words:
                token_dict[word]=token_dict.get(word,0)+1

        #通过对列表等进行处理获得V,self.pos_neg_num_word,tokens，否则超时
        pos_words=set(self.token_num[1].keys())
        neg_words=set(self.token_num[0].keys())

        self.tokens=pos_words|neg_words
        self.V=len(self.tokens)    

        self.pos_neg_num_word[1] = sum(self.token_num[1].values())
        self.pos_neg_num_word[0] = sum(self.token_num[0].values())
       
    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        
        alpha=1


        cleaned_words=text
        pos_prob_log=0.0
        neg_prob_log=0.0

        pos_den_log = math.log(self.pos_neg_num_word[1] + self.V * alpha)
        neg_den_log = math.log(self.pos_neg_num_word[0] + self.V * alpha)



        for word in cleaned_words:
                if word not in self.tokens:
                    continue

                pos_P_log=math.log(self.token_num[1].get(word,0)+alpha)-pos_den_log
                pos_prob_log+=(pos_P_log)

                neg_P_log=math.log(self.token_num[0].get(word,0)+alpha)-neg_den_log
                neg_prob_log+=(neg_P_log)
        
        pos_prob_log+=math.log(self.pos_neg_num[1]/(self.pos_neg_num[0]+self.pos_neg_num[1]))
        neg_prob_log+=math.log(self.pos_neg_num[0]/(self.pos_neg_num[0]+self.pos_neg_num[1]))

        if(pos_prob_log>=neg_prob_log):
            return 1
        else:
            return 0
        
        raise NotImplementedError


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        i = 0
        words=set(self.emb.keys())
        X = np.zeros((max_len, 100))

        for word in text:
            if word in words:
                X[i] = self.emb[word]
                i += 1

            if i == max_len:
                break

        return X


        raise NotImplementedError


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()
    '''
def get_document(root='./qadata'):
    cleaned_tokens = []
    document_list = os.listdir(root)
    document_list.sort()
    all_documents = []
    for path in document_list:
        path = os.path.join(root, path)
        
        with open(path, 'r', encoding='utf-8') as file:
            document = file.read()

        # tokenize document
        cleaned_tokens = tokenize(document)
        now_document = {}
        now_document['document'] = cleaned_tokens

        # tokenize sentences
        sentences = []
        for passage in document.split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                sentences.append([tokens, sentence])

        now_document['sentences'] = sentences
        all_documents.append(now_document)
    return all_documents

all_documents[i]:now_document
now_document['document']=cleaned_tokens :获得所有不去重的有效词
now_document['sentences'][i]=[tokens(in sentence) ,sentence ]

而self.document_list = get_document()
    '''
    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        #统一document就是cleaned_tokens，反正也是自己调用
        # tokenize document
        cleaned_tokens = document
        
        freq=cleaned_tokens.count(word)/len(cleaned_tokens)
        return math.log(freq+1)

        raise NotImplementedError  

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别

        D=len(self.document_list)
        d=0

        for document_handled in self.document_list:
            words=set(document_handled['document'])
            if word in words:
                d+=1

        idf=math.log(D/(1+d)) 

        return idf

        raise NotImplementedError
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        
        tf_num=self.tf(word,document)
        idf_num=self.idf(word)

        return tf_num*idf_num

        raise NotImplementedError

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        
        #用now_document数据结构来记录max_document
        max_document=self.document_list[0]
        max_tfidf=float("-inf")
        for document_handled in self.document_list:
            tfidf_now=0
            document_now=document_handled['document']
            for word in query:
                tfidf_word=self.tfidf(word,document_now)
                tfidf_now+=tfidf_word
            if tfidf_now>=max_tfidf:
                max_document=document_handled
                max_tfidf = tfidf_now


        #在max_document的sentences中利用idf进行筛选
        '''
        Sentences should be ranked according to “matching word measure”:
        namely, the sum of IDF values for any word in the query that also appears in the sentence. 
        Note that term frequency should not be taken into account here, only inverse document frequency.
        If two sentences have the same value according to the matching word measure, 
        then sentences with a higher “query term density” should be preferred. 
        Query term density is defined as the proportion of words in the sentence that are also words in the query. 
        For example, if a sentence has 10 words, 3 of which are in the query, then the sentence’s query term density is 0.3.
        '''
        sentences=max_document['sentences']
        max_sentence=sentences[0][1]
        max_idf_sum=float("-inf")
        qtd_max=0

        #先计算在这个document中，query中包含词的idf
        D=len(sentences)
        word2idf={word:1 for word in query}

        for word in query:
            if word2idf[word]!=1:
                continue
            
            d=0
            for sentence_handled in sentences:
                tokens=set(sentence_handled[0])
                if word in tokens:
                    d+=1

            word2idf[word]=math.log(D/(1+d))



        for sentence_handled  in sentences:
            tokens=sentence_handled[0]
            sentence=sentence_handled[1]
            idf_sum=0
            qtd_now=0
            len_sentence=len(tokens)

            for word in query:
                if word in tokens:
                    idf_sum+=word2idf[word]
                    qtd_now+=1

            if idf_sum>max_idf_sum:
                max_idf_sum=idf_sum
                max_sentence=sentence
                qtd_now/=len_sentence
                qtd_max=qtd_now
            elif idf_sum==max_idf_sum:
                qtd_now/=len_sentence
                if qtd_now>qtd_max:
                    max_sentence=sentence
                    qtd_max=qtd_now
                    max_idf_sum=idf_sum
        return max_sentence




        raise NotImplementedError

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 4e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-5  # L2正则化
    batchsize = 256
    max_epoch = 15
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)