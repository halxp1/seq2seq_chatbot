import numpy as np
"""
维护一个字典，把一个list（或者字符串）编码化，或者反向恢复
"""
class WordSequence(object):
    PAD_TAG='<pad>'
    UNK_TAG='<unk>'
    START_TAG='<s>'
    END_TAG='</S>'

    PAD=0
    UNK=1
    START=2
    END=3
    word_dict={}
    def __init__(self,
                # word_vec_dic='sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', #百度百科中文词向量  https://github.com/Embedding/Chinese-Word-Vectors
                # embedding_dim=300
                ):
        #初始化字典
        self.word_dict={
            WordSequence.PAD_TAG:WordSequence.PAD,
            WordSequence.UNK_TAG:WordSequence.UNK,
            WordSequence.START_TAG:WordSequence.START,
            WordSequence.END_TAG:WordSequence.END
        }
        self.fited=False
        self.word_vec_dic=word_vec_dic
        # self.embedding_dim=embedding_dim
        
        
    def to_index(self,word):
        assert self.fited, 'WordSequence 尚未进行'
        if word in self.word_dict:
            return self.word_dict[word]
        return WordSequence.UNK

    def to_word(self,index):
        assert self.fited
        for k,v in self.word_dict.items():
            if v==index:
                return k
        return WordSequence.UNK_TAG
        
    def size(self):
        assert self.fited
        return len(self.word_dict)+1

    def __len__(self):
        return self.size()

    def fit(self,sentences,min_count=5,max_count=None,max_features=None):
        """
        Args:
        min_count 最小出现次数
        max_count 最大出现次数
        max_features 最大特征数
        """
        assert not self.fited , 'WordSequence 只能 fit 一次'

        count={}
        for sentence in sentences:
            arr=list(sentence)
            for a in arr:
                if a not in count:
                    count[a]=0
                count[a]+=1  
        
        print(count)

        if min_count is not None:
            count={k : v for k,v in count.items() if v >= min_count}  

        if max_count is not None:
            count={k : v for k,v in count.items() if v<=max_features}

        self.word_dict = {
            WordSequence.PAD_TAG:WordSequence.PAD,
            WordSequence.UNK_TAG:WordSequence.UNK,
            WordSequence.START_TAG:WordSequence.START,
            WordSequence.END_TAG:WordSequence.END

        }

        if isinstance(max_features,int):
            count = sorted(list(count.items()),key=lambda x:x[1]) #对value排序 升序 返回list元组
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w,_ in count:
                self.word_dict[w] = len(self.word_dict) #构建{word:index}
        else:
            for w in sorted(count.keys()):   #按照key排序，返回keylist
                self.word_dict[w]=len(self.word_dict)  
            
        self.fited=True

        #采用预训练好的部分词向量
        # embeddings_index={}
        # print("正在加载预训练词向量……")
        # with open(self.word_vec_dic, 'rb') as f:
        #     for line in f:
        #         values = line.decode('utf-8').split(' ')
        #         word = values[0]
        #         embedding=values[1:301]
        #         embeddings_index[word]=embedding
        # print("预训练词向量加载完毕。")
        # nb_words = len(self.word_dict)

        # self.word_embedding_matrix=np.zeros((nb_words,self.embedding_dim),dtype=np.float32)
        # for word,i in self.word_dict.items():
        #     if word in embeddings_index:
        #         self.word_embedding_matrix[i] = embeddings_index[word]
        #     else:
        #         new_embedding = np.array(np.random.uniform(-1,1,self.embedding_dim))
        #         embeddings_index[word] = new_embedding
        #         self.word_embedding_matrix[i] = embeddings_index[word]
        # print('词向量映射完成')        

    def showdict(self):
        assert self.fited

        for k,v in self.word_dict.items():
            print(k,v)


    def transform(self,sentence,max_len=None):
        assert self.fited

        if max_len is not None:
            r = [self.PAD]*max_len
        else:
            r=[self.PAD]*len(sentence)
        
        for index,a in enumerate(sentence):
            if max_len is not None and index >=len(r):
                break
            r[index]=self.to_index(a)

        return np.array(r)  #最后返回的是[3,4,6,5,end,pad,pad,pad]

    def inverse_transform(self,indices,ignore_pad=False,ignore_unk=False,ignore_start=False,ignore_end=False):
        ret=[]
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word==WordSequence.START_TAG and ignore_start:
                continue
            if word==WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret

def test():
    ws = WordSequence()
    ws.fit([
        ['你','好','啊'],
        ['你','好','哦'],
        ['我','是','谁']
    ])
    print(ws.word_embedding_matrix[0])
    print(ws.word_embedding_matrix[1])
 
    # indice =ws.transform(['你','们'])
    # print(indice)

    # back = ws.inverse_transform(indice)
    # print(back)


if __name__ == '__main__':
    test()                     

