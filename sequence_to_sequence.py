import numpy as np
import tensorflow as tf
from tensorflow import layers

from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell  #实现多层RNN
from tensorflow.contrib.rnn import DropoutWrapper  #drop网络
from tensorflow.contrib.rnn import ResidualWrapper #残差网络  就是把输入concat到输出上一起返回

from word_sequence import WordSequence
from data_utils import _get_embed_device

class SequenceToSequence(object):  
    """
    基本流程
    __init__ 基本参数保存，参数验证（验证参数的合法性）
    build_model 构建模型
    init_placeholders  初始化一些Tensorflow的变量占位符
    build_encoder 初始化编码器
        build_single_cell
        build_decoder_cell
    init_optimizer 如果是在训练模式下进行, 那么则需要初始化优化器
    train 训练一个batch 数据
    predict 预测一个batch数据
    """                              
    def __init__(self,               #
                input_vocab_size,    #输入词表的大小
                target_vocab_size,   #输出词表的大小
                batch_size=32,       #数据batch的大小
                embedding_size=300,  #输入词表与输出词表embedding的维度
                mode="train",        #取值为train, 代表训练模式, 取值为decide,代表预训练模式
                hidden_units=256,    #Rnn模型的中间层大小,encoder和decoder层相同
                depth=1,             #encoder和decoder的rnn层数
                beam_width=0,        #是beamsearch的超参数,用于解码
                cell_type="lstm",    #rnn的神经元类型, lstm, gru
                dropout=0.2,         #随机丢弃数据的比例,是要0到1之间
                use_dropout=False,   #是否使用dropout
                use_residual=False,  #是否使用residual
                optimizer='adam',    #使用哪一个优化器
                learning_rate=1e-3,  #学习率
                min_learning_rate=1e-5,  #最小学习率
                decay_steps=50000,   #衰减步数
                max_gradient_norm=5.0,  #梯度正则裁剪的系数
                max_decode_step=None,   #最大decode长度, 可以非常大
                attention_type='Bahdanau', #使用attention类型
                bidirectional=False,     #是否使用双向encoder
                time_major=False,       #是否在计算过程中使用时间作为主要的批量数据
                seed=0,               #一些层间的操作的随机数
                parallel_iterations=None,  #并行执行rnn循环的个数
                share_embedding=False,    #是否让encoder和decoder共用一个embedding
                pretrained_embedding=False):  #是不是要使用预训练的embedding
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode 
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 -dropout
        self.seed = seed
        self.pretrained_embedding =  pretrained_embedding
        self.bidirectional = bidirectional

        if isinstance(parallel_iterations, int):
            self.parallel_iterations= parallel_iterations
        else:
            self.parallel_iterations = batch_size
        self.time_major = time_major
        self.share_embedding = share_embedding
        #生成均匀分布的随机数  用于变量初始化
        self.initializer = tf.random_uniform_initializer(
            -0.05, 0.05, dtype=tf.float32
        )
        assert self.cell_type in ('gru', 'lstm'), 'cell_type 应该是GRU 或者是 LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size, '如果share_embedding 为True 那么两个vocab_size 必须一样'
        assert mode in ('train', 'decode'), 'mode 必须是train 或者是decode , 而不是{}'.format(mode)

        assert dropout >=0.0 and dropout< 1.0, 'dropout 必须大于等于0 且小于等于1'

        assert attention_type.lower() in ('bahdanau', 'loung'), 'attention_type 必须是bahdanau 或者是 loung'
           
        assert beam_width < target_vocab_size, 'beam_width {} 应该小于target_vocab_size{}'.format(beam_width,target_vocab_size)

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )
        self.global_step = tf.Variable(
            0, trainable = False, name = 'global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width 
        self.use_beamsearch_decode = True if self.beam_width > 0 else False
        self.max_decode_step = max_decode_step

        assert self.optimizer.lower() in ('adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'), \
            'optimizer 必须是下列之一: adadelta, adam, rmsprop, momentum, sgd '
        self.build_model()

    def build_model(self):
        """
        1. 初始化训练, 预测所需要的变量
        2. 构建编码器（encoder） build_encoder -> encoder_cell -> build_signal_cell
        3. 构建解码器（decoder）
        4. 构建优化器（optimizer）
        5. 保存
        """
        self.init_placeholders()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)
        
        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()

    def init_placeholders(self):
        """初始化训练，初始化所需要的变量 """
        self.add_loss = tf.placeholder(
            dtype=tf.float32,
            name='add_loss'
        )
        #编码器的输入
        # 编码器输入，shape=(batch_size, time_step)
        # 有 batch_size 句话，每句话是最大长度为 time_step 的 index 表示
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,None),
            name='encoder_inputs'
        )
        #编码器的长度输入
        # 编码器长度输入，shape=(batch_size, 1)
        # 指的是 batch_size 句话每句话的长度
        self.encoder_inputs_length = tf.placeholder(
            dtype = tf.int32,
            shape=(self.batch_size, ),
            name = 'encoder_inputs_length'
        )
        if self.mode =='train':

            #解码器的输入
            # 解码器输入，shape=(batch_size, time_step)
            # 注意，会默认里面已经在每句结尾包含 <EOS>
            self.decoder_inputs = tf.placeholder(
                dtype = tf.int32,
                shape=(self.batch_size, None),
                name = 'decoder_inputs'
            )
            #解码器输入的rewards 用于强化学习训练，shape=(batch_size, time_step)
            self.rewards = tf.placeholder(
                dtype = tf.float32,
                shape=(self.batch_size, 1),
                name='rewards'
            ) 
            
            #解码器的长度输入
            self.decoder_inputs_length = tf.placeholder(
                dtype = tf.int32,
                shape=(self.batch_size,),
                name ='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape=(self.batch_size, 1),
                dtype=tf.int32
            ) * WordSequence.START

            #实际训练时解码器的输入, start_token + decoder_inputs
            self.decoder_inputs_train = tf.concat([
                self.decoder_start_token,
                self.decoder_inputs
            ],axis=1)

    
    def build_signle_cell(self, n_hidden, use_residual):
        """
        构建一个单独的 RNNCell
        n_hidden : 隐藏层的神经元数量
        use_residiual : 是否使用residual wrapper
        """
        
        if self.cell_type == 'gru':
            cell_type = GRUCell  
        else:
            cell_type = LSTMCell
        
        cell = cell_type(n_hidden)
        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype = tf.float32,
                output_keep_prob = self.keep_prob_placeholder,
                seed = self.seed
            )

        if use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        """构建单独的编码器 """
        # 通过MultiRNNCells类来实现Deep RNN
        return MultiRNNCell([
            self.build_signle_cell(self.hidden_units, use_residual=self.use_residual) for _ in range(self.depth)
        ])

    def feed_embedding(self, sess, encoder=None, decoder=None):
        """
        加载预训练好embedding
        """
        assert self.pretrained_embedding, '必须开启pretrained_embedding才能使用feed_embedding'
        assert encoder is not None or decoder is not None, 'encoder 和 decoder 至少得输入一个！'

        if encoder is not None:
            sess.run(self.encoder_embeddings_init,
                    {self.encoder_embeddings_placeholder: encoder})
        
        if decoder is not None:
            sess.run(self.decoder_embeddings_init,
                    {self.decoder_embeddings_placeholder: decoder})


    def build_encoder(self):
        """ 构建编码器"""

        with tf.variable_scope('encoder'): #变量命名空间 ,实现变量共享
            encoder_cell = self.build_encoder_cell()

            with tf.device(_get_embed_device(self.input_vocab_size)):  #判断使用显存还是内存
                if self.pretrained_embedding:
                    self.encoder_embeddings = tf.Variable(
                        tf.constant(0.0,shape=(self.input_vocab_size, self.embedding_size)), trainable=True, name = 'embeddings'
                    )

                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size, self.embedding_size)
                    )
                    self.encoder_embeddings_init = self.encoder_embeddings.assign(  #赋值操作
                        self.encoder_embeddings_placeholder
                    )
                else:
                    self.encoder_embeddings = tf.get_variable(
                        name='embedding',
                        shape=(self.input_vocab_size, self.embedding_size),
                        initializer = self.initializer,
                        dtype = tf.float32
                    )

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(  #函数是在params中查找ids的表示 
            #这里是在二维embeddings中找二维的ids, ids每一行中的一个数对应embeddings中的一行，所以最后是[batch_size, time_step, embedding_size]
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )        
            if self.use_residual:
                #全连接层
                self.encoder_inputs_embedded = layers.dense(self.encoder_inputs_embedded,
                                                            self.hidden_units,
                                                            use_bias = False,
                                                            name='encoder_residual_projection')
            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs,(1,0,2))

            if not self.bidirectional:
                (encoder_outputs,encoder_state) = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs = inputs,
                    sequence_length = self.encoder_inputs_length,
                    dtype = tf.float32,
                    time_major = self.time_major,
                    parallel_iterations = self.parallel_iterations,
                    swap_memory=False
                )
            else:
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(  #动态多层双向lstm_rnn
                    cell_fw=encoder_cell,
                    cell_bw = encoder_cell_bw,
                    inputs = inputs,
                    sequence_length = self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory = True
                )
                encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)
        
                encoder_state = []
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)
            
            return encoder_outputs, encoder_state

    
    def build_decoder_cell(self,encoder_outputs, encoder_state):
        """ 构建解码器cell """
        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))
        
        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier = self.beam_width
            )
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width
            )
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width
            )
            #如果使用了beamsearch， 那么输入应该是beam_width的倍数等于batch_size的
            batch_size *= self.beam_width
        
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = LuongAttention(
                num_units = self.hidden_units,
                memory = encoder_outputs,
                memory_sequence_length = encoder_inputs_length
            ) 
        else:
            #BahdanauAttention 就是初始化时传入 num_units 以及 Encoder Outputs，然后调时传入 query 用即可得到权重变量 alignments。
            self.attention_mechanism = BahdanauAttention(
                num_units = self.hidden_units,
                memory = encoder_outputs,
                memory_sequence_length = encoder_inputs_length
            )
        
        cell = MultiRNNCell([ self.build_signle_cell(self.hidden_units, use_residual=self.use_residual) for _ in range(self.depth) ])
        # 在非训练（预测）模式，并且没开启 beamsearch 的时候，打开 attention 历史信息
        alignment_history = (
            self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs, attention):
            """ 根据attn_input_feeding属性来判断是否在attention计算前进行一次投影的计算"""
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)
            
            attn_projection = layers.Dense(self.hidden_units,
                                            dtype = tf.float32,
                                            use_bias=False,
                                            name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        attention_cell = AttentionWrapper(
            cell = cell,
            attention_mechanism = self.attention_mechanism,
            attention_layer_size= self.hidden_units,
            alignment_history = alignment_history,
            cell_input_fn = cell_input_fn,
            name = 'AttentionWrapper'
        )
        # 空状态
        decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32) 
        
        #传递encoder的状态  定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
        decoder_initial_state = decoder_initial_state.clone(
            cell_state = encoder_state
        )
        return attention_cell, decoder_initial_state
    
    def build_decoder(self, encoder_outputs, encoder_state):
        """ 
        构建解码器
        """
        with tf.variable_scope('decoder') as decoder_scope:
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_outputs, encoder_state)
            #构建解码器的embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                elif self.pretrained_embedding:

                    self.decoder_embeddings = tf.Variable(
                        tf.constant(0.0, shape=(self.target_vocab_size, self.embedding_size)
                        ),
                        trainable = True,
                        name = 'embeddings'
                    )
                    self.decoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.target_vocab_size, self.embedding_size)
                    )
                    self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name = 'embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer = self.initializer,
                        dtype = tf.float32
                    )
            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype = tf.float32,
                use_bias=False,
                name= 'decoder_output_projection'
            )

            if self.mode == 'train':
                self.decoder_inputs_embedded= tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids = self.decoder_inputs_train             
                )

                inputs = self.decoder_inputs_embedded
                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    #根据预测值或者真实值得到下一刻的输入
                    inputs = inputs,
                    sequence_length = self.decoder_inputs_length,
                    time_major = self.time_major,
                    name='training_helper'
                )
                # 训练的时候不在这里应用 output_layer
                # 因为这里会每个 time_step 的进行 output_layer 的投影计算，比较慢
                # 注意这个trick要成功必须设置 dynamic_decode 的 scope 参数
                training_decoder = seq2seq.BasicDecoder(
                    cell= self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state
                    #output_layer = self.decoder_output_projection    #输出映射层，将rnn_size转化为vocab_size维
                )
                #decoder在当前的batch下的最大time_steps
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)

                outputs, self.final_state, _ = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,   #Boolean，为真时会拷贝最后一个时刻的状态并将输出置零，程序运行更稳定，使最终状态和输出具有正确的值，在反向传播时忽略最后一个完成步。但是会降低程序运行速度。
                    maximum_iterations=max_decoder_length, #最大解码步数，一般训练设置为decoder_inputs_length，预测时设置一个想要的最大序列长度即可。程序会在产生<eos>或者到达最大步数处停止
                    parallel_iterations=self.parallel_iterations,  #parallel_iterations是并行执行循环的个数
                    swap_memory=True,
                    scope=decoder_scope
                )
                
                self.decoder_logits_train = self.decoder_output_projection(
                    outputs.rnn_output

                )
                self.masks = tf.sequence_mask(
                    #构建序列长度的mask标志
                    lengths = self.decoder_inputs_length,
                    maxlen = max_decoder_length,
                    dtype = tf.float32,
                    name='masks'
                )

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train, (1, 0, 2))
                
                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train,
                    axis = -1,
                    name= 'decoder_pred_train'
                )

                self.train_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = self.decoder_inputs,  #真实值y
                    logits = decoder_logits_train  #预测值y_
                )

                self.masks_rewards = self.masks * self.rewards

                self.loss_rewards = seq2seq.sequence_loss(
                    logits = decoder_logits_train,  #[batch_size, sequence_length, num_decoder_symbols]
                    targets = self.decoder_inputs,  #[batch_size, sequence_length]  不用做one_hot
                    weights = self.masks_rewards,   #[batch_size, sequence_length]  即mask，滤去padding的loss计算，使loss计算更准确。
                    average_across_timesteps=True,
                    average_across_batch=True
                )

                self.loss = seq2seq.sequence_loss(
                    #序列的损失函数
                    logits=decoder_logits_train,  #[batch_size, sequence_length, num_decoder_symbols]
                    targets = self.decoder_inputs, #[batch_size, sequence_length]  不用做one_hot
                    weights = self.masks,          # 即mask，滤去padding的loss计算，使loss计算更准确。
                    average_across_timesteps=True,
                    average_across_batch = True
                )

                self.loss_add = self.loss + self.add_loss

            elif self.mode == 'decode':
                start_tokens = tf.tile([WordSequence.START],[self.batch_size])
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    return tf.nn.embedding_lookup(self.decoder_embeddings, inputs)

                if not self.use_beamsearch_decode:
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens= start_tokens,
                        end_token=end_token,
                        embedding = embed_and_input_proj
                    )

                    inference_decoder = seq2seq.BasicDecoder(
                        cell = self.decoder_cell,
                        helper=decoding_helper,
                        initial_state = self.decoder_initial_state,
                        output_layer = self.decoder_output_projection
                    )
                else:
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens = start_tokens,
                        end_token = end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.decoder_output_projection
                    )

                if self.max_decode_step is not None:
                    max_decoder_step = self.max_decode_step
                else:
                    max_decoder_step = tf.round(tf.reduce_max(self.encoder_inputs_length) * 4)

                self.decoder_outputs_decode, self.final_state, _= seq2seq.dynamic_decode(
                    decoder = inference_decoder,
                    output_time_major=self.time_major,
                    maximum_iterations = max_decoder_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )             
                
                if not self.use_beamsearch_decode:
                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id
                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode,
                            (1, 0))
                else:
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids
                    
                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode,
                            (1, 0, 2)
                        )
                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1]
                    )
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores
                  
    def save(self, sess, save_path='model.ckpt'):
        """ 
        在tensorflow游两种保存模型:
        ckpt: 训练模型后保存， 这里会保存所有的训练参数, 文件相对来讲较大, 可以用来进行模型的恢复和加载
        pd: 用于模型最后的线上部署, 这里面的线上部署指的是Tensorflow Serving 进行模型发布, 一般发布成grpc形式的接口
        """
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='model.ckpt'):
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)


    def init_optimizer(self):
        """
        sgd, adadelta, adam, rmsprop, momentum
        """
        learning_rate = tf.train.polynomial_decay(
            #多项式衰减
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )

        self.current_learning_rate = learning_rate
        #返回需要训练的参数列表 trainalbe=True
        trainable_params = tf.trainable_variables()
        #设置优化器
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate= learning_rate
            )
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate = learning_rate, momentum=0.9
            )
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            )
        
        gradients = tf.gradients(ys=self.loss, xs=trainable_params) #函数列表ys里的每一个函数对xs中的每一个变量求偏导,返回一个梯度张量的列表
        
        #梯度裁剪 放置梯度爆炸
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm
        )
        #更新model
        self.updates = self.opt.apply_gradients(
            #进行BP算法
            #由于apply_gradients函数接收的是一个(梯度张量, 变量)tuple列表
            #所以要将梯度列表和变量列表进行捉对组合,用zip函数
            zip(clip_gradients, trainable_params),
            global_step = self.global_step
        )

        gradients = tf.gradients(self.loss_rewards, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm
        )
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step
        )

        #添加self.loss_add 的update
        gradients = tf.gradients(self.loss_add, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm
        )
        self.updates_add = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step = self.global_step
        )

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        encoder_inputs :一个整型的二维矩阵,[batch_size, max_source_time_steps]
        encoder_inputs_length: [batch_size], 每一个维度就是encoder句子的真实长度
        decoder_inputs: 一个整型的二维矩阵,[batch_size, max_target_time_steps]
        decoder_inputs_length: [batch_size],每一个维度就是decoder句子的真实长度
        decode: 是训练模式还是train(decode=false),还是预测模式decode(decoder=true)
        return: tensorflow所需要的input_feed，包括encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                'encoder_inputs  和  encoder_inputs_length的第一个维度必须一致'
                '这个维度是batch_size, %d != %d' % (
                    input_batch_size, encoder_inputs_length.shape[0]
                )
            )
        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    'encoder_inputs 和 decoder_inputs 的第一个维度必须一致'
                    '这个维度是batch_size, %d != %d' % (
                        input_batch_sezi, target_batch_size
                    )
                )
                
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    'encoder_inputs 和 decoder_inputs_length的第一个维度必须一致'
                    '这个维度是batch_size, %d != %d' %(
                        input_batch_size, target_batch_size.shape[0]
                    )
                )

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

    def train(self, sess, encoder_inputs, encoder_inputs_length,
        decoder_inputs, decoder_inputs_length,
        rewards=None, return_lr=False,
        loss_only=False, add_loss=None):
        """训练模型"""

        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )
        #设置dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if loss_only:
            #输出
            return sess.run(self.loss, input_feed)
        if add_loss is not None:
            input_feed[self.add_loss.name] = add_loss
            output_feed =[self.updates_add, self.loss_add, self.current_learning_rate]

            _, cost, lr = sess.run(output_feed, input_feed)
            if return_lr:
                return cost, lr
            return cost
        if rewards is not None:
            input_feed[self.rewards.name] = rewards
            output_feed =[self.updates_rewards, self.loss_rewards, self.current_learning_rate]

            _, cost, lr = sess.run(output_feed, input_feed)
            if return_lr:
                return cost, lr
            return cost
        
        output_feed = [self.updates, self.loss, self.current_learning_rate]

        _, cost, lr =sess.run(output_feed, input_feed)

        if return_lr:
            return cost, lr
        return cost

    def get_encoder_embedding(self, sess, encoder_inputs):
        input_feed ={
            self.encoder_inputs.name : encoder_inputs
        }
        emb = sess.run(self.encoder_inputs_embedded, input_feed)
        return emb
    
    def entropy(self, sess, encoder_inputs, encoder_inputs_length,
                decoder_inputs, decoder_inputs_length):
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, deocder_inputs_length,
            False
        )
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.train_entropy, self.decoder_pred_train]
        entropy, logits = sess.run(output_feed, input_feed) 
        return entropy, logits   

    def predict(self, sess,
        encoder_inputs,
        encoder_inputs_length,
        attention=False):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, None, None, True)

        input_feed[self.keep_prob_placeholder.name] =1.0

        if attention:
            assert not self.use_beamsearch_decode, 'Attention 模式不能打开BeamSearch'

            pred, atten = sess.run(
                [self.decoder_pred_decode, self.final_state.aligment_history.stack()], 
                input_feed)
            return pred, atten
            
        if self.use_beamsearch_decode:
            pred, beam_prob = sess.run(
                [self.decoder_pred_decode, self.beam_prob],
                input_feed)
            beam_prob = np.mean(beam_prob, axis=1)
            pred = pred[0]
            return pred
        
        pred, = sess.run([self.decoder_pred_decode], input_feed)
        return pred



    
        


    
        

                   


                








            


        


            



