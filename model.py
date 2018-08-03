from keras.models import Model
from keras.layers import Input, Multiply
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.engine.topology import Layer
from keras.activations import softmax

from keras import backend as K, initializers


def dot(a, b):
    return K.squeeze(K.dot(a, K.expand_dims(b)), axis=-1)

# https://www.kaggle.com/sermakarevich/hierarchical-attention-network
class AttentionLayer(Layer):
    """
    Attention layer. 
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.u_w = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.u_w = self.add_weight(shape=(input_dim, ),
                                   initializer='glorot_normal',
                                   name='u_w',
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, h, mask=None):
        u = TimeDistributed(Dense(200, activation='tanh'))(h)  # equation 5 and 8

        alpha = softmax(dot(u, self.u_w))  # equation 6 and 9

        if mask is not None:
            alpha *= K.cast(mask, K.floatx())
        alpha = K.expand_dims(alpha)

        s = K.sum(alpha * h, axis=1)  # equation 7 and 10

        return s

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# dropSentenceRnnOut = 0.5


def createHierarchicalAttentionModel(maxSeq, 
                                     embWeights=None, embeddingSize = None, vocabSize = None, #embedding
                                  recursiveClass = GRU, wordRnnSize=100, sentenceRnnSize=100,  #rnn 
                                  #wordDenseSize = 100, sentenceHiddenSize = 128, #dense
                                  dropWordEmb = 0.2, dropWordRnnOut = 0.2, dropSentenceRnnOut = 0.5):
    """
    Creates a model based on the Hierarchical Attention model according to : https://arxiv.org/abs/1606.02393
    inputs:
    maxSeq : max size for sentences
        embedding
            embWeights : numpy matrix with embedding values
            embeddingSize (if embWeights is None) : embedding size
            vocabSize (if embWeights is None) : vocabulary size
        Recursive Layers 
            recursiveClass : class for recursive class. Default is GRU
            wordRnnSize : RNN size for word sequence 
            sentenceRnnSize :  RNN size for sentence sequence
        Dense Layers
            wordDenseSize: dense layer at exit from RNN , on sentence at word level
            sentenceHiddenSize : dense layer at exit from RNN , on document at sentence level 
        Dropout
            
    returns : Two models. They are the same, but the second contains multiple outputs that can be use to analyse attention. 
    """

    # Sentence level logic

    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')
    if embWeights is None:
        emb = Embedding(vocabSize, embeddingSize, mask_zero=True)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], mask_zero=True, weights=[embWeights], trainable=False)(wordsInputs)
    
    if dropWordEmb != 0.0:
        emb = Dropout(dropWordEmb)(emb)
    word_rnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(emb)
    if dropWordRnnOut  > 0.0:
        word_rnn = Dropout(dropWordRnnOut)(word_rnn)
    #word_dense = TimeDistributed(Dense(200))(wordRnn)
    attention = AttentionLayer()(word_rnn)
    modelSentence = Model(wordsInputs, attention)
    
    
    documentInputs = Input(shape=(None,maxSeq), dtype='int32', name='document_input')
    #sentenceMasking = Masking(mask_value=0)(documentInputs)
    sentenceEmbbeding = TimeDistributed(modelSentence)(documentInputs)
    sentenceRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(sentenceEmbbeding)
    if dropSentenceRnnOut > 0.0:
        sentenceRnn = Dropout(dropSentenceRnnOut)(sentenceRnn)

    #sentence_dense = TimeDistributed(Dense(200))(sentenceRnn)
    attentionSent = AttentionLayer()(sentenceRnn)
    # documentEmb = merge([sentenceRnn, attentionSent], mode=lambda x:x[1]*x[0], output_shape=lambda x:x[0])
    #documentEmb = Multiply()([sentenceRnn, attentionSent])
    #documentEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]), name="att2")(documentEmb)
    documentOut = Dense(1, activation="sigmoid", name="documentOut")(attentionSent)
    
    
    model = Model(inputs=[documentInputs], outputs=[documentOut])
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    return model