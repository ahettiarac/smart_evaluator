from tensorboard.plugins.hparams.summary_v2 import hparams
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from .AttentionLayer import AttentionLayer
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorboard.plugins import projector
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import os
import numpy as np
from keras import backend as K
K.clear_session()

class PredictorRNN:

    model = None
    input_size = 0
    output_size = 0
    encoder_input = None
    encoder_output = None
    state_h = None
    state_c = None
    decoder_embedding_layer = None
    decoder_lstm = None
    attn_layer = None
    decoder_dense = None
    decoder_input = None
    HP_NUM_UNITS_EMBED = hp.HParam('num_units',hp.Discrete([100,200]))
    HP_NUM_UNITS_LATENT = hp.HParam('num_units',hp.Discrete([300,400]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2,0.5))
    HP_OPTIMIZER = hp.HParam('optimizer',hp.Discrete(['adam','sgd','rmsprop','adadelta','adagrad']))
    METRIC_ACCURACY = 'accuracy'
    hparams = {
          HP_DROPOUT: 0.5,
          HP_OPTIMIZER: 'rmsprop'
    }
    vocab_size = 0
    x_vocab_size = 0

    def __init__(self,input_len,output_len,vocab_size,x_vocab_size):
        self.input_size = input_len
        self.output_size = output_len
        self.vocab_size = vocab_size
        self.x_vocab_size = x_vocab_size  

    def build_model(self):
        self.encoder_input = Input(shape=(self.input_size,))
        encoder_embedding = Embedding(self.x_vocab_size, 100, trainable=True)(self.encoder_input)
        encoder_lstm1 = LSTM(300, return_sequences=True, return_state=True, dropout=self.hparams[self.HP_DROPOUT])
        encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)
        encoder_lstm2 = LSTM(300, return_sequences=True, return_state=True, dropout=self.hparams[self.HP_DROPOUT])
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
        encoder_lstm3 = LSTM(300, return_sequences=True, return_state=True, dropout=self.hparams[self.HP_DROPOUT])
        self.encoder_output, self.state_h, self.state_c = encoder_lstm3(encoder_output2)

        self.decoder_input = Input(shape=(None,))
        self.decoder_embedding_layer = Embedding(self.vocab_size, 100, trainable=True)
        decoder_embedding = self.decoder_embedding_layer(self.decoder_input)
        self.decoder_lstm = LSTM(300, return_sequences=True, return_state=True, dropout=self.hparams[self.HP_DROPOUT])
        decoder_outputs, decoder_fwd_state, decoder_backward_state = self.decoder_lstm(decoder_embedding,initial_state=[self.state_h, self.state_c])

        self.attn_layer = AttentionLayer(name='attention_layer')
        attn_output, attn_state = self.attn_layer([self.encoder_output, decoder_outputs])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_output])

        self.decoder_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(decoder_concat_input)

        self.model = Model([self.encoder_input, self.decoder_input], decoder_output)
        self.model.compile(optimizer=self.hparams[self.HP_OPTIMIZER], loss='sparse_categorical_crossentropy')
        self.model.summary()

    def train_model(self, x_train, x_test, y_train, y_test,word_dict):
        for dropout_rate in (self.HP_DROPOUT.domain.min_value,self.HP_DROPOUT.domain.max_value):
            for optimizer in self.HP_OPTIMIZER.domain.values:
                self.hparams = {
                    self.HP_DROPOUT: dropout_rate,
                    self.HP_OPTIMIZER: optimizer 
                }
                self.build_model() 
                early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
                tensorboard = TensorBoard(log_dir='logs',histogram_freq=10,embeddings_freq=10)
                logs_dir = 'logs' 
                hpmatrix = hp.KerasCallback(logs_dir,self.hparams)
                history = self.model.fit([x_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:], epochs=50, 
                                            callbacks=[early_stopping,tensorboard,hpmatrix],batch_size=128,
                                            validation_data=([x_test, y_test[:, :-1]], y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))
                self.model.save_weights('model_weights.h5')   
                with open(os.path.join(logs_dir, 'metadata.csv'),"w") as f:
                    for word in word_dict:
                        f.write("{}\n".format(word))
                weights = tf.Variable(self.model.layers[1].get_weights()[0][1:])
                checkpoint = tf.train.Checkpoint(embedding=weights)
                checkpoint.save(os.path.join(logs_dir,"embedding.ckpt"))        
                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
                embedding.metadata_path = "metadata.csv"
                projector.visualize_embeddings(logs_dir, "embedding.ckpt")

    def get_inference_model(self,max_text_length,train=False):
        if not train:
            print("train not figured")
            self.model.load_weights('model_weights_rmsprop_0.5.h5')
        encoder_model = Model(inputs=self.encoder_input,outputs=[self.encoder_output,self.state_h,self.state_c])
        decoder_state_input_h = Input(shape=(300,))
        decoder_state_input_c = Input(shape=(300,))
        decoder_hidden_state_input = Input(shape=(max_text_length, 300))
        decoded_embed = self.decoder_embedding_layer(self.decoder_input)
        decoder_output2,state_h2,state_c2 = self.decoder_lstm(decoded_embed,initial_state=[decoder_state_input_h,decoder_state_input_c])
        attn_out_inf,attn_states_inf = self.attn_layer([decoder_hidden_state_input, decoder_output2])
        decoder_inf_concat2 = Concatenate(axis=-1,name='concat')([decoder_output2,attn_out_inf])
        decoder_output2 = self.decoder_dense(decoder_inf_concat2)
        decoder_model = Model([self.decoder_input] + [decoder_hidden_state_input,decoder_state_input_h,decoder_state_input_c],
                            [decoder_output2] + [state_h2,state_c2])
        return encoder_model,decoder_model

    def decode_sequence(self,input_seq,encoder_model,decoder_model,target_word_index,max_summary_length,reverse_target_word_index):
        en_out,en_h,en_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0,0] = target_word_index['sostock']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens,h,c = decoder_model.predict([target_seq]+[en_out,en_h,en_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]

            if sampled_token != 'eostock':
                decoded_sentence += ' '+sampled_token

            if (sampled_token == 'eostock' or len(decoded_sentence.split()) >= (max_summary_length-1)):
                stop_condition = True

            target_seq = np.zeros((1,1))
            target_seq[0,0] = sampled_token_index
            en_h,en_c = h,c
        return decoded_sentence                  




 

