from custom.layers import *
from custom.callback import *
import numpy as np
import params as par
import utils
import sys
from tensorflow.python import keras
import re
import json
tf.executing_eagerly()


class MusicTransformer(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+3, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dist = dist
        


        if loader_path is not None:
            loader_path = tf.train.latest_checkpoint(loader_path)
            config_path = loader_path.replace('ckpt','config')
            print('config model from: {}'.format(config_path))
            self.load_config_file(config_path)

        self.Encoder = Encoder(
            d_model=self.embedding_dim, input_vocab_size=self.vocab_size,
            num_layers=self.num_layer, rate=dropout, max_len=max_seq)
        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = keras.layers.Dense(vocab_size, activation=None, name='output')

        self._set_metrics()

        if loader_path is not None:
            print('load model from: {}'.format(loader_path))
            self.load_ckpt_file(loader_path)

    def call(self, inputs, targets, training=None, eval=None, src_mask=None, trg_mask=None, lookup_mask=None):
        encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder = self.Decoder(targets, enc_output=encoder, training=training, lookup_mask=lookup_mask, mask=trg_mask)
#         print('encoder_inputs', inputs)
#         print('decoder_input', targets)
#         print('deocder_output', decoder.shape, tf.argmax(decoder[-1][-1]))
        fc = self.fc(decoder)
#         print('fc',fc.shape, max(fc[-1][-1]), tf.argmax(fc[-1][-1]))
        if training or eval:
            return fc
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if self._debug:
            tf.print('sanity:\n',self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        x, dec_input, target = MusicTransformer.__prepare_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, dec_input)
#         print('train enc_mask', enc_mask)
#         print('train tar_mask', tar_mask)
#         print('train look_ahead_mask', look_ahead_mask)

        if self.dist:
            predictions = self.__dist_train_step(
                x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)
        else:
            predictions = self.__train_step(x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True)

        if self._debug:
            print('train step finished')
        result_metric = []

        if self.dist:
            loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, self.loss_value, None)
        else:
            loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(target, predictions).numpy())

        return [loss.numpy()]+result_metric

    # @tf.function
    def __dist_train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        return self._distribution_strategy.experimental_run_v2(
            self.__train_step, args=(inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training))

    # @tf.function
    def __train_step(self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inp, targets=inp_tar, src_mask=enc_mask, trg_mask=tar_mask, lookup_mask=lookup_mask, training=training
            )
#             print('predictions:',predictions.shape, tf.argmax(predictions,-1))
#             print('out_tar:', out_tar)
            self.loss_value = self.loss(out_tar, predictions)
#             print('self.loss_value',self.loss_value)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False):

        x, inp_tar, out_tar = MusicTransformer.__prepare_data(x, y)
        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        
#         print('evaluate enc_mask', enc_mask)
#         print('evaluate tar_mask', tar_mask)
#         print('evaluate look_ahead_mask', look_ahead_mask)
        
        predictions = self.call(
                x,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(out_tar, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(out_tar, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric

    def save(self, filepath, epochs, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config--{}.json'.format(epochs)
        ckpt_path = filepath+'/ckpt--{}'.format(epochs)

        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")

    def sanity_check(self, x, y, mode='v'):
        # mode: v -> vector, d -> dict
        x, inp_tar, out_tar = MusicTransformer.__prepare_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, inp_tar)
        predictions = self.call(
            x,
            targets=inp_tar,
            src_mask=enc_mask,
            trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)

        if mode == 'v':
            return predictions
        elif mode == 'd':
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            return tf.argmax(predictions, -1)

    def get_config(self):
        config = {}
        config['debug'] = self._debug
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        config['dist'] = self.dist
        return config

    def kk2k(self, kks, k):
    	# kks 是一个k^2长度的list
    	# ks 是一个k长度的list
        ks = []
        for seq in kks:
            if seq[-1] != seq[-2]:
                ks.append(seq)
        ks.shuffle()
        if len(ks) < k:
            return kks[:k]
        else:
            ks = ks[:k]
        return ks
        
    def generate(self, prior: list, beam=None, length=2048):
        prior = tf.constant([prior])

        decode_array = [par.token_sos]
        decode_array = tf.constant([decode_array])


        # TODO: add beam search
        if beam is not None:
            
            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
#                 enc_mask, tar_mask, look_ahead_mask = \
#                     utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)
                
#                 print('genaration enc_mask', enc_mask)
#                 print('genaration tar_mask', tar_mask)
#                 print('genaration look_ahead_mask', look_ahead_mask)
                

                result = self.call(prior, targets=decode_array, src_mask=None,
                                    trg_mask=None, lookup_mask=None, training=False)
                
                result = tf.nn.top_k(result, beam).indices
                result = tf.cast(result, tf.int32)
#                 print('top3',result.shape, result)
                # top3 shape (1, i, 3)
#                 if result[0][-1][0] != decode_array[0][-1]:
#                     result = result[:,-1,0]
#                 else:
#                     result = result[:,-1,1]
                    #result shape (1)
                result = tf.random.shuffle(result)
                result = result[:,-1]
                decode_array = tf.concat([decode_array, tf.expand_dims(result[:,-1], 0)], -1)
#                 print('decode_array:',decode_array)
                #decode_array shape (1,i)
                
#                 del enc_mask
#                 del tar_mask
#                 del look_ahead_mask
            print('generated:',decode_array)
            return decode_array.numpy()

        else:
            for i in range(min(self.max_seq, length)):
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
#                 enc_mask, tar_mask, look_ahead_mask = \
#                     utils.get_masked_with_pad_tensor(decode_array.shape[1], prior, decode_array)
                enc_mask, tar_mask, look_ahead_mask = None, None, None

                result = self.call(prior, targets=decode_array, src_mask=enc_mask,
                                    trg_mask=tar_mask, lookup_mask=look_ahead_mask, training=False)
                
                result = tf.argmax(result, -1)
                result = tf.cast(result, tf.int32)
#                 print('generated',result.shape, result[:, -1].shape)
                decode_array = tf.concat([decode_array, tf.expand_dims(result[:, -1], 0)], -1)
#                 print('decode_array:',decode_array)
                
#                 del enc_mask
#                 del tar_mask
#                 del look_ahead_mask
            return decode_array.numpy()

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']
        self.dist = config['dist']

    @staticmethod
    def __prepare_data(x, y):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos
        # end_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_
        

        # # method with eos
        # out_tar = tf.concat([y[:, :-1], end_token], -1)
        # inp_tar = tf.concat([start_token, y[:, :-1]], -1)
        # x = tf.concat([start_token, x[:, 2:], end_token], -1)

        # method without eos
        out_tar = y
        inp_tar = y[:, :-1]
        inp_tar = tf.concat([start_token, inp_tar], -1)
        return x, inp_tar, out_tar

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return


if __name__ == '__main__':
    import utils
    print(tf.executing_eagerly())

    src = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    trg = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    src_mask, trg_mask, lookup_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
    print(lookup_mask)
    # print(src_mask, trg_mask)
    mt0 = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size)
    res0 = mt0(src,trg)
    print(res0)
#     mt1 = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size)
#     res1 = mt1(src,trg)
#     print(res1)
#     mt2 = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size, loader_path='result')
#     res2 = mt2(src,trg)
#     print(res2)
#     mt3 = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size, loader_path='result')
#     res3 = mt3(src,trg)
#     print(res3)
    
    mt.save_weights('my_model.h5', save_format='h5')
    mt.load_weights('my_model.h5')
    # print('compile...')
    # mt.compile(optimizer='adam', loss=callback.TransformerLoss(debug=True))
    # # print(mt.train_step(inp=src, tar=trg))
    #
    # print('start training...')
    # for i in range(2):
    #     mt.train_on_batch(x=src, y=trg)
    result = mt.generate([27, 186,  43, 213, 115, 131], length=100)
    print(result)
    import sequence
    sequence.EventSeq.from_array(result[0]).to_note_seq().to_midi_file('result.midi')
    pass