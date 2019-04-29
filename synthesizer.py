import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
import re
import nltk
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    print ('***cleaner_names:', cleaner_names)
    print ('***text:', text)
    texts = tokenizer.tokenize(text)
    waves=[]

    for text in texts:
      seq = text_to_sequence(text, cleaner_names)
      print ('***seq:', seq)

      feed_dict = {
        self.model.inputs: [np.asarray(seq, dtype=np.int32)],
        self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
      }
      wav = self.session.run(self.wav_output, feed_dict=feed_dict)
      wav = audio.inv_preemphasis(wav)
      wav = wav[:audio.find_endpoint(wav)]
      waves.append(wav)
    wavestack=waves[0]
    for wave in waves[1:]:
      wavestack=np.hstack((wavestack,wave))  
    out = io.BytesIO()
    audio.save_wav(wavestack, out)
    return out.getvalue()
