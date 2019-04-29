import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer

with open("index.html", "r", encoding='utf8') as f:
  html_body = f.read()

class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(req.params.get('text'))
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  print(hparams_debug_string())
  synthesizer.load(args.checkpoint)
  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
  synthesizer.load(os.environ['CHECKPOINT'])
