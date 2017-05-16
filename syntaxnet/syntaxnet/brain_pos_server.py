#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import time
import tempfile
import tensorflow as tf
import StringIO

import random
import string
import mmap

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet import sentence_pb2
from syntaxnet import graph_builder
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

import time
import BaseHTTPServer
import cgi

from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('task_context', '',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('resource_dir', '',
                    'Optional base directory for task context resources.')
flags.DEFINE_string('model_path', '', 'Path to model parameters.')
flags.DEFINE_string('arg_prefix', None, 'Prefix for context parameters.')
flags.DEFINE_string('graph_builder', 'greedy',
                    'Which graph builder to use, either greedy or structured.')
flags.DEFINE_string('input', 'stdin',
                    'Name of the context input to read data from.')
flags.DEFINE_string('output', 'stdout',
                    'Name of the context input to write data to.')
flags.DEFINE_string('hidden_layer_sizes', '200,200',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('beam_size', 8, 'Number of slots for beam parsing.')
flags.DEFINE_integer('max_steps', 1000, 'Max number of steps to take.')
flags.DEFINE_bool('slim_model', False,
                  'Whether to expect only averaged variables.')
                  
MODEL_DIR = '/home/ubuntu/models/syntaxnet/'
USE_SLIM_MODEL = True

TASK_CONTEXT = '/home/ubuntu/models/syntaxnet/models/brain_pos/greedy/128-0.008-10000-0.9-0/context'
TASK_INPUT = 'stdin'
TASK_OUTPUT = 'stdout-conll'

HIDDEN_LAYER = '256,256'
ARG_PREFIX = 'brain_pos'
MODEL_PATH = 'models/brain_pos/greedy/128-0.008-10000-0.9-0/model'
BATCH_SIZE = 32
MAX_STEPS = 1000

PORT_NUMBER = 5080

PAGE = u'''
<!DOCTYPE html>
<html>
<body>

<form action="/parse" method="POST">
请输入一个完整句子：<br>
<textarea rows="12" cols="100" type="text" name="sentence" >
</textarea>
<br/>
<input type="submit">
</form>
<br/>
转换结果：
<br/>
<pre>
{0}
</pre>
</body>
</html>
'''

def RewriteContext(task_context, in_corpus_name):
  context = task_spec_pb2.TaskSpec()
  with gfile.FastGFile(task_context, 'rb') as fin:
    text_format.Merge(fin.read(), context)
  tf_in = tempfile.NamedTemporaryFile(delete=False)
  for resource in context.input:
    for part in resource.part:
      if part.file_pattern != '-':
        part.file_pattern = os.path.join(MODEL_DIR, part.file_pattern)
    if resource.name == in_corpus_name:
      for part in resource.part:
        if part.file_pattern == '-':
          part.file_pattern = tf_in.name
  fout = tempfile.NamedTemporaryFile(delete=False)
  fout.write(str(context))
  return fout.name, tf_in.name


def UnderscoreIfEmpty(part):
  if not part:
    return unicode('_')
  return unicode(part)


def GetMorphAttributes(token):
  extension = (sentence_pb2.TokenMorphology.morphology)
  if not token.HasExtension(extension):
    return unicode('_')
  morph = token.Extensions[extension]
  if not morph:
    return unicode('_')
  if len(morph.attribute) == 0:
    return unicode('_')
  attrs = []
  for attribute in morph.attribute:
    value = attribute.name
    if attribute.value != 'on':
      value += unicode('=')
      value += attribute.value
    attrs.append(value)
  return unicode('|').join(attrs);

  
def ConvertTokenToString(index, token):
  fields = []
  fields.append(unicode(index + 1))
  fields.append(UnderscoreIfEmpty(token.word))
  fields.append(unicode('_'))
  fields.append(UnderscoreIfEmpty(token.category))
  fields.append(UnderscoreIfEmpty(token.tag))
  fields.append(GetMorphAttributes(token))
  fields.append(unicode(token.head + 1))
  fields.append(UnderscoreIfEmpty(token.label))
  fields.append(unicode('_'))
  fields.append(unicode('_'))
  return unicode('\t').join(fields)

  
def ConvertToString(sentence):
  value = unicode('')
  lines = []
  for index in range(len(sentence.token)):
    lines.append(ConvertTokenToString(index, sentence.token[index]))
  return unicode('\n').join(lines) + unicode('\n\n')


class ParserEval:
  def __init__(self, sess, task_context, arg_prefix, hidden_layer_sizes, model_path, in_corpus_name, out_corpus_name):
    self.task_context, self.in_name = RewriteContext(task_context, in_corpus_name)
    self.arg_prefix = arg_prefix
    self.sess = sess
    self.in_corpus_name = in_corpus_name
    self.out_corpus_name = out_corpus_name
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
        gen_parser_ops.feature_size(task_context=self.task_context,
                                    arg_prefix=self.arg_prefix))
    self.feature_sizes = feature_sizes
    self.domain_sizes = domain_sizes
    self.embedding_dims = embedding_dims
    self.num_actions = num_actions
    self.hidden_layer_sizes = map(int, hidden_layer_sizes.split(','))
    self.parser = graph_builder.GreedyParser(
        self.num_actions,
        self.feature_sizes,
        self.domain_sizes,
        self.embedding_dims,
        self.hidden_layer_sizes,
        gate_gradients=True,
        arg_prefix=self.arg_prefix)
    self.parser.AddEvaluation(self.task_context,
                              BATCH_SIZE,
                              corpus_name=self.in_corpus_name,
                              evaluation_max_steps=MAX_STEPS)
    self.parser.AddSaver(USE_SLIM_MODEL)
    self.sess.run(self.parser.inits.values())
    self.parser.saver.restore(self.sess, os.path.join(MODEL_DIR, model_path))
 

  def __del__(self):
    os.remove(self.task_context)
    # os.remove(self.in_name)
    # os.remove(self.out_name)


  def Parse(self, sentence):
    with open(self.in_name, "w") as f:
      f.write(sentence)

    self.parser.AddEvaluation(self.task_context,
                              BATCH_SIZE,
                              corpus_name=self.in_corpus_name,
                              evaluation_max_steps=MAX_STEPS)
    # tf_documents = self.sess.run([self.parser.evaluation['documents'],])
    num_epochs = None
    num_docs = 0;
    while True: 
      tf_epochs, _, tf_documents = self.sess.run([self.parser.evaluation['epochs'],
                                                  self.parser.evaluation['eval_metrics'],
                                                  self.parser.evaluation['documents']])
      print len(tf_documents)
      # assert len(tf_documents) == 1
      #print type(tf_documents[len(tf_documents)-1])
      if len(tf_documents) > 0:
        doc = sentence_pb2.Sentence()
        doc.ParseFromString(tf_documents[len(tf_documents)-1])
        #print unicode(doc)
        return ConvertToString(doc)
      if num_epochs is None:
        num_epochs = tf_epochs
      elif num_epochs < tf_epochs:
        break;


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
  def do_HEAD(s):
    s.send_response(200)
    s.send_header("Content-type", "text/html")
    s.end_headers()

  def do_GET(s):
    """Respond to a GET request."""
    s.send_response(200)
    s.send_header("Content-type", "text/html")
    s.end_headers()
    s.wfile.write(PAGE.format(' ').encode("utf-8"))
  def do_POST(s):
    form = cgi.FieldStorage(fp=s.rfile,
                            headers=s.headers,
                            environ={"REQUEST_METHOD": "POST"})
    target_text = ''
    for item in form.list:
      #print "begin: %s = %s" % (item.name, item.value)
      if item.name == 'sentence':
        target_text = item.value
    if target_text:
      target_text = s.parser.Parse(target_text)
    s.send_response(200)
    s.send_header("Content-type", "text/html")
    s.end_headers()
    #print target_text
    s.wfile.write(target_text.encode("utf-8"))


def main(unused_argv):
  sess = tf.Session()
  # parser = ParserEval(sess,
  #                    TOKENIZER_TASK_CONTEXT,
  #                    TOKENIZER_ARG_PREFIX,
  #                    TOKENIZER_HIDDEN_LAYER,
  #                    TOKENIZER_MODEL_PATH,
  #                    TOKENIZER_INPUT,
  #                    TOKENIZER_OUTPUT)
                     
  # with tf.Session() as morpher_sess:
  # parser = ParserEval(sess,
  #                     TASK_CONTEXT,
  #                     MORPHER_ARG_PREFIX,
  #                     MORPHER_HIDDEN_LAYER,
  #                     MORPHER_MODEL_PATH,
  #                     TASK_INPUT,
  #                     TASK_OUTPUT)

  # with tf.Session() as tagger_sess:
  parser = ParserEval(sess,
                      TASK_CONTEXT,
                      ARG_PREFIX,
                      HIDDEN_LAYER,
                      MODEL_PATH,
                      TASK_INPUT,
                      TASK_OUTPUT)
  # with tf.Session() as parser_sess:
  #   parser = ParserEval(parser_sess,
  #                       TASK_CONTEXT,
  #                       PARSER_ARG_PREFIX,
  #                       PARSER_HIDDEN_LAYER,
  #                       PARSER_MODEL_PATH,
  #                       TASK_INPUT,
  #                       TASK_OUTPUT)

  # result = tokenizer.Parse("俄罗斯最新一艘亚森级核动力潜艇喀山号31日在北德文斯克举行下水礼.")
  # result = morpher.Parse(result)
  # result = tagger.Parse(result)
  # result = parser.Parse(result)
  # print result
  server_class = BaseHTTPServer.HTTPServer
  MyHandler.parser = parser
  httpd = server_class(('', PORT_NUMBER), MyHandler)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  httpd.server_close()


if __name__ == '__main__':
  tf.app.run()

