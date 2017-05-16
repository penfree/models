#!/usr/bin/env python
# coding=utf-8
'''
Author: qiupengfei@iyoudoctor.com

'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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
    return unicode('|').join(attrs)


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


class PosParserEval(object):

    def __init__(self, task_context, arg_prefix, hidden_layer_sizes, model_path, in_corpus_name, out_corpus_name):
        self.task_context, self.in_name = RewriteContext(
            task_context, in_corpus_name)
        self.arg_prefix = arg_prefix
        self.sess = tf.Session()
        self.in_corpus_name = in_corpus_name
        self.out_corpus_name = out_corpus_name
        feature_sizes, domain_sizes, embedding_dims, num_actions = self.sess.run(
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

    def RewriteContext(self, task_context, in_corpus_name):
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
        num_docs = 0
        while True:
            tf_epochs, _, tf_documents = self.sess.run([self.parser.evaluation['epochs'],
                                                        self.parser.evaluation[
                'eval_metrics'],
                self.parser.evaluation['documents']])
            print len(tf_documents)
            # assert len(tf_documents) == 1
            # print type(tf_documents[len(tf_documents)-1])
            if len(tf_documents) > 0:
                doc = sentence_pb2.Sentence()
                doc.ParseFromString(tf_documents[len(tf_documents) - 1])
                # print unicode(doc)
                return ConvertToString(doc)
            if num_epochs is None:
                num_epochs = tf_epochs
            elif num_epochs < tf_epochs:
                break


class ParserParserEval(object):

    def __init__(self, task_context, arg_prefix, hidden_layer_sizes, model_path, in_corpus_name, out_corpus_name):
        self.task_context, self.in_name = RewriteContext(task_context, in_corpus_name)
        self.arg_prefix = arg_prefix
        self.sess = tf.Session()
        self.in_corpus_name = in_corpus_name
        self.out_corpus_name = out_corpus_name
        feature_sizes, domain_sizes, embedding_dims, num_actions = self.sess.run(
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
        num_docs = 0
        while True:
            tf_epochs, _, tf_documents = self.sess.run([self.parser.evaluation['epochs'],
                                                        self.parser.evaluation['eval_metrics'],
                                                        self.parser.evaluation['documents']])
            print len(tf_documents)
            # assert len(tf_documents) == 1
            # print type(tf_documents[len(tf_documents)-1])
            if len(tf_documents) > 0:
                doc = sentence_pb2.Sentence()
                doc.ParseFromString(tf_documents[len(tf_documents) - 1])
                # print unicode(doc)
                return ConvertToString(doc)
            if num_epochs is None:
                num_epochs = tf_epochs
            elif num_epochs < tf_epochs:
                break

def main(unused_args):
    print ParserParserEval.__name__
    print PosParserEval.__name__

if __name__ == '__main__':
    tf.app.run()