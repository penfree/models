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
import jieba.posseg as pseg
from os.path import join

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


class TextParserEval(object):

    def __init__(self, task_context, arg_prefix, hidden_layer_sizes, model_dir,
                 model_path, in_corpus_name, out_corpus_name, batch_size,
                 max_steps, use_slim_model=True):
        self.model_dir = model_dir
        self.task_context, self.in_name = self.RewriteContext(
            task_context, in_corpus_name)
        self.arg_prefix = arg_prefix
        self.graph = tf.Graph()
        self.in_corpus_name = in_corpus_name
        self.out_corpus_name = out_corpus_name
        with self.graph.as_default():
            self.sess = tf.Session()
            feature_sizes, domain_sizes, embedding_dims, num_actions = self.sess.run(
                gen_parser_ops.feature_size(task_context=self.task_context,
                                            arg_prefix=self.arg_prefix))
        self.feature_sizes = feature_sizes
        self.domain_sizes = domain_sizes
        self.embedding_dims = embedding_dims
        self.num_actions = num_actions
        self.hidden_layer_sizes = map(int, hidden_layer_sizes.split(','))
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.use_slim_model = use_slim_model
        with self.graph.as_default():
            self.parser = graph_builder.GreedyParser(
                self.num_actions,
                self.feature_sizes,
                self.domain_sizes,
                self.embedding_dims,
                self.hidden_layer_sizes,
                gate_gradients=True,
                arg_prefix=self.arg_prefix)
            self.parser.AddEvaluation(self.task_context,
                                    self.batch_size,
                                    corpus_name=self.in_corpus_name,
                                    evaluation_max_steps=self.max_steps)
            self.parser.AddSaver(self.use_slim_model)
            self.sess.run(self.parser.inits.values())
            self.parser.saver.restore(self.sess, os.path.join(self.model_dir, model_path))

    def RewriteContext(self, task_context, in_corpus_name):
        context = task_spec_pb2.TaskSpec()
        with gfile.FastGFile(task_context, 'rb') as fin:
            text_format.Merge(fin.read(), context)
        tf_in = tempfile.NamedTemporaryFile(delete=False)
        for resource in context.input:
            for part in resource.part:
                if part.file_pattern != '-':
                    part.file_pattern = os.path.join(self.model_dir, part.file_pattern)
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
        with self.graph.as_default():
            with open(self.in_name, "w") as f:
                f.write(sentence)

            self.parser.AddEvaluation(self.task_context,
                                    self.batch_size,
                                    corpus_name=self.in_corpus_name,
                                    evaluation_max_steps=self.max_steps)
            # tf_documents = self.sess.run([self.parser.evaluation['documents'],])
            num_epochs = None
            num_docs = 0
            while True:
                tf_epochs, _, tf_documents = self.sess.run([self.parser.evaluation['epochs'],
                                                            self.parser.evaluation[
                    'eval_metrics'],
                    self.parser.evaluation['documents']])
                #print len(tf_documents)
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

class TextParser(object):
    '''
        Use syntaxnet model to parse text
    '''
    def __init__(self, model_dir, pos_param, parser_param, pos_hidden_layer,
                 parser_hidden_layer, pos_batch_size=32, pos_max_steps=1000,
                 parser_batch_size=8, parser_max_steps=1000, pos_use_slim=True,
                 parser_use_slim=True):
        self.pos_parser = TextParserEval(
            join(model_dir, 'models/brain_pos/greedy', pos_param, 'context'),
            'brain_pos',
            pos_hidden_layer,
            model_dir,
            join('models/brain_pos/greedy', pos_param, 'model'),
            'stdin', 'stdout-conll',
            pos_batch_size,
            pos_max_steps,
            pos_use_slim
        )
        self.parser = TextParserEval(
            join(model_dir, 'models/brain_parser/structured', parser_param, 'context'),
            'brain_parser',
            parser_hidden_layer,
            model_dir,
            join('models/brain_parser/structured', parser_param, 'model'),
            'stdin-conll', 'stdout-conll',
            parser_batch_size,
            parser_max_steps,
            parser_use_slim
        )

    def parse(self, text):
        if not isinstance(text, unicode):
            text = text.decode('utf-8')

        # cut words
        seg_list = [word for word, flag in pseg.cut(text)]
        target_text = u' '.join(seg_list).encode('utf-8')

        target_text = self.pos_parser.Parse(target_text)

        target_text = self.parser.Parse(target_text)
        return target_text


def main():
    print TextParser.__name__

if __name__ == '__main__':
    main()
