from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import wmt
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils
from collections import defaultdict
import os
import tarfile



from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf
_QA_DATASETS_FROM_RESPONSE_TO_QUESTION = [
    [
        "",
        ("toby.response-question.response",
         "toby.response-question.question")
    ]
]

_QA_DATASETS_FROM_QUESTION_TO_RESPONSE = [
    [
        "",
        ("toby.response-question.question",
         "toby.response-question.response")
    ]
]

@registry.register_hparams
def transformer_toby():
  """ Toby's customized HParams"""
  hparams = transformer.transformer_base()
  hparams.learning_rate_cosine_cycle_steps = 100000
  return hparams

@registry.register_problem("predicting_questions_from_answers_twitter_entitiesAtEnd_tokens_16k")
class AtoQWithEntities16k(wmt.WMTProblem):
    @property
    def targeted_vocab_size(self):
        return 2 ** 14  # 16384
    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def vocab_name(self):
        return "vocab_QnA"

    def dataset(self):
        return _QA_DATASETS_FROM_RESPONSE_TO_QUESTION

    def train_generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, self.dataset())

        datasets = self.dataset()
        tag = "train" if train else "dev"

        #TODO: need to put training and dev data into the below folders
        #data_path = os.path.join(tmp_dir, "AtoQWithEntities16k_%s" % tag)
        data_path = wmt._compile_data(tmp_dir, datasets, "AtoQWithEntities16k_%s" % tag)
        return wmt.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, wmt.EOS)


@registry.register_problem("predicting_answers_from_questions_twitter_entitiesAtEnd_tokens_16k")
class QtoAWithEntities16k(AtoQWithEntities16k):
    def dataset(self):
        return _QA_DATASETS_FROM_QUESTION_TO_RESPONSE



def get_or_generate_vocab(data_dir, tmp_dir,
                          vocab_filename, vocab_size, sources):
  """Generate a vocabulary from the datasets in sources (_DATA_FILE_URLS)."""
  vocab_filepath = os.path.join(data_dir, vocab_filename)
  if tf.gfile.Exists(vocab_filepath):
    tf.logging.info("Found vocab file: %s", vocab_filepath)
    vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
    return vocab
  sources = sources
  tf.logging.info("Generating vocab from: %s", str(sources))
  token_counts = defaultdict(int)
  for source in sources:
    for lang_file in source[1]:
      tf.logging.info("Reading file: %s" % lang_file)
      filepath = os.path.join(tmp_dir, lang_file)

      # For some datasets a second extraction is necessary.
      if ".gz" in lang_file:
        new_filepath = os.path.join(tmp_dir, lang_file[:-3])
        if tf.gfile.Exists(new_filepath):
          tf.logging.info(
              "Subdirectory %s already exists, skipping unpacking" % filepath)
        else:
          tf.logging.info("Unpacking subdirectory %s" % filepath)
          generator_utils.gunzip_file(filepath, new_filepath)
        filepath = new_filepath

      # Use Tokenizer to count the word occurrences.
      with tf.gfile.GFile(filepath, mode="r") as source_file:
        file_byte_budget = 3.5e5 if "en" in filepath else 7e5
        for line in source_file:
          if file_byte_budget <= 0:
            break
          line = line.strip()
          file_byte_budget -= len(line)
          for tok in tokenizer.encode(text_encoder.native_to_unicode(line)):
            token_counts[tok] += 1

  vocab = text_encoder.SubwordTextEncoder.build_to_target_size(
      vocab_size, token_counts, 1, 1e3)
  vocab.store_to_file(vocab_filepath)
  return vocab