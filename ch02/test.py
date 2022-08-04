# coding: utf-8
import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, create_co_matrix

# preprocess関数のテスト
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus, word_to_id, id_to_word)

# create_co_matrix関数のテスト
co_matrix = create_co_matrix(corpus, len(word_to_id))
print(co_matrix)
