import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy

import testnet
from sklearn.metrics import *


logger=logging.getLogger(__name__)


class Graph:

	def accuracy_score(x1,y1,x2,y2,y_orig_1,y_orig_2):
		
		return [metrics.accuracy_score(y_orig_1,y1),metrics.accuracy_score(y_orig_2,y2)]


	def auc(x1,y1,x2,y2):
		
		return [metrics.auc(x1,y1),metrics.auc(x2,y2)]


	def average_percision_score(x1,y1,x2,y2):
		
		return [metrics.average_percision_score(y_orig_1,y1),metrics.average_percision_score(y_orig_2,y2)]

	def classificaiton_report(x1,y1,x2,y2):

		return [metrics.classificaiton_report(y_orig_1,y1),metrics.classificaiton_report(y_orig_2,y2)]


	def jaccordian_similarity(x1,y1,x2,y2):

		return [metrics.jaccard_similarity_score(y_orig_1,y1),metrics.metrics.jaccard_similarity_score(y_orig_2,y2)]

	def log_loss(x1,y1,x2,y2):
		return [metrics.log_loss(y_orig_1,y1),metrics.log_loss(y_orig_2,y2)]


