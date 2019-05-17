from amaai.classify import logistic_regression
from amaai.openSmileFeatures import getOpenSmileFeatures
from amaai.visualise import visualise
import pandas as pd


# Demo/testing file, uncomment what you want to do.


# loading features from test (we'd need to run this command twice, one for training set, one for test set.
# features = getOpenSmileFeatures('emobase', '../testdata/')
# features.to_pickle('features')


features = pd.read_pickle('features')

visualise(features, 'tsne')
visualise(features, 'explore')


y = features['class']
x = features[:991]  # just removing the filename feature


# todo, second x and y should be from test set, first ones are for training
# logistic_regression(x, y, x, y)



