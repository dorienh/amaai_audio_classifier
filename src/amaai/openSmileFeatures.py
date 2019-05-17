# import extractOpenSmile
import os
import shutil

import pandas as pd
from arff2pandas import a2p
import subprocess

from amaai.extractOpenSmile import extractFeatures





def getDataFrameFromCSV(filepath, sep, fileheader):
    # for each config:
    files = os.listdir(filepath)
    # print(files)
    for thisfile in files:
        contents = pd.read_csv(filepath + thisfile, sep = sep, header=None, names=fileheader)
    return contents

def getDataFrameFromCSVwithHeader(filepath, sep):
    # for each config:
    files = os.listdir(filepath)
    # print(files)
    for thisfile in files:
        contents = pd.read_csv(filepath + thisfile)
    return contents



def fixEmoArff(fname):
    subprocess.call(["sed -i '' '993d' "+fname], shell=True)
    subprocess.call(["sed -i '' 's/,unassigned//g' "+fname], shell=True)



def getDataFrameFromARFF(filepath):
    # for each config:
    files = os.listdir(filepath)
    allFeatures = []
    # print(files)
    for thisfile in files:
        print(thisfile)
        fixEmoArff(filepath+thisfile)
        with open(filepath + thisfile) as f:
            contents = a2p.load(f)
            # add a column for filename
            contents['filename'] = thisfile
            allFeatures.append(contents)
        # print(contents)

    return allFeatures


def consolidate(featuresEmobaseDraft):
    jointdf = pd.concat(featuresEmobaseDraft)
    return jointdf




def checkPathExists(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        for file in os.listdir(filepath):
            shutil.rmtree(filepath+file)
        # os.rmdir(filepath)
        print('Careful, '+ filepath + ' directory wasn\'t empty!')
        # os.makedirs(filepath)




def getChromaFeatures(datadir, workingdir, label):
    """
    Get chroma features, note, these are not yet consolidated per audio file, and stored in an csv file. TODO consolidation (average max, min etc. needs to be implemented)
    :param datadir: folder with audio files
    :param workingdir: where can we put the opensmile extracted feature files (should ideally be empty)
    :param label: foldername will stored as the class label
    :return:
    """
    checkPathExists(workingdir)
    checkPathExists(workingdir + 'chroma_fft')
    extractFeatures(datadir, workingdir + "chroma_fft/", "chroma_fft.conf")
    # load Chroma features
    header = ["chroma1", "chroma2", "chroma3", "chroma4", "chroma5", "chroma6", "chroma7", "chroma8", "chroma9", "chroma10", "chroma11", "chroma12"]
    featuresChroma = getDataFrameFromCSV(workingdir + 'chroma_fft', ";",header)
    # todo needs consolidating
    featuresChroma['class'] = label
    return featuresChroma




def getEmobaseFeatures(datadir, workingdir, label):
    """
    Get emobase features, note, these are already consolidated per audio file, and stored in an arff file.
    :param datadir: folder with audio files
    :param workingdir: where can we put the opensmile extracted feature files (should ideally be empty)
    :param label: foldername will stored as the class label
    :return:
    """
    checkPathExists(workingdir)
    checkPathExists(workingdir + 'emobase')
    extractFeatures(datadir, workingdir + "emobase/", "emobase.conf")
    # load emobase features
    featuresEmobaseDraft = getDataFrameFromARFF('../OpenSmileFeatures/emobase/')
    featuresEmobase = consolidate(featuresEmobaseDraft)
    featuresEmobase['class'] = label
    # fix issue of not having index.
    # featuresTest = featuresTest.reset_index(drop=True)

    return featuresEmobase






def getOpenSmileFeatures(type, dir):
    """
    :param type: emobase, chroma, or others to indicate which opensmile config file to us
    :param dir: dir where the input data is located
    :return: pandas dataframe with features and label (directory name)
    """
    if (type == 'emobase'):
        bufferdir = "../OpenSmileFeatures/";

        allFeaturesList = []
        for dirs in os.walk(dir):
            for thisdir in dirs[1]:
                features  = getEmobaseFeatures(dirs[0] + '' + thisdir +'/', bufferdir, thisdir)
                allFeaturesList.append(features)

        featuresTest = consolidate(allFeaturesList)

        # remove columns with string values:
        featuresTest = featuresTest.iloc[:,2:]

        # fix issue of not having index.
        featuresTest = featuresTest.reset_index(drop=True)

        print(featuresTest)

        return featuresTest







# functionals of all the features per file

# print(contents.Dataframe(1))
# mean1 = contents[1].mean()

