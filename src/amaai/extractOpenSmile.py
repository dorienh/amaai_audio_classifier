import os
import subprocess
import sys
from multiprocessing.dummy import Pool as ThreadPool

def extractFeatures(inpath, outpath, configfile):
    commands = []
    pool = ThreadPool(4)
    audios = os.listdir(inpath)
    print(audios)
    for audio in audios:
        # command1 = "SMILExtract -C /Users/dorienherremans/Dropbox/DoBrain/AC/Projects/kaggle/config/openSmile/IS13_ComParE_lld-func.conf.inc -I " + str(inpath) + audio + " -O " + str(outpath) + audio + ".csv"
        command1 = "SMILExtract -C ../config/openSmile/" + configfile + " -I " + str(inpath) + audio + " -O " + str(outpath) + audio + ".csv"
        commands.append(command1)
    pool.map(runcommand, commands)

def runcommand(command):
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    extractFeatures(sys.argv[1], sys.argv[2])
