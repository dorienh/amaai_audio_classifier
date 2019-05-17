from __future__ import print_function

import pandas as pd
# from ggplot import ggplot, geom_point, ggtitle, aes
from sklearn.manifold import TSNE
# import numpy as np
# import pandas as pd
# from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def visualise(data, type):
    """
    show
    :param data: dataframe with class attribute. Last two columns will be removed, they are typically: filename and class
    :param type: tsne, later pca will also be implemented and others
    """



    if (type=='tsne'):
        n_sne = 7000
        a = (len(data.columns))
        print(a)
        xvalues = data.iloc[:,:len(data.columns)-2]
        # xvalues = xvalues[2:]

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(xvalues)

        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['class'] = data['class']

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="class",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3
        )


        plt.show()
        plt.savefig('tsne.png')

    elif type == 'explore':
        data['class'].value_counts()

        sns.countplot(x='class', data=data, palette ='hls')
        plt.show()
        plt.savefig('classes.png')

    #     todo more data exploration


    else:
        print('No correct visualisation type has been given. ')

    pass

