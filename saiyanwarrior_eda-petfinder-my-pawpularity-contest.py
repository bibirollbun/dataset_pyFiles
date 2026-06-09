!pip install ipyplot -qq

import pandas as pd
import ipyplot
from PIL import Image

train = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
train.head()


# Range of score
train.Pawpularity.hist()


not_popular = train.Id[train.Pawpularity < 10].values.tolist()
popular = train.Id[train.Pawpularity > 90].values.tolist()
not_popular = [Image.open('../input/petfinder-pawpularity-score/train/' + x +'.jpg') for x in not_popular[:12]]
popular = [Image.open('../input/petfinder-pawpularity-score/train/' + x +'.jpg') for x in popular[:12]]


ipyplot.plot_images(not_popular, max_images=12, img_width=300)



ipyplot.plot_images(popular, max_images=12, img_width=300)





