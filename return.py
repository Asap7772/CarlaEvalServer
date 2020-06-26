import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image
import io

path = '/home/asap7772/wrapper_env/2020-06-25 17:41:13.908294.pickle'
dic = pickle.load(open(path, 'rb'))
for x in dic:
    plt.plot(dic[x], label=x.split('/')[5])
plt.ylabel('Average Return')
plt.legend()
buf = io.BytesIO()
plt.draw()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img.show()
