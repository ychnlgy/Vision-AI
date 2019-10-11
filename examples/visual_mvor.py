import torch
from mvordata import *
from unet_mvor import *
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


model_path = 'model_100epoch'
model = Unet_Mvor()
model.load_state_dict(torch.load(model_path))
model.eval()
path = 'test2.pickle'
test_path = None
for i in range(5):
	with open(path, 'rb') as f:
		test_path = pickle.load(f)
		image_path = test_path[i]
		print(image_path)
		data = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)
		data = np.transpose(data, (2, 0, 1))
		data = torch.tensor(data).float().view(1, 3, 480, 640)
		start_time = time.time()
		xh = model(data)
		print("--- %s seconds ---" % (time.time() - start_time))
		xh = xh[0]
		xh = xh.permute(1, 2, 0)
		xh1 = xh[:, :, 1]
		xh0 = xh[:, :, 0]
		xh_box = xh1 > xh0
		f = plt.figure()
		f.add_subplot(1,2, 1)
		# imgplot = mpimg.imread(image_path[0])
		# plt.imshow(imgplot)
		# f.add_subplot(1,2, 2)
		# plt.imshow(xh_box)
		# plt.show(block=True)


#plt.imshow(mpimg.imread(image_path))
#imgplot = plt.imshow(xh_box)
#plt.show()

