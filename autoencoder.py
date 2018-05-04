########################################################################################################################
## DEPENDENCIES
########################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from skimage import filters
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm



########################################################################################################################
## GLOBALS
########################################################################################################################
CUDA_AVAILABLE = torch.cuda.is_available()
IMG_SHAPE = (256, 256) # (28, 28) # (3024, 3024)
IMAGE_SIZE = IMG_SHAPE[0]**2
IMAGE_WIDTH = IMAGE_HEIGHT = IMG_SHAPE[0]
CODE_SIZE = 100
EPOCHS = 20
BATCH_SIZE = 128 # 128
LR = 1e-3 # 1e-3
PATH_TO_DATA = "data/train/type_2"


########################################################################################################################
## GLOBALS
########################################################################################################################
def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 1, IMG_SHAPE[0], IMG_SHAPE[1])
	return x
def convLayerOutputSize(F, stride, pad, N=IMG_SHAPE[0]):
	'''
	convLayerOutputSize(F=3, stride=1, pad=0)
	'''
	return(
		((N+(pad*2)-F)/stride) + 1
	)
def convTransLayerOutputSize(N, F, stride, pad):
	'''
	convTransLayerOutputSize(F=3, stride=1, pad=0)
	'''
	return(
		stride*(N-1)+F-2*pad
	)


########################################################################################################################
## DATA LOADING
########################################################################################################################
# img_transform = transforms.Compose([
# 	transforms.Grayscale(),
# 	transforms.Resize(IMG_SHAPE[0]),
# 	transforms.CenterCrop(IMG_SHAPE[0]),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# dataset = ImageFolder(
# 	root=os.path.join(PATH_TO_DATA),
# 	transform=img_transform
# )
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Get sample image from dataset
# dataset.classes
# sample = np.array(dataset[0][0].tolist())[0]
# plt.imshow(sample, cmap='gray'), plt.show()


########################################################################################################################
## MODEL DEFINITION
########################################################################################################################
class ConvAutoencoder(nn.Module):
	def __init__(self):
		super(ConvAutoencoder, self).__init__()
		self.conv_encoder = nn.Sequential(
			# convLayerOutputSize(N=IMG_SHAPE[0], F=6, stride=2, pad=0)
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=6, stride=2, padding=0),  # b, 8, 126, 126
			nn.ReLU(True),
			nn.Conv2d(in_channels=8, out_channels=4, kernel_size=6, stride=2, padding=0),  # b, 4, 61, 61
			nn.ReLU(True),
			nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=0),  # b, 2, 30, 30
			nn.ReLU(True),
			nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=3, padding=0),  # b, 1, 10, 10
			nn.ReLU(True)

		)
		self.linear_encoder = nn.Sequential(
			nn.Linear(100, 64),
			nn.ReLU(True),
			nn.Linear(64, 25),
			nn.ReLU(True)
		)
		self.linear_decoder = nn.Sequential(
			nn.Linear(25, 64),
			nn.ReLU(True),
			nn.Linear(64, 100),
			nn.ReLU(True),
			nn.Linear(100, IMG_SHAPE[0]*IMG_SHAPE[1]),
			nn.ReLU(True)
		)
		self.conv_decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=3, stride=3),  # b, 2, 30, 30
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=3, stride=2),  # b, 4, 61, 61
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=6, stride=2),  # b, 8, 126, 126
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=6, stride=2),  # b, 1, 512, 512
			nn.ReLU(True)
		)

	def forward(self, x):
		# print("BEGIN CONV ENCODE")
		conv_encode = self.conv_encoder(x)
		# print("END CONV ENCODE")

		# print("BEGIN FLATTEN")
		flat_conv_encode = conv_encode.view(conv_encode.size(0), -1)
		# print("END FLATTEN")

		# print("BEGIN LINEAR ENCODE")
		code = self.linear_encoder(flat_conv_encode)
		# print("END LINEAR ENCODE")

		# print("BEGIN LINEAR DECODE")
		decode = self.linear_decoder(code)
		# print("END LINEAR DECODE")

		# print("BEGIN CONV DECODE")
		# out = self.conv_decoder(decode.view(10, 10).unsqueeze(0).unsqueeze(0))
		#out = self.conv_decoder(decode.view([decode.size(0), 1, 10, 10]))
		# print("END CONV DECODE")

		return decode

	# def encode(self, x):
	# 	r = self.encoder(x)
	# 	return r
	#
	# def decode(self, x):
	# 	r = self.decoder(x)
	# 	return r

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(IMG_SHAPE[0] * IMG_SHAPE[1], 4096),
			nn.ReLU(True),
			nn.Linear(4096, 2048),
			nn.ReLU(True),
			nn.Linear(2048, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 512),
			nn.ReLU(True),
			nn.Linear(512, 256),
			nn.ReLU(True),
			nn.Linear(256, 128),
			nn.ELU()
		)
		self.decoder = nn.Sequential(
			nn.Linear(128, 256),
			nn.ReLU(True),
			nn.Linear(256, 512),
			nn.ReLU(True),
			nn.Linear(512, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 2048),
			nn.ReLU(True),
			nn.Linear(2048, 4096),
			nn.ReLU(True),
			nn.Linear(4096, IMG_SHAPE[0] * IMG_SHAPE[1]),
			nn.ELU()
		)

	def forward(self, x):
		code = self.encoder(x)
		out = self.decoder(code)
		return code, out


########################################################################################################################
## TRAIN
########################################################################################################################


# CONV Model Architecture
# code = model.encode(
# 	Variable(
# 		dataset[0][0].unsqueeze(0)
# 	)
# ).squeeze().data.numpy()
# code.shape
# plt.imshow(code, cmap='gray'), plt.show()
#
# code, out = model.forward(
# 	Variable(
# 		dataset[1][0].unsqueeze(0)
# 	)
# )
# code = code.squeeze().data.numpy()
# out = out.squeeze().data.numpy()
# plt.imshow(code, cmap='gray'), plt.show()
# plt.imshow(out, cmap='gray'), plt.show()



# FC Model Architecture
# out = model.forward(
# 	Variable(
# 		dataset[0][0].unsqueeze(0).view(IMG_SHAPE[0]*IMG_SHAPE[1])
# 	)
# )
# out = out.data.numpy().reshape(IMG_SHAPE[0], IMG_SHAPE[1])
# plt.imshow(out, cmap='gray'), plt.show()


# CONV to FC Model Architecture
# out = model.forward(
# 	Variable(
# 		dataset[0][0].unsqueeze(0)
# 	)
# )
# out = out.data.numpy().reshape(IMG_SHAPE[0], IMG_SHAPE[1])
# plt.imshow(out, cmap='gray'), plt.show()


# loss = [0.2953468859195709, 0.28386807441711426, 0.267879456281662, 0.21737217903137207, 0.1944325864315033, 0.16255643963813782, 0.1725526601076126, 0.16674351692199707, 0.15301336348056793, 0.1602495163679123, 0.13760493695735931, 0.12440656870603561, 0.11369849741458893, 0.10819631069898605, 0.11254667490720749, 0.11354783922433853, 0.10089431703090668, 0.10193857550621033, 0.09554678201675415, 0.09509849548339844]


# Training loop
if __name__ == "__main__":
	img_transform = transforms.Compose([
		transforms.Grayscale(),
		transforms.Resize(IMG_SHAPE[0]),
		transforms.CenterCrop(IMG_SHAPE[0]),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	dataset = ImageFolder(
		root=os.path.join(PATH_TO_DATA),
		transform=img_transform
	)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



	model = autoencoder() # model = ConvAutoencoder()
	if CUDA_AVAILABLE:
		model = model.cuda()
		print("MODEL IN CUDA MODE...")
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)



	agg_loss = []
	for epoch in range(EPOCHS):
		for i, (images, _) in enumerate(tqdm(dataloader)):
			# print("Batch %d" % i)
			images = images.view(images.size(0), -1)

			if CUDA_AVAILABLE:
				images = Variable(images).cuda()
			else:
				images = Variable(images)

			# ===================forward=====================
			code, out = model(images)
			loss = criterion(out, images)
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# ===================log========================
		print('epoch [{}/{}], loss:{:.4f}'
			  .format(epoch + 1, EPOCHS, loss.data[0]))
		agg_loss.append(loss.data[0])
		if epoch % 1 == 0:
			pic = to_img(out.cpu().data)
			save_image(pic, './output/image_{}.png'.format(epoch))
			try:
				np.save("./encodings/encoding_epoch_{}.npy".format(epoch), code.cpu().data.numpy())
			except Exception:
				print("Couldn't save encoding to numpy file")
				pass

		print("AGGREGATED LOSS : {}".format(agg_loss))
		try:
			torch.save(model, "./models/autoencoder.pt")
		except Exception:
			print("Couldn't save model to file...")
			pass

# .. to load your previously training model:
# model = torch.load(
# 	"./models/conv_autoencoder_LR_0.001.pt",
# 	map_location=lambda storage, loc: storage
# )
