# Based on https://github.com/SherlockLiao/pytorch-beginner/tree/master/08-AutoEncoder

########################################################################################################################
## DEPENDENCIES
########################################################################################################################
import os
import numpy as np
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



########################################################################################################################
## GLOBALS
########################################################################################################################
CUDA_AVAILABLE = torch.cuda.is_available()
IMG_SHAPE = (28, 28) # (28, 28) # (3024, 3024)
IMAGE_SIZE = IMG_SHAPE[0]**2
IMAGE_WIDTH = IMAGE_HEIGHT = IMG_SHAPE[0]
CODE_SIZE = 100
EPOCHS = 5
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


class AutoEncoder(nn.Module):
	def __init__(self, code_size):
		super().__init__()
		self.code_size = code_size

		# Encoder specification
		self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
		self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
		self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
		self.enc_linear_2 = nn.Linear(50, self.code_size)

		# Decoder specification
		self.dec_linear_1 = nn.Linear(self.code_size, 160)
		self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)

	def forward(self, images):
		code = self.encode(images)
		out = self.decode(code)
		return out, code

	def encode(self, images):
		code = self.enc_cnn_1(images)
		code = F.selu(F.max_pool2d(code, 2))

		code = self.enc_cnn_2(code)
		code = F.selu(F.max_pool2d(code, 2))

		code = code.view([images.size(0), -1])
		code = F.selu(self.enc_linear_1(code))
		code = self.enc_linear_2(code)
		return code

	def decode(self, code):
		out = F.selu(self.dec_linear_1(code))
		out = F.sigmoid(self.dec_linear_2(out))
		out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
		return out


# class ConvAutoencoder(nn.Module):
# 	def __init__(self):
# 		super(ConvAutoencoder, self).__init__()
# 		self.encoder = nn.Sequential(
# 			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=3, padding=0),  # b, 16, 170, 170
# 			nn.ReLU(True),
# 			nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=0),  # b, 8, 56, 56
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2, stride=2),  # b, 8, 28, 28
# 			nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=3, padding=1),  # b, 8, 10, 10
# 			nn.ReLU(True)
# 		)
# 		self.decoder = nn.Sequential(
# 			nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),  # b, 16, 21, 21
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0),  # b, 16, 45, 45
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=7, stride=5, padding=0),  # b, 8, 227, 227
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=6, stride=3, padding=86),  # b, 8, 512, 512
# 			nn.Tanh()
# 		)
#
# 	def forward(self, x):
# 		r = self.encoder(x)
# 		r = self.decoder(x)
# 		return r
#
# 	def encode(self, x):
# 		r = self.encode(x)
# 		return r
#
# 	def decode(self, x):
# 		r = self.decode(x)
# 		return r
#



model = AutoEncoder(CODE_SIZE)
if CUDA_AVAILABLE:
	model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# Training loop
agg_loss = []
for epoch in range(EPOCHS):
	for i, (images, _) in enumerate(dataloader):
		print("Batch %d" % i)
		if CUDA_AVAILABLE:
			images = Variable(images.cuda())
		else:
			images = Variable(images)

		# ===================forward=====================
		out, code = model(images)
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

torch.save(model.state_dict(), './conv_autoencoder.pth')
