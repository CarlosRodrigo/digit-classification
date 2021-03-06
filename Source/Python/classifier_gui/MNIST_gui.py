#Number Classifier GUI

from Tkinter import *
import random

from MNISTClassifier import predict, initClassifier
from sklearn.externals import joblib
import numpy as np

def gray_hex(grayvalue):
	grayvalue = 255 - grayvalue
	return '#%02x%02x%02x' % (grayvalue, grayvalue, grayvalue)


class MNISTGUI:
	def __init__(self):
		self.root = Tk()
		self.canvas = Canvas(self.root, width=569, height=569)
		self.canvas.pack()

		self.canvas.bind("<Configure>", self._on_resize)
		self.canvas.bind("<Button-1>", self._on_click)
		self.canvas.bind("<B1-Motion>", self.paint)

		self.reset_button = Button(self.root, text="Reset", command=self.canvas_reset)
		self.reset_button.pack(side=LEFT)

		self.classify_button = Button(self.root, text="Classify", command=self.classify)
		self.classify_button.pack(side=RIGHT)
		
		self.root.geometry('{}x{}'.format(569, 615))
		self.root.wm_title("MNIST GUI Classifier")
		
		self.model = np.zeros(784)
		self.classifier = joblib.load('../deep_learning/ConvNetModel.pkl') 


	def _on_resize(self, event):
		self.canvas_width = self.canvas.winfo_width()
		self.canvas_height = self.canvas.winfo_height()

		self.number_width = self.canvas_width // 28
		self.number_height = self.canvas_height // 28

		self.h_pixels = 20
		self.v_pixels = 20

		# print ('Canvas: ', self.canvas_width, self.canvas_height)

		self.draw()


	def _on_click(self, event):
		self.draw()


	def draw(self):
		self.canvas.delete(ALL)

		self.xcoord = 4
		self.ycoord = 4

		for i in range(28):
			for j in range(28):
				self.canvas.create_rectangle(self.xcoord, self.ycoord, self.xcoord+self.h_pixels, self.ycoord+self.v_pixels, fill=gray_hex(self.model[i*28+j]))
				self.xcoord += self.h_pixels
			self.xcoord = 4
			self.ycoord += self.v_pixels


	def _fill_model(self, x, y):
		# print ('Fill model:', x, y)

		for i in range(-1, 2):
			for j in range(-1, 2):
				pos = (y+i) * 28 + (x+j)
				current = 0

				if pos >= 0 and pos <= 28*28 and x+j >= 0 and x+j < 28 and y+i >= 0 and y+i < 28:
					current = self.model[pos]

					if i == j == 0:
						new_value = random.randint(230, 255)
					else:
						new_value = random.randint(200, 210)

					if new_value > current:
						self.model[pos] = new_value


	def paint(self, event):
		# print ('Mouse:', event.x, event.y)
		x = int(event.x // self.number_width)
		y = int(event.y // self.number_height)

		self._fill_model(x, y)
		self.draw()


	def classify(self):
		# prediction = predict(self.classifier, self.model)
		prediction = self.classifier.predict(self.model.reshape((-1, 1, 28, 28)))
		print 'Numero escrito foi', prediction


	def canvas_reset(self):
		self.model = np.zeros(784)
		self.draw()


	def start(self):
		self.root.mainloop()


if __name__ == '__main__':
	gui = MNISTGUI()
	gui.start()