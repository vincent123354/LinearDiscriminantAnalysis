import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from lda import LDA

if __name__=='__main__':
	m = 10000

	x1 = np.random.normal(loc=[1, 2], scale=(0.5, 0.2), size=(m, 2))
	y1 = np.zeros((m, 1))

	x2 = np.random.normal(loc=[5, 4], scale=(0.6, 1.0), size=(m, 2))
	y2 = np.ones((m, 1))

	x3 = np.random.normal(loc=[10, 3], scale=(0.4, 0.1), size=(m, 2))
	y3 = np.ones((m, 1)) * 2

	x = np.concatenate([x1,x2,x3])
	y = np.concatenate([y1,y2,y3])

	x_train, x_test, y_train, y_test = train_test_split(x, y)

	lda = LDA()

	lda.fit(x_train, y_train)
	train_acc = (lda.predict(x_train).argmax(1)==y_train.squeeze()).mean() * 100
	test_acc = (lda.predict(x_test).argmax(1)==y_test.squeeze()).mean() * 100

	print(f'Train accuracy : {train_acc}%')
	print(f'Test accuracy : {test_acc}%')

	# plot graph
	xx1 = np.arange(x.min(), x.max(), 0.1)
	xx2 = np.arange(x.min(), x.max(), 0.1)
	xx1, xx2 = np.meshgrid(xx1, xx2)
	xx = np.c_[xx1.ravel(), xx2.ravel()]
	predict = lda.predict(xx).argmax(1)

	decision_boundary = lda.decision_function(xx)

	plt.plot(x1[:, 0], x1[:, 1], '.', label='class 1')
	plt.plot(x2[:, 0], x2[:, 1], '.', label='class 2')
	plt.plot(x3[:, 0], x3[:, 1], '.', label='class 3')
	plt.ylim(xx.min(), xx.max())
	plt.xlim(xx.min(), xx.max())
	plt.pcolormesh(xx1,xx2, predict.reshape(xx1.shape), cmap=plt.cm.Paired, shading='auto')
	plt.contour(xx1,xx2, decision_boundary[:, 0].reshape(xx1.shape), levels=[0])
	plt.contour(xx1,xx2, decision_boundary[:, 1].reshape(xx1.shape), levels=[0])
	plt.contour(xx1,xx2, decision_boundary[:, 2].reshape(xx1.shape), levels=[0])

	plt.title('Decision Boundary Plot')
	plt.ylabel('feature2')
	plt.xlabel('feature1')
	plt.legend()

	plt.show()
