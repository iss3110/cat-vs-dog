from utilities import *
x_train, y_train, x_test, y_test = load_data()

#print(x_train.shape)
#print(np.unique(y_train,return_counts=True))
#plt.figure(figsize=(16,8))
#for i in range(1,10):
#    plt.subplot(4,5,i)
#    plt.imshow(x_train[i],cmap='gray')
#    plt.title(y_test[i])
#    plt.tight_layout()
#plt.show()



#print(x_train.shape)
#print(x_train.ndim)


#print(x_test.flags)

x_train_reshaped = x_train.reshape(x_train.shape[0],-1)
x_test_reshaped = x_test.reshape(x_test.shape[0],-1)
W , b = artificial_neuron(x_train_reshaped, y_train, x_test_reshaped, y_test, learning_rate=0.1, n_iter=100)

