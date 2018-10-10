import pandas as pd
import numpy as np
import time
from datetime import timedelta
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from PredictBankruptcy import x_pca, x_complete, y, df

X_scale = StandardScaler()
X = X_scale.fit_transform(x_pca)

Y = y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# Fix random seed for reproducibility
np.random.seed(7)

start_time = time.time() # Start recording time of program

model = Sequential()
model.add(Dense(15, input_dim=7, activation='relu', kernel_initializer='uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.60, epochs=300, batch_size=25, verbose=0)

# List all data in history
#print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(X_test, y_test)

print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))

elapsed_time_secs = time.time() - start_time # Calculate elapsed time
msg = "Execution took: %s secs" % timedelta(seconds=round(elapsed_time_secs))

print("\n%s" % msg)

#print("\n%s" % scores)

#predictions = model.predict(X)
#rounded = [round(x[0]) for x in predictions]
#rounded = np.array(rounded)
#print(rounded)

#print("\nActual Results: %s" % Y)

#print("\nPredictions: %s" % rounded)

#data = np.stack((Y, rounded), axis=-1)
#data_df = pd.DataFrame(data)
#data_df.to_csv("Prediction_Comparison_Sigmoid(5-10-1).csv")




