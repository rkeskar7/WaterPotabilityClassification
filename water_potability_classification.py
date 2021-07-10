import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from collections import Counter

import matplotlib
from matplotlib import pyplot as plt
water_potability_classification = pd.read_csv("water_potability.csv") #Read in the csv file
water_potability_classification = water_potability_classification.dropna() #Drop any rows with
#print(water_potability_classification.head()) Examine the first few lines

#water_potability_classification.info() #Gather info about the dataset
water_potability_counter = Counter(water_potability_classification["Potability"]) #Count the number of classifications of potability


#Split the data
x_data = water_potability_classification.iloc[:,0:-1] #Gather all columns, except for the potability column
y_data = water_potability_classification.iloc[:,-1] #Gather only the potability column

#Split the data into training and testing data
x_training_data, x_testing_data, y_training_data, y_testing_data = train_test_split(x_data, y_data, test_size = 0.3, random_state = 7)
#Convert y testing data to a numpy array
#Create the model
water_potability_model = Sequential()

#Input layer
input_layer = InputLayer(input_shape = (x_training_data.shape[1], ))

water_potability_model.add(input_layer)

label_encoder = LabelEncoder()
y_training_data = label_encoder.fit_transform(y_training_data.astype(int))
y_testing_data = label_encoder.transform(y_testing_data.astype(int))

y_training_data = to_categorical(y_training_data, dtype = 'Int64')
y_testing_data = to_categorical(y_testing_data, dtype = 'Int64')

#Hidden layers
hidden_layer_1 = Dense(2048, activation = "relu") #2048 neurons
water_potability_model.add(hidden_layer_1)

hidden_layer_2 = Dense(2048, activation = "relu") #2048 neurons
water_potability_model.add(hidden_layer_2)

hidden_layer_3 = Dense(2048, activation = "relu") #2048 neurons
water_potability_model.add(hidden_layer_3)


hidden_layer_4 = Dense(2048, activation = "relu") #2048 neurons
water_potability_model.add(hidden_layer_4)


hidden_layer_5 = Dense(1024, activation = "relu") #1024 neurons
water_potability_model.add(hidden_layer_5)


hidden_layer_6 = Dense(1024, activation = "relu") #1024 neurons
water_potability_model.add(hidden_layer_6)


hidden_layer_7 = Dense(1024, activation = "relu") #1024 neurons
water_potability_model.add(hidden_layer_7)


hidden_layer_8 = Dense(2048, activation = "relu") #1024 neurons
water_potability_model.add(hidden_layer_8)


#Output layer
output_layer = Dense(2, activation = "softmax") #Two outcomes are possible (1 - Potable or 0 - Not Potable)
water_potability_model.add(output_layer)

#Compile the model
LOSS = "categorical_crossentropy"
OPTIMIZER = Adam(learning_rate = 0.01)
METRICS = ["accuracy"]
water_potability_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)

#Fit the model
EPOCHS = 70
BATCH_SIZE = 128
VERBOSE = 1
model_fit = water_potability_model.fit(x_training_data, y_training_data, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE)
loss, accuracy = water_potability_model.evaluate(x_testing_data, y_testing_data, verbose = 1)

print("The testing loss is " + str(loss)) #Print out the validation loss
print("The testing accuracy is " + str(accuracy)) #Print out the validation accuracy

#Test the model
y_testing_prediction = water_potability_model.predict(x_testing_data)
y_testing_prediction = np.argmax(y_testing_prediction, axis = 1)
print("y_testing_prediction using argmax")
print(y_testing_prediction)

y_testing_actual = np.argmax(y_testing_data, axis = 1)
print("y_testing_actual")
print(y_testing_actual)

classification_report = classification_report(y_testing_actual, y_testing_prediction)
print("Classification Report")
print(classification_report)

#Plot Validation Accuracy vs. Validation Loss
epochs_list = []
for i in range (1, EPOCHS + 1):
    epochs_list.append(i)
axes = plt.subplot()
validation_accuracy = model_fit.history["accuracy"]
validation_loss = model_fit.history["loss"]
plt.plot(validation_loss)
plt.plot(validation_accuracy)
plt.legend(['validation loss', 'validation accuracy'])
plt.xlabel('Epochs')
axes.set_xticks(range(len(epochs_list)))
axes.set_yticks([0,1])
axes.set_xticklabels(epochs_list)
plt.title("Validation Accuracy vs. Validation Loss")
plt.show()
