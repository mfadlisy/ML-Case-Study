import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
data = pd.read_csv('/Users/muhammadfadlisyukur/Downloads/data_kartu_kredit.csv')
data.shape

# Melakukan standarisasi kolom amount
from sklearn.preprocessing import StandardScaler
data['Standar'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1)) 

# Mendefinsikan variabel dependen (y) dan independen (X)
y = np.array(data.iloc[:,-2])
X = np.array(data.drop(['Time','Amount','Class'], axis=1))

# Membagi data ke training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=111)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, 
                                                            test_size=0.2, 
                                                            random_state=111)

# Membuat desain model ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout
classifier = Sequential()
classifier.add(Dense(units=16, input_dim=29, activation='relu'))
classifier.add(Dense(24, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(20, activation='relu'))
classifier.add(Dense(24, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
classifier.summary()

# Membuat visualisasi model ANN
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_ann.png', show_shapes=True, show_layer_names=False)

# Prosess training model ANN
run_model = classifier.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1,
                           validation_data=(X_validate, y_validate))

# Melihat parameter apa saja yang tersimpan
print(run_model.history.keys())

# plot accuracy training dan validation set
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# plot loss training dan validation set
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# mengevaluasi model ANN
evaluasi = classifier.evaluate(X_test, y_test)
print('evaluasi:{:.2f}'.format(evaluasi[1]*100))

# memprediksi test set
hasil_prediksi = classifier.predict(X_test)
predicted_class_binary = (hasil_prediksi > 0.5).astype(int)

# membuat confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_class_binary)

# membuat cm dengan seaborn
cm_label = pd.DataFrame(cm, columns=np.unique(y_test), index=np.unique(y_test))
cm_label.index.name = 'Aktual'
cm_label.columns.name = 'Prediksi'
sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')

# membuat classification report
from sklearn.metrics import classification_report
jumlah_kategori = 2
target_names = ['class {}'.format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, predicted_class_binary, target_names=target_names))
    
# melakukan teknik undersampling
index_fraud = np.array(data[data.Class == 1].index)
n_fraud = len(index_fraud)
index_normal = np.array(data[data.Class == 0].index)
index_data_normal = np.random.choice(index_normal, n_fraud, replace=False)
index_data_baru = np.concatenate([index_fraud, index_data_normal])
data_baru = data.iloc[index_data_baru,:]

# membagi ke variabel dependen (y) dan indenpenden (X)
y_baru = np.array(data_baru.iloc[:,-2])
X_baru = np.array(data_baru.drop(['Time','Amount','Class'], axis=1))

# membagi ke train dan test set
X_train2, X_test_final, y_train2, y_test_final = train_test_split(X_baru, y_baru,
                                                                  test_size=0.1,
                                                                  random_state=111)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train2, y_train2,
                                                                  test_size=0.1,
                                                                  random_state=111)
X_train2, X_validate2, y_train2, y_validate2 = train_test_split(X_train2, y_train2,
                                                                  test_size=0.2,
                                                                  random_state=111)

# merancang ANN yang baru (model balanced)
classifier2 = Sequential()
classifier2.add(Dense(units=16, input_dim=29, activation='relu'))
classifier2.add(Dense(24, activation='relu'))
classifier2.add(Dropout(0.25))
classifier2.add(Dense(20, activation='relu'))
classifier2.add(Dense(24, activation='relu'))
classifier2.add(Dense(1, activation='sigmoid'))
classifier2.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
classifier2.summary()

# Prosess training model ANN
run_model2 = classifier2.fit(X_train2, y_train2, batch_size=8, epochs=5, verbose=1,
                           validation_data=(X_validate2, y_validate2))

# plot accuracy training dan validation set
plt.plot(run_model2.history['accuracy'])
plt.plot(run_model2.history['val_accuracy'])
plt.title('model2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# plot loss training dan validation set
plt.plot(run_model2.history['loss'])
plt.plot(run_model2.history['val_loss'])
plt.title('model2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# mengevaluasi model ANN
evaluasi2 = classifier2.evaluate(X_test2, y_test2)
print('evaluasi:{:.2f}'.format(evaluasi2[1]*100))

# memprediksi test set
hasil_prediksi2 = classifier2.predict(X_test2)
predicted_class_binary2 = (hasil_prediksi2 > 0.5).astype(int)

# membuat confusion matrix
cm2 = confusion_matrix(y_test2, predicted_class_binary2)
cm_label2 = pd.DataFrame(cm2, columns=np.unique(y_test2), index=np.unique(y_test2))
cm_label2.index.name = 'Aktual'
cm_label2.columns.name = 'Prediksi'
sns.heatmap(cm_label2, annot=True, cmap='Reds', fmt='g')

# membuat classification report
print(classification_report(y_test2, predicted_class_binary2, target_names=target_names))

# memprediksi test set final
hasil_prediksi3 = classifier2.predict(X_test_final)
predicted_class_binary3 = (hasil_prediksi3 > 0.5).astype(int)

# membuat confusion matrix
cm3 = confusion_matrix(y_test_final, predicted_class_binary3)
cm_label3 = pd.DataFrame(cm3, columns=np.unique(y_test_final), index=np.unique(y_test_final))
cm_label3.index.name = 'Aktual'
cm_label3.columns.name = 'Prediksi'
sns.heatmap(cm_label3, annot=True, cmap='Reds', fmt='g')

# membuat classification report
print(classification_report(y_test_final, predicted_class_binary3, target_names=target_names))


















