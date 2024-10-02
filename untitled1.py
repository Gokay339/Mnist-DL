
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense , BatchNormalization
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from keras.optimizers import Adam
import warnings
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
from keras_preprocessing.image import img_to_array, load_img

def load_and_preprocess(data):
    data = data.to_numpy()   # dataframe i  numpy matrisine dönüştürür
    np.random.shuffle(data)  # verilerin rastgele karıştırılmasını sağlar
    x = data[:,1:].reshape(-1,28,28,1)/255.0
    y = data[:,0].astype(np.int32) # İlk sütundaki etiketleri alır ve int32 veri tipine dönüştürür
    y = to_categorical(y,num_classes=len(set(y)))  # onehot encoding yapmamızı sağlar
    # num_classes parametresi, sınıf sayısını belirlemek için kullanılır. 
    # len(set(y)), etiketlerin kaç farklı sınıfa ait olduğunu sayar.
    return x,y

train = pd.read_csv("C:/Users/gokay/OneDrive/Masaüstü/DerinOgrenme_1/odev3/mnist_train.csv")
test = pd.read_csv("C:/Users/gokay/OneDrive/Masaüstü/DerinOgrenme_1/odev3/mnist_test.csv")

x_train, y_train = load_and_preprocess(train)
x_test, y_test = load_and_preprocess(test)


#%% görselleştirme
index = 55
vis = x_train.reshape(60000,28,28) # XTRAİNDE 60000 GÖZÜKTÜĞÜ İÇİN O DEĞERİ VERİYORUZ KAFAMIZA GÖRE DEĞİL
plt.imshow(vis[index,:,:])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index]))




#%% CNN

# Veri artırma için ImageDataGenerator oluşturma
"""datagen = ImageDataGenerator(
    rotation_range=10,  # Rastgele döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    zoom_range=0.2,  # Zoom
    horizontal_flip=False  # Yatay çevirme
)"""


numberOfClass = y_train.shape[1]

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(28,28,1)))
# 3x3 boyutlu 16 filtre ekler
model.add(BatchNormalization())   # ağırlıkları normalleştirir
model.add(Activation("relu"))
model.add(MaxPooling2D())
# küçültme işlemi yapar

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.3))
# Dropout(0.2), aşırı öğrenmeyi önlemek için katmanın %20’sinin rastgele sıfırlanmasını sağlar

model.add(Flatten())          # çok boyutlu veriyi tek boyutlu yapar
model.add(Dense(units=256))
model.add(Activation("relu"))
model.add(Dropout(0.3))  
model.add(Dense(units=numberOfClass))
model.add(Activation("softmax")) # Son katmandaki"softmax")sınıflar arasında olasılık dağılımı oluşturur

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])


# EĞİTİM

"""hist = model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                  validation_data=(x_test, y_test), 
                  epochs=25)"""
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test)
                 ,epochs=25,batch_size=64)
#%%  ağırlıkları kaydetme
model.save_weights("cnn_mnist_model.h5")

#%% değerlendirme

print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()
plt.figure()

plt.plot(hist.history["accuracy"],label="Train Accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()

#%% save history
import json
with open('cnn_mnist_hist.json','w') as f:
    json.dump(hist.history,f)

#%% load history
import codecs
with codecs.open("cnn_mnist_hist.json","r",encoding="utf-8") as f:
    h = json.loads(f.read())
    
plt.plot(h["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.plot(h["accuracy"],label="Train acc")
plt.plot(h["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()





