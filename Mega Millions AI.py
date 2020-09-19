#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import time
from selenium import webdriver
from bs4 import BeautifulSoup


driver = webdriver.Chrome('chromedriver')
driver.maximize_window()
driver.get('https://www.megamillions.com/Winning-Numbers/Previous-Drawings.aspx')
time.sleep(3)
driver.execute_script('document.getElementsByClassName("loadMoreBtn")[0].scrollIntoView()')
load_more_button = driver.find_element_by_class_name('loadMoreBtn')
while load_more_button.is_displayed():
    load_more_button.click()
    time.sleep(1)

soup = BeautifulSoup(driver.page_source, 'lxml')
winners = soup.find_all('a', class_='prevDrawItem')
driver.close()


Num1s, Num2s, Num3s, Num4s, Num5s, NumMBs = [], [], [], [], [], []
for winner in winners:
    if winner.find(class_="drawItemDate").get_text() == '10/27/2017':
        break
    Num1s.append(winner.find(class_="pastNum1").get_text())
    Num2s.append(winner.find(class_="pastNum2").get_text())
    Num3s.append(winner.find(class_="pastNum3").get_text())
    Num4s.append(winner.find(class_="pastNum4").get_text())
    Num5s.append(winner.find(class_="pastNum5").get_text())
    NumMBs.append(winner.find(class_="pastNumMB").get_text())


df = pd.DataFrame({'Number 1': Num1s, 'Number 2': Num2s, 'Number 3': Num3s, 'Number 4': Num4s, 
              'Number 5': Num5s, 'Megaball': NumMBs})
df = df.astype('float32')
df = df[::-1]
df.head()


def split_dataset(data):
    features = data[:,:-1,:]
    labels = data[:,-1:,:]
    return features, labels


def create_dataset(data):
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data,
        targets = None,
        sequence_length = 2, 
        sequence_stride = 1, 
        batch_size = 1, 
        shuffle=False, 
    )

    return ds.map(split_dataset)

ds = create_dataset(df)


def lstm_model():
    inputs = tf.keras.Input(shape = (1, 6), name='input')

    num_1 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)
    num_2 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)
    num_3 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)
    num_4 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)
    num_5 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)
    megaball = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')(inputs)

    x = tf.keras.layers.concatenate([num_1, num_2, num_3, num_4, num_5, megaball])

    num_1_prediction = tf.keras.layers.Dense(70, name="num_1", activation='softmax')(x)
    num_2_prediction = tf.keras.layers.Dense(70, name="num_2", activation='softmax')(x)
    num_3_prediction = tf.keras.layers.Dense(70, name="num_3", activation='softmax')(x)
    num_4_prediction = tf.keras.layers.Dense(70, name="num_4", activation='softmax')(x)
    num_5_prediction = tf.keras.layers.Dense(70, name="num_5", activation='softmax')(x)
    megaball_prediction = tf.keras.layers.Dense(25, name="megaball", activation='softmax')(x)

    model = tf.keras.Model(
        inputs=[inputs],
        outputs=[num_1_prediction, num_2_prediction, num_3_prediction, num_4_prediction, num_5_prediction, megaball_prediction]
    )

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    return model

model = lstm_model()


features = []
num_1, num_2, num_3, num_4, num_5, megaball = [], [], [], [], [], []
for feature, label in ds:
    features.append(feature.numpy().reshape(-1,6))
    labels = label.numpy().reshape(-1,6)
    num_1.append(labels[0,0])
    num_2.append(labels[0,1])
    num_3.append(labels[0,2])
    num_4.append(labels[0,3])
    num_5.append(labels[0,4])
    megaball.append(labels[0,5])
    
features = np.array(features)
num_1 = np.array(num_1).reshape(-1, 1, 1) - 1
num_2 = np.array(num_2).reshape(-1, 1, 1) - 1
num_3 = np.array(num_3).reshape(-1, 1, 1) - 1
num_4 = np.array(num_4).reshape(-1, 1, 1) - 1
num_5 = np.array(num_5).reshape(-1, 1, 1) - 1
megaball = np.array(megaball).reshape(-1, 1, 1) - 1

for index, divisor in enumerate([70, 70, 70, 70, 70, 25]):
    features[:, :, index] = features[:, :, index]/divisor


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
model.fit(
    {"input": features},
    {"num_1": num_1, "num_2": num_2,
     "num_3": num_3, "num_4": num_4, 
     "num_5": num_5, "megaball": megaball},
    epochs = 20,
    batch_size = 32,
    validation_split = 0.1,
    callbacks=[early_stopping]
)


prev_num_1 = 25
prev_num_2 = 28
prev_num_3 = 38
prev_num_4 = 59
prev_num_5 = 62
prev_megaball = 22

random_numbers = np.array([[prev_num_1/70, prev_num_2/70, prev_num_3/70, prev_num_4/70, prev_num_5/70, prev_megaball/25]])
random_numbers = random_numbers.reshape(-1,1,6)
prediction = model.predict(random_numbers)
mega_million_numbers = [np.argmax(prediction[0]) + 1,
 np.argmax(prediction[1]) + 1,
 np.argmax(prediction[2]) + 1,
 np.argmax(prediction[3]) + 1,
 np.argmax(prediction[4]) + 1,
 np.argmax(prediction[5]) + 1]
print(mega_million_numbers)
