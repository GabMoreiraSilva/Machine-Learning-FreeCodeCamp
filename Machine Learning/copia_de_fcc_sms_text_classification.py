# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# helps in text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

#Preparando dataset e separando Labels
train_file = pd.read_table(train_file_path, header=None)
test_file = pd.read_table(test_file_path, header=None)

#separando X/Y train_file
X_train = train_file[1]
Y_train = np.array(train_file[0].to_list())

#separando test_file
X_test = test_file[1]
Y_test = np.array(test_file[0].to_list())

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

#Pré-processando os dados
t = Tokenizer(vocab_size,oov_token=oov_tok)
t.fit_on_texts(X_train)

word_index = t.word_index

X_train_t = t.texts_to_sequences(X_train)
training_padded = pad_sequences(X_train_t, maxlen=max_length, padding=padding_type, truncating=trunc_type)


X_test_t = t.texts_to_sequences(X_test)
testing_padded = pad_sequences(X_test_t, maxlen=max_length, padding=padding_type, truncating=trunc_type)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_y = le.fit_transform(Y_train)
test_y = le.transform(Y_test)

training_padded = np.array(training_padded)
train_label = np.array(train_y)
testing_padded = np.array(testing_padded)
test_label = np.array(test_y)

#criando o modelo e treinando
model = keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 17
history = model.fit(training_padded, train_label, epochs=num_epochs, validation_data=(testing_padded, test_label), verbose=2)

#mostrando o treino gráficamente
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


#Criando função de teste
sentence = ["sale today! to stop texts call 98912460324"]

def previsao(sentence):
  sequences = t.texts_to_sequences(sentence)
  padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  return (float(model.predict(padded)), 'ham' if model.predict(padded) < 0.5 else 'spam' )

previsao(sentence)

#função para o teste
def predict_message(sentence):
  sequences = t.texts_to_sequences([sentence])
  padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  return (float(model.predict(padded)), 'ham' if model.predict(padded) < 0.5 else 'spam' )

#teste
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):   
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()

