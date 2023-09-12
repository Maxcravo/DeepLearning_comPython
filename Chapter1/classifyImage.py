# Resolver um problema de classificação de imagens em 10 categorias
# As imagens são usadas do banco de dados do MNIST 

# em machine learning categorias(category) são chamadas de classes(class), e pontos de dados São chamados de samples
# a classe associada a cada sample é um label

from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy
import pandas

(trainImages, trainLabels), (testImages,testLabels) = mnist.load_data() # carregando as imagens do dataset

# trainImages e trainLabels são o conjunto de treino dos dados que o modelo irá aprender

print(numpy.shape(trainImages)) # resultado são 60000 amostras de treino com imagens 28x28
print(trainLabels)

# e os dados de test são

print(numpy.shape(testImages))
print(testLabels)

# o workflow é mais ou menos assim:
# 1) alimentamos a rede neural com os dados de treinamento (trainImages, trainLabels)
# 2) a rede vai aprender a associar imagens e os rótulos(labels)
# 3) vamos pedir para a rede produzir predições para nossos testes(testImages)
# 4) verificamos se essas predições batem com os rotulos(test_labels)  

#montando a arquitetura da rede

# layers servem como um filtro de dados aonde a partir desses layers os dados
# saem de uma forma mais útil.

# layers extraem representações dos dados alimentados neles. Consistindo no final
# em combinar layers simples que de forma progressiva implementam destilação de dados.

# um modelo de aprendizagem profunda é como uma peneira para o processamento de dados.\
# (A deep learning model is like a sieve for data processing)

model = keras.Sequential([
  layers.Dense(512, activation="relu"), # aqui estamos criando nosso layer
  layers.Dense(10, activation="softmax")
])

# Nosso modelo definido acima consiste em uma sequencia de dois layers Dense
# isso declara que esses dois layers são densamente conectados.

# o segundo layer criado é um 10-way "softmax classification layer", ou seja,
# vai retornar um array de com 10 pontuações de probabilidade

# cada pontuação vai ser a probabilidade que a imagem do digito atual
# pertence a uma de nossas 10 classes de digitos 10.

# precisamos ainda de mais 3 passos para a compilação do modelo.

# 1) O otimizador(Optimizer) no qual o model melhora a sí mesmo baseadoo nos dados treinados para melhorar performance
# 2) Função de perda(loss function) como o model vai mensurar a sua performance sobre os dados de dados
# e portante como ele vai dirigir a sí mesmo na direção certa 
# 3) métricas a serem monitoradas durante o treinamento e teste, nesse caso a fração de imagens corretamente classificadas.

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# precisamos transformar nossoos dados em um formato esperado pelo modelo

# vamos então transformar os dados em float32 de formato (6000, 28 * 28)

trainImages = numpy.reshape(trainImages,(60000, 28 * 28))
trainImages = trainImages.astype("float32") / 255
testImages = numpy.reshape(testImages,(10000, 28 * 28))
testImages = testImages.astype("float32") / 255

model.fit(trainImages, trainLabels, epochs=5, batch_size=128)

test_digits = testImages[0:10]
predictions = model.predict(test_digits)
print(predictions[0].argmax())
print(predictions[0][7])

# avaliando o modelo em novos digitos nunca antes vistos.    
# vemos que o resultado de accuracy é menor que o adquirido quando avaliamos em dados já conhecidos pelo modelo
# isso é chamado overfitting, que é o fato de que modelos de aprendizado de máquina tendem a performar pior em novos dados.
test_loss, test_acc = model.evaluate(testImages,testLabels)
print(f"test_acuracy: {test_acc}")