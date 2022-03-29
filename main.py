from pickletools import optimize
from tabnanny import verbose
import tensorflow as tf
import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

def descomprimirArchivos():
    trainZip = './zip/train.zip'
    zip_ref = zipfile.ZipFile(trainZip, 'r')
    zip_ref.extractall('./temp')
    zip_ref.close()

    testZip = './zip/test.zip'
    zip_ref = zipfile.ZipFile(testZip, 'r')
    zip_ref.extractall('./temp')
    zip_ref.close()

def ObtenerImagenesClasificadas():
    descomprimirArchivos()
    TRAININ_DIR = './temp/train'
    trainin_datagen = ImageDataGenerator(rescale= 1./255)

    trainin_generator = trainin_datagen.flow_from_directory(
        TRAININ_DIR,
        target_size= (150, 150),
        class_mode= 'categorical'
    )

    VALIDATION_DIR = './temp/test'
    validation_datagen = ImageDataGenerator(rescale= 1./255)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size= (150 ,150),
        class_mode= 'categorical'
    )

    return [trainin_generator, validation_generator]

def imagenesClasificadasBlancoYNegro():
    TRAININ_DIR = './temp/train'
    trainin_datagen = ImageDataGenerator(rescale= 1./255)
    trainin_generator = trainin_datagen.flow_from_directory(
        TRAININ_DIR,
        color_mode= 'grayscale',
        target_size= (150,150),
        class_mode='categorical',
        batch_size=8
    )

    VALIDATION_DIR = './temp/test'
    validation_datagen = ImageDataGenerator(rescale= 1./255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        color_mode= 'grayscale',
        target_size= (150,150),
        class_mode='categorical',
        batch_size=8
    )

    return [trainin_generator, validation_generator]


def crearModeloConvulucional():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(50, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(50, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(100, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(100, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(500, activation = 'relu'),
        tf.keras.layers.Dense(5, activation = 'softmax')
    ])

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = ['accuracy']
    )

    history = model.fit_generator(
        ObtenerImagenesClasificadas()[0], 
        epochs = 20, 
        validation_data = ObtenerImagenesClasificadas()[1], 
        verbose = 1)

    loss = history.history['loss']

    return loss

def crearModeloDenso1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (150, 150, 1)),
        tf.keras.layers.Dense(50, activation = tf.nn.relu),
        tf.keras.layers.Dense(50, activation = tf.nn.relu),
        tf.keras.layers.Dense(5, activation = tf.nn.softmax)
    ])

    model.compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    historia = model.fit_generator(
        imagenesClasificadasBlancoYNegro()[0], 
        epochs = 20,
        validation_data = imagenesClasificadasBlancoYNegro()[1],
        verbose = 1)

    loss = historia.history['loss']

    return loss

def crearModeloDenso2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (150, 150, 1)),
        tf.keras.layers.Dense(70, activation = tf.nn.relu),
        tf.keras.layers.Dense(70, activation = tf.nn.relu),
        tf.keras.layers.Dense(70, activation = tf.nn.relu),
        tf.keras.layers.Dense(5, activation = tf.nn.softmax)
    ])

    model.compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    historia = model.fit_generator(
        imagenesClasificadasBlancoYNegro()[0], 
        epochs = 20,
        validation_data = imagenesClasificadasBlancoYNegro()[1],
        verbose = 1)

    loss = historia.history['loss']

    return loss
    

def generarGrafica():
    loss_convulucional = crearModeloConvulucional()
    loss_denso1 = crearModeloDenso1()
    loss_denso2 = crearModeloDenso2()
    plt.xlabel('# Epoca')
    plt.ylabel('Magnitud de perdida')
    plt.plot(loss_convulucional, label = 'Convulucional')
    plt.plot(loss_denso1, label = 'Denso 2 C 50 N')
    plt.plot(loss_denso2, label = 'Denso 3 C 70 N')

    plt.show()

if __name__ == '__main__':
    # descomprimirArchivos()
    generarGrafica()