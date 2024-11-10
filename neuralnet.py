import pickle
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


batch1 = unpickle("data_batch_1")

images = batch1[b"data"]
labels = batch1[b"labels"]
labels = np.array(labels).astype(np.int32)

images = images.reshape(-1, 32, 32, 3).astype(np.float32) / 255.0  # Normalize after reshaping


def generate_architecture():
    num_layers = random.choice([3, 4, 5, 6])
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(32, 32, 3)))

    current_size = 32
    architecture_details = {
        "num_layers": num_layers,
        "layers": []
    }
    
    for i in range(num_layers):
        filters = random.choice([32, 64, 128])
        kernel_size = random.choice([3,5])

        if current_size >= kernel_size:
            model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu'))
            architecture_details["layers"].append({"type": "Conv2D", "filters": filters, "kernel_size": kernel_size})
            current_size -= (kernel_size - 1)
        
        if current_size >= 2:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            architecture_details["layers"].append({"type": "MaxPooling2D", "pool_size": (2, 2)})
            current_size //= 2
    
    model.add(layers.GlobalAveragePooling2D())
    architecture_details["layers"].append({"type": "GlobalAveragePooling2D"})
    model.add(layers.Dense(10, activation='softmax'))
    architecture_details["layers"].append({"type": "Dense", "units": 10, "activation": "softmax"})
    
    return model, architecture_details





def nas(images, labels, num_candidates=10):
    best_accuracy = 0
    best_model = None
    results = []

    for _ in range(num_candidates):
        model, architecture_details = generate_architecture()
        val_accuracy = train_and_evaluate(model, images, labels)
        
        architecture_details["accuracy"] = val_accuracy
        results.append(architecture_details)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
        print(f"Candidate model accuracy: {val_accuracy:.4f}")
    
    print(f"Best model accuracy: {best_accuracy:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", mode='a', header=False, index=True)  # Append without headers
    print(results_df)
    
    return best_model

#bestmodel = nas(images, labels)

images, labels = [], []
for i in range(1, 6):  # CIFAR-10 has 5 data batches
    batch = unpickle(f"data_batch_{i}")
    images.append(batch[b"data"])
    labels.extend(batch[b"labels"])
    print(f"Batch {i} loaded.")

images = np.vstack(images).reshape(-1, 32, 32, 3).astype(np.float32) / 255.0
labels = np.array(labels).astype(np.int32)

def train_and_evaluate(model, images, labels, epochs=5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    split_index = int(0.8 * len(images))
    x_train, x_val = images[:split_index], images[split_index:]
    y_train, y_val = labels[:split_index], labels[split_index:]
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=64, verbose=0)
    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

# Training best model 

df = pd.read_csv("results.csv")  
best_model_row = df.loc[df["accuracy"].idxmax()]
best_layers = eval(best_model_row["layers"])  

def create_model_from_architecture(layers_config):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(32, 32, 3)))  # CIFAR-10 input shape

    for layer in layers_config:
        if layer["type"] == "Conv2D":
            model.add(layers.Conv2D(filters=layer["filters"], kernel_size=(layer["kernel_size"], layer["kernel_size"]), activation='relu'))
        elif layer["type"] == "MaxPooling2D":
            model.add(layers.MaxPooling2D(pool_size=layer["pool_size"]))
        elif layer["type"] == "GlobalAveragePooling2D":
            model.add(layers.GlobalAveragePooling2D())
        elif layer["type"] == "Dense":
            model.add(layers.Dense(units=layer["units"], activation=layer["activation"]))
    
    return model

model = create_model_from_architecture(best_layers)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 50  # Set higher epochs for extensive training
history = model.fit(images, labels, epochs=epochs, batch_size=64, validation_split=0.2, verbose=1)

model.save("best_cifar10_model.h5")