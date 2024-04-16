import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image
import random
from sklearn.utils import shuffle
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# super parameters
batch_size = 40
early_stop_patience = 14
epochs = 100
split_symbol = '/'
lr = 0.0005

# fixed parameters
num_classes = 24
image_size = 224


# load raw data
def get_origin_data():
    X = []
    Y = []
    path = u'../origin'
    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = dir_path + split_symbol + file_name
            array = np.array(Image.open(file_path), dtype=np.uint8)
            X.append(array)
            label_y = int(file_name.split('.')[-2])
            Y.append(label_y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# Model non c'è questo modello su pytorch TODO: da capire come sostituire 
""" def inception_residual_network(input_shape):
    inception = InceptionResNetV2(include_top=False,
                                  input_shape=input_shape)
    model = Flatten(name='flatten')(inception.output)
    model = Dense(num_classes, activation='softmax')(model)
    model = Model(inception.input, model, name='inception_residual_network')
    # model.summary()
    # plot_model(model, to_file=u'../model/inception_residual_network.png')
    return model """


################################################# test con inceptionV3 #########################################
def inception_residual_network(input_shape, num_classes):
    # Load pre-trained InceptionResNetV2 model from torchvision
    inception = models.inception_v3(pretrained=True)

    # Modify the model to match the TensorFlow architecture
    # Remove the final layer and set the pre-trained weights as not trainable
    inception.fc = nn.Identity()
    for param in inception.parameters():
        param.requires_grad = False

    # Define the new final layer
    model = nn.Sequential(
        inception,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, num_classes),  # Assuming the output of InceptionResNetV2 is 2048
        nn.Softmax(dim=1)
    )

    return model


################################################################################################################
################################################### test con resnet 50 #########################################
def resnet50_chromosome_classifier(num_classes):
    # Load pre-trained ResNet-50 model from torchvision
    resnet = models.resnet50(pretrained=True)

    # Modify the model to fit the number of classes in your task
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)

    return resnet


################################################################################################################


# training
""" def training(model, x_train, y_train, x_test, y_test, model_name):
    parallel_model = model
    # multi_gpu_model(model, gpus=2) # if you have multi GPU
    parallel_model.compile(loss=categorical_crossentropy,
                           # optimizer=rmsprop(lr=lr, decay=0.9),
                           optimizer=adam(lr=lr),
                           # optimizer=optimizer,   # vgg16
                           metrics=['accuracy'])
    early_stop_callback = EarlyStopping(monitor='val_acc',
                                        patience=early_stop_patience,
                                        verbose=1,
                                        restore_best_weights=True,
                                        mode='auto')
    # if you need to see the full-stack log.
    # log_callback = TensorBoard(log_dir=u'../data/log',
    #                            histogram_freq=0,
    #                            write_graph=True,
    #                            write_grads=True,
    #                            write_images=True,
    #                            embeddings_freq=0)
    parallel_model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[early_stop_callback,
                                  # log_callback
                                  ],
                       validation_data=(x_test, y_test))
    test_score = parallel_model.evaluate(x_test, y_test, verbose=1)
    print(test_score)
    val_acc = test_score[1]
    parallel_model.save(u'../model/' + model_name + '.' + str(val_acc) + '.h5')
    return parallel_model """


def training(model, x_train, y_train, x_test, y_test, model_name, batch_size=32, epochs=10, lr=0.001,
             early_stop_patience=3):
    # Assuming x_train, y_train, x_test, y_test are numpy arrays or PyTorch tensors

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch tensors and create DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move model to device (e.g., GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            val_preds = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
            val_acc = accuracy_score(y_test, val_preds)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_acc:.4f}")

            # Check for early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"../model/{model_name}_{best_val_acc:.4f}.pt")
            else:
                if epoch - early_stop_patience >= 0:
                    print("Early stopping.")
                    break

    return model


# run
def run_train(mode):
    X, Y = get_origin_data()
    print('X.shape, Y.shape:', X.shape, Y.shape)
    scores = []
    index = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(X, Y):
        train_data = X[train_idx]  # 80% of data
        val_data = X[val_idx]  # 20% of data
        train_target = Y[train_idx]
        val_target = Y[val_idx]
        train_x, train_y = [], []
        if mode == 'cda':
            for (x, y) in zip(train_data, train_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                for times in range(int(360 / 15)):
                    train_y.append(label_y)
                    train_x.append(np.array(Image.fromarray(x).rotate(times * 15)))

            train_x = np.array(train_x)
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
            train_y = np.array(train_y)

            val_x, val_y = [], []
            for (x, y) in zip(val_data, val_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                val_y.append(label_y)
                val_x.append(np.array(Image.fromarray(x).rotate(random.randint(0, 360))))
            val_x = np.array(val_x)
            val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
            val_y = np.array(val_y)
        elif mode == 'straighten':
            for (x, y) in zip(train_data, train_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                train_y.append(label_y)
                train_x.append(x)

            train_x = np.array(train_x)
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
            train_y = np.array(train_y)

            # prepare val data
            val_x, val_y = [], []
            for (x, y) in zip(val_data, val_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                val_y.append(label_y)
                val_x.append(x)
            val_x = np.array(val_x)
            val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
            val_y = np.array(val_y)
        else:  # mode = None
            for (x, y) in zip(train_data, train_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                train_y.append(label_y)
                train_x.append(np.array(Image.fromarray(x).rotate(random.randint(0, 360))))
            train_x = np.array(train_x)
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
            train_y = np.array(train_y)

            # prepare val data
            val_x, val_y = [], []
            for (x, y) in zip(val_data, val_target):
                label_y = np.zeros(24)
                label_y[y] = 1
                val_y.append(label_y)
                val_x.append(np.array(Image.fromarray(x).rotate(random.randint(0, 360))))
            val_x = np.array(val_x)
            val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
            val_y = np.array(val_y)

        train_x = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in train_x]
        train_x = np.array(train_x)
        val_x = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in val_x]
        val_x = np.array(val_x)
        train_x, train_y = shuffle(train_x, train_y)

        model = inception_residual_network((image_size, image_size, 3))
        model_name = 'inception_residual_network_cross_validation'

        model = training(model=model,
                         model_name=model_name,
                         x_train=train_x,
                         y_train=train_y,
                         x_test=val_x,
                         y_test=val_y)
        # predict_data_processing(model=model,
        #                         model_name= model_name + str(index),
        #                         y_test=val_x,
        #                         x_test=val_y)
        index += 1
        test_score = model.evaluate(val_x, val_y, verbose=1)
        scores.append(test_score[1])
        print(test_score[1])
        model = None
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.array(scores).mean(), np.array(scores).std() * 2))


# cda, straighten, or None
mode = 'straighten'
run_train(mode)
