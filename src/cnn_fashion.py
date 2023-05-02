##### IMPORTING DEPENDENCIES #####
# system tools and parse
import os 
import argparse
import warnings
warnings.filterwarnings("ignore")
# data tools
import pandas as pd
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.efficientnet import (EfficientNetB0)
# layers
from tensorflow.keras.layers import (Flatten, Dense, BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser() # This is the argument parser. I add the arguments below.
    ap.add_argument("-bs",
                    "--batch_size",
                    help="Batch size for training.",
                    type = int, default=64) # This is the argument for the batch size.
    ap.add_argument("--train_subset",
                    help="Number of training images to use. If not specified, all images are used.",
                    type = int, default=91166) # This is the argument for the number of training images.
    ap.add_argument("--val_subset",
                    help="Number of validation images to use. If not specified, all images are used.",
                    type = int, default=7500) # This is the argument for the number of validation images.
    ap.add_argument("--test_subset",
                    help="Number of test images to use. If not specified, all images are used.",
                    type = int, default=7500) # This is the argument for the number of test images.
    ap.add_argument("-po",
                    "--pooling_type",
                    help="Type of pooling to use. Either 'avg' or 'max'.",
                    type = str, default="avg") # This is the argument for the pooling type.
    ap.add_argument("-ilr",
                    "--initial_learning_rate",
                    help="Initial learning rate for the optimizer.",
                    type = float, default=0.01) # This is the argument for the initial learning rate.
    ap.add_argument("-ds",
                    "--decay_steps",
                    help="Decay steps for the learning rate.",
                    type = int, default=10000) # This is the argument for the decay steps.
    ap.add_argument("-dr",
                    "--decay_rate",
                    help="Decay rate for the learning rate.",
                    type = float, default=0.9) # This is the argument for the decay rate.
    ap.add_argument("-l",
                    "--loss",
                    help="Loss function to use.",
                    type = str, default="categorical_crossentropy") # This is the argument for the loss function.
    ap.add_argument("-e",
                    "--epochs",
                    help="Number of epochs to train for.",
                    type = int, default=10) # This is the argument for the number of epochs.
    args = ap.parse_args() # Parse the args
    return args

def import_and_preprocess_data():
    # Importing labels
    train_df = pd.read_json(os.path.join(os.getcwd(), "images", "metadata", "train_data.json"), lines=True)
    test_df = pd.read_json(os.path.join(os.getcwd(), "images", "metadata", "test_data.json"), lines=True)
    val_df = pd.read_json(os.path.join(os.getcwd(), "images", "metadata", "val_data.json"), lines=True)

    # Changing path names to absolute path
    def convert_image_path(image_path):
        base_dir = os.getcwd()
        return os.path.join(base_dir, image_path)

    train_df['image_path'] = train_df['image_path'].apply(convert_image_path)
    test_df['image_path'] = test_df['image_path'].apply(convert_image_path)
    val_df['image_path'] = val_df['image_path'].apply(convert_image_path)
    
    return train_df, test_df, val_df


# Set parameters for data loading and image processing

def setup_generators(train_df, test_df, val_df):
    # Parameters for loading data and images

    train_generator = ImageDataGenerator(horizontal_flip=True,
                                         rescale = 1./255,
                                         rotation_range=20
                                         )

    val_generator = ImageDataGenerator(horizontal_flip=True,
                                       rescale = 1./255,
                                       rotation_range=20
                                       )
    
    test_generator = ImageDataGenerator()
    
    return train_generator, val_generator, test_generator

def setup_data(train_df, test_df, val_df, train_generator, val_generator, test_generator, batch_size_arg, train_subset_arg, val_subset_arg, test_subset_arg):
    # Split the data into three categories.
    train_ds = train_generator.flow_from_dataframe(
        dataframe=train_df.sample(n = train_subset_arg, random_state=42),
        x_col='image_path',
        y_col='class_label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=True,
        seed=42
    )

    val_ds = val_generator.flow_from_dataframe(
        dataframe=val_df.sample(n = val_subset_arg, random_state=42),
        x_col='image_path',
        y_col='class_label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=True,
        seed=42
    )

    test_ds = test_generator.flow_from_dataframe(
        dataframe=test_df.sample(n = test_subset_arg, random_state=42),
        x_col='image_path',
        y_col='class_label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=False
    )
    
    return train_ds, val_ds, test_ds

def model_setup(pooling_type_arg, initial_learning_rate_arg, decay_steps_arg, decay_rate_arg, loss_arg):
    # load model without classifier layers
    model = EfficientNetB0(include_top=False, 
                pooling=pooling_type_arg,
                input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(15, 
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate_arg,
        decay_steps=decay_steps_arg,
        decay_rate=decay_rate_arg)

    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss=loss_arg,
                metrics=['accuracy'])
    
    return model

def train_model(model, train_ds, val_ds, epochs_arg, batch_size_arg):
    history = model.fit(train_ds,
                        validation_data = val_ds,
                        epochs=epochs_arg,
                        batch_size=batch_size_arg,
                        )
    
    return history

##### PLOTTING FUNCTION #####

def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), "out", "cnn_fashion.png"))

def make_predictions(model, test_ds, batch_size_arg):
    y_test = test_ds.classes

    y_pred = model.predict(test_ds,
                        batch_size = batch_size_arg,)

    y_pred = np.argmax(y_pred, axis=1)
    
    return y_test, y_pred

def print_report(y_test, y_pred, test_ds):
    # Get the classification report
    report = classification_report(y_test,
                                   y_pred,
                                   target_names = test_ds.class_indices.keys()
                                   )
    # Save the report
    with open(os.path.join(os.getcwd(), "out", "classification_report.txt"), "w") as f:
            f.write(report)
    # Print the report
    print(report)

def main():
    args = input_parser() # Parse the input arguments.
    print("Loading and preprocessing data...")
    train_df, test_df, val_df = import_and_preprocess_data()
    print("Setting up generators and data...")
    train_generator, val_generator, test_generator = setup_generators(train_df, test_df, val_df)
    print("Setting up data...")
    train_ds, val_ds, test_ds = setup_data(train_df, test_df, val_df, train_generator, val_generator, test_generator, args.batch_size, args.train_subset, args.val_subset, args.test_subset)
    print("Setting up model...")
    model = model_setup(args.pooling_type, args.initial_learning_rate, args.decay_steps, args.decay_rate, args.loss)
    print("Training model...")
    history = train_model(model, train_ds, val_ds, args.epochs, args.batch_size)
    print("Plotting and saving learning curves...")
    plot_history(history, args.epochs)
    print("Making predictions...")
    y_test, y_pred = make_predictions(model, test_ds, args.batch_size)
    print("Printing classification report...")
    print_report(y_test, y_pred, test_ds)

if __name__ == "__main__":
    main()