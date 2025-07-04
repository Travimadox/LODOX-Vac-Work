import numpy as np
import matplotlib.pyplot as plt
import pydicom
from n2v.models import N2V, N2VConfig
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import time
from csbdeep.models import CARE, Config
import warnings
import argparse
warnings.filterwarnings("ignore")

class LODOX():
    def __init__(self, model_path, model_name):
        self.model_path = model_path #Path to save the model
        self.model_name = model_name
        self.data_generator = N2V_DataGenerator()

    def load_model_n2n(self):
        return CARE(
            config = None,
            name = self.model_name,
            basedir = self.model_path,
        )

    def load_model_n2v(self):
        return N2V(
            config = None,
            name = self.model_name,
            basedir = self.model_path,
        )

    def load_image(self, image_path):
        image = pydicom.dcmread(image_path)
        image = image.pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    def generate_patches(self, image, patch_size):
        patch_shape = (patch_size, patch_size)
        patches = self.data_generator.generate_patches_from_list([image], shape=patch_shape)
        return patches

    def train_n2n(self, image1_path, image2_path, patch_size, epochs):
        # Load images first
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)

        if image1 is None or image2 is None:
            raise ValueError("Failed to load images")
        
        # Generate patches and split into training and validation sets
        patches1 = self.generate_patches(image1, patch_size)
        patches2 = self.generate_patches(image2, patch_size)
        
        if patches1 is None or patches2 is None:
            raise ValueError("Failed to generate patches from images")
            
        X = patches1[:int(len(patches1)*0.8)]
        X_val = patches1[int(len(patches1)*0.8):]
        Y = patches2[:int(len(patches2)*0.8)]
        Y_val = patches2[int(len(patches2)*0.8):]

        # Normalize the data
        mean = np.mean(X)
        std = np.std(X)
        X = (X - mean) / std
        X_val = (X_val - mean) / std
        Y = (Y - mean) / std
        Y_val = (Y_val - mean) / std

        # Create the model(you can update the model hyperparameters here)
        config = Config(
            axes = 'YX',
            n_channel_in = 1,
            n_channel_out = 1,
            train_loss = 'mse',
            train_epochs = epochs,
            train_steps_per_epoch = 10,
            
        )

        model = CARE(
            config,
            self.model_name,
            basedir=self.model_path,
        )

        # Train the model
        start_time = time.time()
        history = model.train(
            X,Y, #Training data
            (X_val,Y_val) #Validation data
        )
        end_time = time.time()
        training_time = end_time - start_time

        training_loss = history['loss']
        validation_loss = history['val_loss']
        training_mse = history['mse']
        validation_mse = history['val_mse']

        # Plot the training and validation loss
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.legend()
        plt.show()

        return training_loss, training_mse, validation_loss, validation_mse, training_time

    def train_n2v(self, image_path, patch_size, batch_size, epochs):
        # Load images first
        image = self.load_image(image_path)

        if image is None:
            raise ValueError("Failed to load image")

        # Generate patches and split into training and validation sets
        patches = self.generate_patches(image, patch_size)

        if patches is None:
            raise ValueError("Failed to generate patches from image")

        X = patches[:int(len(patches)*0.8)]
        X_val = patches[int(len(patches)*0.8):]

        # Create the model(you can update the model hyperparameters here)
        config = N2VConfig(
            X,
            unet_kern_size = 3,
            train_steps_per_epoch = len(X) // batch_size,
            train_epochs = epochs,
            train_loss ='mse',
            batch_norm = True,
            train_batch_size = batch_size,
            n2v_perc_pix = 0.15,
            n2v_patch_shape = (patch_size, patch_size),
            unet_n_first = 32
        )

        model = N2V(config=config, name=self.model_name, basedir=self.model_path)

        # Train the model
        start_time = time.time()
        history = model.train(
            X,
            X_val
        )
        end_time = time.time()
        training_time = end_time - start_time
        training_loss = history['loss']
        validation_loss = history['val_loss']
        training_mse = history['n2v_mse']
        validation_mse = history['val_n2v_mse']

        # Plot the training and validation loss
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.legend()
        plt.show()

        return training_loss, training_mse, validation_loss, validation_mse, training_time


def main(args):
    model_path = args.model_path
    model_name = args.model_name
    image1_path = args.image1_path
    image2_path = args.image2_path
    patch_size = args.patch_size
    batch_size = args.batch_size
    epochs = args.epochs

    if args.use_n2n:
        n2n_model = LODOX(model_path, model_name)
        n2n_model.train_n2n(image1_path, image2_path, patch_size, epochs)
    elif args.use_n2v:
        n2v_model = LODOX(model_path, model_name)
        n2v_model.train_n2v(image1_path, patch_size, batch_size, epochs)
    else:
        raise ValueError("Please specify which model to use")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_path', type=str, default='models', help='Path to save the model')
    parser.add_argument('--model_name', type=str, default='n2n_model', help='Name of the model')
    parser.add_argument('--image1_path', type=str, default='DICOM_Images\Palm down 50kV.DCM', help='Path to the first image')
    parser.add_argument('--image2_path', type=str, default='DICOM_Images\Palm down 70kV.DCM', help='Path to the second image')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of the patches')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use_n2n', type=bool, default=False, help='Use n2n model')
    parser.add_argument('--use_n2v', type=bool, default=False, help='Use n2v model')
    args = parser.parse_args()
    main(args)
    
