import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import math
import nibabel as nib
from scipy import ndimage

class SaveBestModel(keras.callbacks.Callback):
    def __init__(self, path_model_save):
        super(SaveBestModel, self).__init__()
        self.path_model_save = path_model_save
        self.best_val_acc = -float('inf')
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        val_loss = logs.get('val_loss')

        if val_acc is None or val_loss is None:
            return

        if val_acc > self.best_val_acc or (val_acc == self.best_val_acc and val_loss < self.best_val_loss):
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.model.save(self.path_model_save)
            print(f"\nSaved model with val_acc={val_acc:.4f} and val_loss={val_loss:.4f}")
            
class CNN_class:
    def __init__(self, initial_learning_rate_model = 0.0001, epochs_model = 100, batch_size_model = 2,
                 decay_steps_model = 100000, decay_rate_model = 0.80):
        self.initial_learning_rate = initial_learning_rate_model
        self.epochs = epochs_model
        self.batch_size = batch_size_model
        self.decay_steps = decay_steps_model
        self.decay_rate = decay_rate_model

        self.tran_data = [None, None]
        self.val_data = [None, None]
        self.test_data = [None, None, None]

        self.width_g = None
        self.height_g = None
        self.depth_g = None
    def train_model(self, path_model_save):
        # def valid(X_test, y_test, model, criterion):

        x_train, y_train = self.tran_data
        x_val, y_val = self.val_data

        print(
            "Number of samples in train, validation are %d and %d."
            % (x_train.shape[0], x_val.shape[0])
        )
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        train_dataset = (
            train_loader.shuffle(len(x_train))
            .batch(self.batch_size)
            .prefetch(2)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .batch(self.batch_size)
            .prefetch(2)
        )

        # Build model.
        model = self.get_model(width=self.width_g, height=self.height_g, depth=self.depth_g)
        # model = get_model(width=193, height=229, depth=193)
        model.summary()

        ##*************************************************************************************
        # Compile model.

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            staircase=True
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["acc"],
        )

        # Define callbacks.
       # checkpoint_cb = keras.callbacks.ModelCheckpoint(
        #    path_model_save, save_best_only=True, monitor="val_acc", verbose=1
        #)
        checkpoint_cb = SaveBestModel(path_model_save)
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=40)

        history=model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.epochs,
            shuffle=True,
            verbose=1,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )
        return pd.DataFrame(history.history)

    def test_model(self, path_model):
        x_test, y_test, ID = self.test_data
        #x_test = np.array([self.process_scan(path) for path in x_test_paths])
        #criterion = tf.keras.losses.CategoricalCrossentropy()
        model = self.get_model(width=self.width_g, height=self.height_g, depth=self.depth_g)
        model.load_weights(path_model)
        results=[]
        real_results=[]
        predicted_results=[]
        for y, x, i in zip(y_test, x_test,ID):
            prediction = model.predict(np.expand_dims(x, axis=0))[0]
            # print(model.predict(np.expand_dims(x, axis=0)))
            # scores = [1 - prediction[0], prediction[0]]
            real_results.append(y)
            if (prediction[0] > 0.5):
                results.append([i,prediction[0],1,y])
                predicted_results.append(1)
            else:
                results.append([i,prediction[0],0,y])
                predicted_results.append(0)

        results_valid = (self.valid(real_results, predicted_results))

        headers = ["True_negative", "False_positive", "False_negative", "True_positive", "Accuracy_score", "Precision_score", "Sensitivity_score", "F1_score",
                  "Matthews_corrcoef", "Specificity"]

        results_valid=pd.DataFrame([results_valid])

        results_valid.columns = headers
        # acc = accuracy_score(targets.numpy(), predicted.numpy())
        # bacc = balanced_accuracy_score(targets.numpy(), predicted.numpy())
        # prec = precision_score(targets.numpy(), predicted.numpy())
        # rec = recall_score(targets.numpy(), predicted.numpy())
        # f1 = f1_score(targets.numpy(), predicted.numpy())
        # mc = matthews_corrcoef(targets.numpy(), predicted.numpy())
        #True positive
        #False positive
        #True negative
        #False negative
        headers = ["ID","Prediction%","Prediction","Label"]
        results=pd.DataFrame(results)
        results.columns = headers
        return results_valid,results
    def valid(self,targets_test, predicted_test):
        # print(X_test)
        # print(type(X_test))
        targets = np.array(targets_test)  # =np.float_)#torch.from_numpy(X_test).float()
        predicted = np.array(predicted_test)  # ,dtype=np.float_)#torch.from_numpy(y_test).long()
        # outputs = inputs #model(inputs)
        #predicted = inputs
        # loss = criterion(outputs, targets)
        # _, predicted = torch.max(outputs, 1)
        # cm = confusion_matrix(targets.numpy(), predicted.numpy())
        #print(targets)
        #print(type(targets))
        #print(predicted)
        #print(type(predicted))

        cm = confusion_matrix(targets, predicted)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        # acc = accuracy_score(targets.numpy(), predicted.numpy())
        # bacc = balanced_accuracy_score(targets.numpy(), predicted.numpy())
        # prec = precision_score(targets.numpy(), predicted.numpy())
        # rec = recall_score(targets.numpy(), predicted.numpy())
        # f1 = f1_score(targets.numpy(), predicted.numpy())
        # mc = matthews_corrcoef(targets.numpy(), predicted.numpy())
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = (tp + tn) / (tp + fp + fn + tn)
            #bacc = (tp / (tp + fn) + tn / (fp + tn)) / 2
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * tp / (2 * tp + fn + fp)
            mc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            spe = tn / (tn + fp)#specificity
            if (math.isnan(mc)):
                mc = 0
            if (math.isnan(prec)):
                prec = 0
        # return loss.item(), tn, fp, fn, tp, acc, bacc, prec, rec, f1, mc
        return [tn, fp, fn, tp, acc, prec, rec, f1, mc, spe]



    def normalize(self, volume):
        """Normalize the volume"""
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def resize_volume(self,img):
        """Resize across z-axis"""

        for indexi in range(len(img)):
            for indexj in range(len(img[0])):
                for indexk in range(len(img[0][0])):
                    if img[indexi,indexj,indexk]< 0.50:
                        img[indexi, indexj, indexk]=0

        if (self.width_g == None):
            self.width_g = img.shape[0]
            self.height_g = img.shape[1]
            self.depth_g = img.shape[-1]
        # Rotate
        img = ndimage.rotate(img, 90, reshape=False)

        return img

    def read_nifti_file(self,filepath):
        """Read and load volume"""
        # Read file
        for try_read in range(0,10):
            try:
                scan = nib.load(filepath)
                break
            except:
                print("Bad file descriptor(warning)"+filepath)
                print("Try to read " + filepath+" for "+str(try_read+1)+" time!")
                #scan = nib.load(filepath)

        #scan = nib.load(filepath)
        # Get raw data
        scan = scan.get_fdata()
        return scan
    def process_scan(self,path):
        """Read and resize volume"""
        # Read scan
        volume = self.read_nifti_file(path)
        # Normalize
        # volume = normalize(volume)
        # Resize width, height and depth
        volume = self.resize_volume(volume)
        return volume
    def load_MRI_files(self, train_mri_paths, evaluation_mri_paths, test_mri_paths):
        x_train_paths, y_train = train_mri_paths
        x_val_paths, y_val = evaluation_mri_paths
        x_test_paths, y_test, ID = test_mri_paths

        x_train = np.array([self.process_scan(path) for path in x_train_paths])
        x_val = np.array([self.process_scan(path) for path in x_val_paths])
        x_test = np.array([self.process_scan(path) for path in x_test_paths])

        self.tran_data = [x_train, y_train]
        self.val_data = [x_val, y_val]
        self.test_data = [x_test, y_test, ID]


    def get_model(self,width, height, depth):
        """Build a 3D convolutional neural network model."""

        inputs = keras.Input((width, height, depth, 1))

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn")
        return model

