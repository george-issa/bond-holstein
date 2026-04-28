import numpy as np
import csv
import random as rn
import os

import tensorflow # type: ignore
from tensorflow import keras # type: ignore

from tensorflow import random # type: ignore
from keras import layers, regularizers, utils # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # type: ignore

# Special ID for this run
sID = 1

alphas = [0.240, 0.245, 0.250, 0.255]

# Numer of lambdas to skip. 
# Half of them before and half fof them after the guessed critical lambda
# Has to be an even number
nskips = [4, 6, 8]
for a in alphas:

    for nskip in nskips:
        
        f = open(f'augmented_BondBond012345_momentum_L12_w0.5_m0.0_a{a:.4f}-{sID}.csv', 'r')

        out_dir = f'out_BondBond012345_momentum_L12_w0.5_m0.0_a{a:.4f}_nskip{nskip}-{sID}'
        
        os.makedirs(out_dir, exist_ok=True)

        data = []
        csv_reader = csv.reader(f)
        # To skip the header row, uncomment the line below:
        #next(csv_reader)
        for row in csv_reader:
                data.append(row)
        data = np.array(data, dtype=float)

        # Creating the matrix of data
        L = 12
        Size = L*L

        # Sweep parameter and an array of its unique values
        betas = data[:,-1]
        u_beta = np.unique(betas)
        print(f"\nUnique lambdas in the dataset: {u_beta}\n")
        NL = len(u_beta)
        print(f"\nNumber of distinct temperatures in the dataset: {NL}\n")

        # Number of lambdas on each side
        half = nskip // 2

        # Number of guessed parameters
        NG = NL - nskip

        # Number of data points per pair of temperatures
        # involved in the training
        N = nskip * len(data) // NL # 6 * (31,000 // 31) = 6,000

        # Truncated unique lambdas that will be guessed as critical points
        trunc_beta = u_beta[nskip//2:-nskip//2]
        print(f"\n betas after truncating:{trunc_beta}\n")

        # Initializing arrays for the average accuracies of the trained network
        ntry = 1
        nepochs = 100

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]

        test_losses = []
        test_accuracies = []

        # Seed 
        # nt = 10

        # keras.utils.set_random_seed(41*nt)
        He = keras.initializers.HeNormal()

        # Loop over guessed critical betas
        for iL in range(NG):
            
            print(f"\n Training for beta index iL = {iL} / {NG-1} (beta_c = {trunc_beta[iL]:.4f})\n")

            # Lambda indices for this guessed critical point
            beta_left  = u_beta[iL : iL + half]
            beta_right = u_beta[iL + half + 1 : iL + nskip + 1]

            # Masks (order-independent)
            mask_left  = np.isin(betas, beta_left)
            mask_right = np.isin(betas, beta_right)

            X0 = data[mask_left, :Size]   # class 0
            X1 = data[mask_right, :Size]  # class 1

            # Balance classes (important!)
            n_min = min(len(X0), len(X1))
            X0 = X0[:n_min]
            X1 = X1[:n_min]

            # Stack
            X = np.vstack([X0, X1])

            # Labels
            y = np.concatenate([
                np.zeros(len(X0), dtype=int),
                np.ones(len(X1), dtype=int)
            ])

            # Shuffle
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]

            # One-hot
            Y = keras.utils.to_categorical(y, num_classes=2)
            
            Ntot = len(X)

            X_train = X[:int(0.8*Ntot)].reshape(-1, L, L, 1)
            X_val   = X[int(0.8*Ntot):int(0.9*Ntot)].reshape(-1, L, L, 1)
            X_test  = X[int(0.9*Ntot):].reshape(-1, L, L, 1)

            Y_train = Y[:int(0.8*Ntot)]
            Y_val   = Y[int(0.8*Ntot):int(0.9*Ntot)]
            Y_test  = Y[int(0.9*Ntot):]

            # Build a FRESH model for each iL (important!)
            model = Sequential([
                
                keras.Input(shape=(L,L,1)),
                
                Conv2D(8, (3,3), padding="same", activation=None,
                        kernel_initializer=He),
                
                # layers.BatchNormalization(),
                layers.Activation('relu'),
                MaxPooling2D((2,2)),
                layers.Dropout(0.25),
                
                Conv2D(16, (3,3), padding="same", activation=None,
                    kernel_initializer=He),
                    
                # layers.BatchNormalization(),
                layers.Activation("relu"),
                MaxPooling2D((2,2)),
                layers.Dropout(0.25),
                
                Flatten(),
                
                Dense(64, activation=None,
                        kernel_initializer=He),
                
                # layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.5),
                
                Dense(2, activation='softmax'),
            ])

            loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            opt = keras.optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4)
            model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=nepochs,
                batch_size=32,
                verbose=1,
                callbacks=callbacks
            )
            
            train_accs = history.history['accuracy']
            val_accs = history.history['val_accuracy']
            train_losses = history.history['loss']
            val_losses = history.history['val_loss']
            np.savetxt(f'{out_dir}/train_accs_iL{iL}-{sID}.csv', np.array(train_accs), fmt='%.5f')
            np.savetxt(f'{out_dir}/val_accs_iL{iL}-{sID}.csv', np.array(val_accs), fmt='%.5f')
            np.savetxt(f'{out_dir}/train_losses_iL{iL}-{sID}.csv', np.array(train_losses), fmt='%.5f')
            np.savetxt(f'{out_dir}/val_losses_iL{iL}-{sID}.csv', np.array(val_losses), fmt='%.5f')
            
            test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for beta index iL = {iL}: {test_accuracy:.5f}")
            print(f"Test loss for beta index iL = {iL}: {test_loss:.5f}")
            np.savetxt(f'{out_dir}/test_accuracies-{sID}.csv', np.array(test_accuracies), fmt='%.5f')
            np.savetxt(f'{out_dir}/test_losses-{sID}.csv', np.array(test_losses), fmt='%.5f')
            
            np.savetxt(f'{out_dir}/betas-{sID}.csv', u_beta, fmt='%.5f')
            np.savetxt(f'{out_dir}/trunc_betas-{sID}.csv', trunc_beta, fmt='%.5f')