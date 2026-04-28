# Import libraries
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the image augmentation method
def image_augmentation(data, L):
        
    hflip_configs = []
    vflip_configs = []
    hvflip_configs = []
    vhflip_configs = []
    
    for i in range(data.shape[0]):

        config = data[i, :-1]
        temp = data[i, -1]

        lattice = np.reshape(config, (L, L))

        hflip = np.ndarray.flatten(np.flip(lattice, 0))
        hflip = np.ndarray.tolist(hflip)
        hflip.append(temp)
        hflip_configs.append(hflip)

        vflip = np.ndarray.flatten(np.flip(lattice, 1))
        vflip = np.ndarray.tolist(vflip)
        vflip.append(temp)
        vflip_configs.append(vflip)

        hvflip = np.ndarray.flatten(np.flip(lattice, (0, 1)))
        hvflip = np.ndarray.tolist(hvflip)
        hvflip.append(temp)
        hvflip_configs.append(hvflip)
        
        vhflip = np.ndarray.flatten(np.flip(lattice, (1, 0)))
        vhflip = np.ndarray.tolist(vhflip)
        vhflip.append(temp)
        vhflip_configs.append(vhflip)

    hflip_configs = np.array(hflip_configs)
    vflip_configs = np.array(vflip_configs)
    hvflip_configs = np.array(hvflip_configs)
    vhflip_configs = np.array(vhflip_configs)

    augmented_data = np.vstack((data, hflip_configs, vflip_configs, hvflip_configs, vhflip_configs))    # stack augmented configurations
    augmented_data = augmented_data[augmented_data[:, -1].argsort()]                    # sort by lambda

    return augmented_data

def _radial_stats(map2d, center=None, nbins=None, smooth_sigma=None, eps=1e-12):
    """
    Compute the radial average S̄(|q|) for a 2D array (e.g., structure factor).
    Returns:
      radial_map: same shape as map2d, with each pixel set to the mean over its radius bin
      edges: bin edges used (pixels)
    """
    H, W = map2d.shape
    if center is None:
        # center = ((H - 1) / 2.0, (W - 1) / 2.0)  # (cy, cx) — works for fftshifted maps
        center = (0, 0) 
    cy, cx = center

    yy, xx = np.indices((H, W), dtype=np.float64)
    rr = np.hypot(xx - cx, yy - cy)  # pixel radii

    if nbins is None:
        nbins = int(np.ceil(np.hypot(H, W)))  # fine bins by default

    # Bin radii
    r_min, r_max = 0.0, rr.max() + 1e-9
    edges = np.linspace(r_min, r_max, nbins + 1)
    bins = np.digitize(rr.ravel(), edges) - 1  # 0..nbins-1

    # Radial mean via bincount
    vals = map2d.ravel()
    counts = np.bincount(bins, minlength=nbins)
    sums   = np.bincount(bins, weights=vals, minlength=nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        radial_mean = sums / np.maximum(counts, 1)

    # Optional 1D smoothing along radius (helps denoise the background estimate)
    if smooth_sigma is not None and smooth_sigma > 0:
        radial_mean = gaussian_filter1d(radial_mean, smooth_sigma, mode='nearest')

    # Paint back to image
    radial_map = radial_mean[np.clip(bins, 0, nbins - 1)].reshape(H, W)

    # Guard against zeros for division whitening
    radial_map = np.where(radial_map == 0, eps, radial_map)
    return radial_map, edges

def whiten_structure_factor(map2d,
                            mode="divide",   # "divide" or "subtract"
                            center=None,
                            nbins=None,
                            smooth_sigma=2.0,
                            eps=1e-12):
    """
    Physics-driven whitening:
      - Compute radial average S̄(|q|)
      - Return whitened map:
            divide:    S / S̄ - 1
            subtract:  S - S̄
    """
    radial_map, _ = _radial_stats(map2d, center=center, nbins=nbins,
                                  smooth_sigma=smooth_sigma, eps=eps)

    if mode.lower() == "divide":
        return (map2d / radial_map) - 1.0, radial_map
    elif mode.lower() == "subtract":
        return map2d - radial_map, radial_map
    else:
        raise ValueError("mode must be 'divide' or 'subtract'")

# Define the neural network model
class CNN(nn.Module):
    
    def __init__(self, num_classes, L, dropout_prob):
        
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=32),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # # nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout_prob),
            
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d((L, L))
            
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout_prob),
            
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=128),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d((L, L))  # Adaptive pooling to ensure the output size is LxL
            
            )
        
        self.classifier = nn.Sequential(
            
            nn.Linear((L // 2) * (L // 2) * 16, 256), # changed from 512 to 256 after maxpooling since CNN output is (512, 512)
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
            
        )
        
    def forward(self, x):
        
        B = x.shape[0]
        
        out = self.features(x)
        out = out.view(B, -1)
        out = self.classifier(out)
        
        return out

# Define the training function
def train_model(w, a, L, seed, sID,
                tc_folder, output_folder, 
                model, criterion, optimizer, scheduler, 
                train_loader, val_loader, num_epochs, stopping_criterion):
    
    """
    This function trains the model defined in the class CNN.
    
    Parameters:
        - model: the neural network model
        - criterion: the loss function
        - optimizer: the optimization algorithm to update the weights of the model
        - scheduler: the learning rate scheduler to reduce the learning rate when the validation loss stops decreasing after (patience) epochs, given in the function run_simulation
        - train_loader: the DataLoader object for the training data to loop over the training data
        - val_loader: the DataLoader object for the validation data to loop over the validation data
        - num_epochs: the maximum number of epochs to train the model. The training stops early if the stopping criterion is met
        - stopping_criterion: the number of epochs without improvement in the validation loss to stop the training early
        
    Returns:
        - train_losses: the training losses of maximum length num_epochs
        - val_losses: the validation losses of maximum length num_epochs
        - train_accuracies: the training accuracies of maximum length num_epochs
        - val_accuracies: the validation accuracies of maximum length num_epochs
        
    """
    
    # Initialize the best validation loss
    best_val_loss = np.inf
    
    # Initialize the number of epochs without improvement
    # This is used for the early stopping of the training
    epochs_no_improve = 0
    
    # Initialize the training and validation losses of maximum length num_epochs
    # Later, we will only keep the losses up to the epoch where the training stops, which happens when the validation loss stops decreasing
    train_losses = []
    val_losses = []
    
    # Initialize the training and validation accuracies
    train_accuracies = []
    val_accuracies = []
    
    # Create a state folder to store the model for each epoch
    state_folder = f'CNN_states'
    os.makedirs(f'{output_folder}/{tc_folder}/{state_folder}', exist_ok=True)
    
    # Loop over the number of epochs
    for epoch in range(num_epochs):
        
        # Save the model for each epoch for future reference
        torch.save(
            model.state_dict(), f'{output_folder}/{tc_folder}/{state_folder}/CNN_state_w{w:.2f}_a{a:.4f}_L{L}_seed{seed}_epoch{epoch + 1}-{sID}.pt')
        
        # Print the Epoch number
        print('-' * 75)
        print(f'--- Epoch [{epoch + 1} / {num_epochs}] ---')
        print('-' * 75 + '\n')
        
        # Initialize the training loss
        train_loss = 0.0
        
        # Initialize the number of correct predictions in the training data
        train_accuracy = 0
        
        # Set the model to training mode
        model.train()
        
        print()
        
        # Loop over the training data
        for i, (X_batch, y_batch) in enumerate(train_loader):
            
            # Forward pass
            outputs = model(X_batch)[:, 0]
            
            # Compute the loss
            loss = criterion(outputs, y_batch)
            
            # print(f'Batch {i + 1} | Loss: {loss.item()}')
            
            # Backward pass
            loss.backward()
            
            # Optimize the weights
            optimizer.step()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Update the training loss by the sum of losses of all batches
            train_loss += loss.item() * X_batch.shape[0]
            
            # Compute the accuracy
            is_correct = ((outputs >= 0.5).float() == y_batch).float()
            train_accuracy += is_correct.sum()
            
            # print(f'Batch {i + 1} | Accuracy: {is_correct.sum() / X_batch.shape[0]}')
        
        # Compute the average training loss
        train_loss /= len(train_loader.dataset)
        
        # Update the training losses
        train_losses.append(train_loss)
        
        # Compute the average training accuracy
        train_accuracy /= len(train_loader.dataset)
        
        # Update the training accuracies
        train_accuracies.append(train_accuracy)
        
        
        # Set the model to evaluation mode
        model.eval()
        
        # Initialize the validation loss
        val_loss = 0.0
        
        # Initialize the number of correct predictions in the validation data
        val_accuracy = 0
        
        # Loop over the validation data
        with torch.no_grad():
            
            for X_batch, y_batch in val_loader:
                
                # Forward pass
                outputs = model(X_batch)[:, 0]
                
                # Compute the loss
                loss = criterion(outputs, y_batch)
                
                # Update the validation loss
                val_loss += loss.item() * X_batch.shape[0]
                
                # Compute the accuracy
                is_correct = ((outputs >= 0.5).float() == y_batch).float()
                val_accuracy += is_correct.sum()
        
        # Compute the average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Update the validation losses
        val_losses.append(val_loss)
        
        # Compute the average validation accuracy
        val_accuracy /= len(val_loader.dataset)
        
        # Update the validation accuracies
        val_accuracies.append(val_accuracy)
        
        # Print the training and validation losses
        print(f'Train Loss: {train_loss:.8f}       |        Train Accuracy: {train_accuracy:.8f}')
        
        # Print the training and validation accuracies
        print(f'Val Loss:   {val_loss:.8f}       |        Val Accuracy:   {val_accuracy:.8f}\n')
        
        # Check if the validation loss improved
        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save the best model
            torch.save(
                model.state_dict(), f'{output_folder}/{tc_folder}/best_CNN_w{w:.2f}_a{a:.4f}_L{L}-{sID}.pt')
            
            print(f'\n ✰✰✰ New best model saved ✰✰✰ \n')
            
        else:
            
            epochs_no_improve += 1
        
        # Check if early stopping criterion is met
        if epochs_no_improve == stopping_criterion:
            
            print(f'\n☑︎ Early stopping at epoch {epoch + 1} with best validation loss: {best_val_loss:.8f}')
            print(f' ☞ Losses and Accuracies up to epoch {epoch + 1} will be returned \n')
            
            # Stop the training
            break
        
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        # Step the scheduler based on a plateau in the validation loss
        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]['lr']
        
        # Print the new learning rate if it changes
        if current_lr != next_lr:
            print(f'Learning rate changed from {current_lr} -> {next_lr}')
        
    return train_losses, val_losses, train_accuracies, val_accuracies # Could return model too

# Define the simulation as a function so we can run it later
# We assume a sweep in b here
def CNN_Bond_Holstein_simulation(
    w, a, L, seed, sID, t_min, t_max, num_t, CSV_path, input_data, output_folder, 
    num_classes=1, batch_size=32, num_epochs=100, dropout_prob=0.1,
    initial_lr=1E-02, weight_decay=1E-03, patience=10, stopping_criterion=20): 
    
    # Print the simulation parameters
    print(f'\n--- Parameters: w = {w:.1f} | a = {a:.4f} | L = {L} | seed = {seed} | sID = {sID} ---\n')
    
    # Create a folder to store all output files
    os.makedirs(output_folder, exist_ok=True)
    
    # Print the PyTorch version
    print(f'↪︎ PyTorch version: {torch.__version__} \n')
    
    # Load the data from the CSV file
    data = np.loadtxt(input_data, delimiter=',')
    
    # Uncomment for normalizing data
    # normalized_data = data.copy()
    
    # # Make the first data point zero
    # normalized_data = make_first_zero(normalized_data)
    
    # # Divide the data by the max value
    # normalized_data = divide_by_max(normalized_data)
    
    # # Augment the data
    augmented_data = image_augmentation(data, L)
    
    # Uncomment for whitening data
    # whitened_data = data.copy()
    
    # X = data[:, :-1]
    # T = data[:, -1]
    
    # # Whiten data
    # for i in range(data.shape[0]):
        
    #     lattice_whitened, radial_map = whiten_structure_factor(X[i].reshape(L, L), mode="divide", smooth_sigma=2.0)
    #     X[i] = lattice_whitened.flatten()
        
    # whitened_data[:, :-1] = X
    
    # Save the augmented data
    np.savetxt(f'{output_folder}/augmented_data_w{w:.2f}_a{a:.4f}_L{L}-{sID}.csv', augmented_data, delimiter=',')
    # np.savetxt(f'{output_folder}/normalized_data_w{w:.2f}_a{a:.4f}_L{L}-{sID}.csv', normalized_data, delimiter=',')
    # np.savetxt(f'{output_folder}/whitened_data_w{w:.2f}_a{a:.4f}_L{L}-{sID}.csv', whitened_data, delimiter=',')
    # np.savetxt(f'{output_folder}/data_w{w:.2f}_a{a:.4f}_L{L}-{sID}.csv', data, delimiter=',')

    # Define the samples and the labels
    # X = data[:, :-1]
    # T = data[:, -1]
    # X = normalized_data[:, :-1]
    # T = normalized_data[:, -1]
    X = augmented_data[:, :-1]
    T = augmented_data[:, -1]

    # Print the shape of the resulting data and the interval of the labels
    print(f'➫ X shape: {X.shape}, T shape: {T.shape}')
    print(f'➫ T interval: [{min(T):.3f}, {max(T):.3f}]\n')
    
    # Initialize the array storing the number of correct predictions in the testing data
    test_accuracies = []
    
    # Create t_array
    t_array = np.linspace(t_min, t_max, num_t)
    
    # Loop over each tc and label the data accordingly
    for tc in t_array:
        
        # Create a folder to store all outputs for tc
        tc_folder = f'tc{tc:.4f}'
        os.makedirs(f'{output_folder}/{tc_folder}', exist_ok=True)
        
        # Split the data into training and testing sets
        X_train, X_test, t_train, t_test = train_test_split(X, T, test_size=0.1, random_state=1, shuffle=True)
        
        # Choose some of the training data to be used for validation
        X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.15, random_state=1, shuffle=True)
        
        # Convert the data to 4D tensors
        X_train = X_train.reshape(X_train.shape[0], 1, L, L)
        X_val = X_val.reshape(X_val.shape[0], 1, L, L)
        X_test = X_test.reshape(X_test.shape[0], 1, L, L)
        
        # # Reshape the data to be compatible with the CNN
        # X_train = X_train.reshape(X_train.shape[0], 1, L * L)
        # X_val = X_val.reshape(X_val.shape[0], 1, L * L)
        # X_test = X_test.reshape(X_test.shape[0], 1, L * L)
        
        # Print the shape of the resulting data
        print(f' ↳ X_train shape: {X_train.shape}, t_train shape: {t_train.shape}')
        print(f' ↳ X_val shape: {X_val.shape}, t_val shape: {t_val.shape}')
        print(f' ↳ X_test shape: {X_test.shape}, t_test shape: {t_test.shape} \n')
        
        print(f'  ➤ Now labeling data according to tc = {tc}')
        
        # Make the labels 0s and 1s and call them y
        # True if t < tc, False if t >= tc
        if tc != t_max:
            y_train = (t_train < tc) 
            y_val = (t_val < tc)
            y_test = (t_test < tc)
        
        # True if t <= tc
        else:
            y_train = (t_train <= tc) 
            y_val = (t_val <= tc)
            y_test = (t_test <= tc)

        # Convert the data to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        # Create a DataLoader object for the training data
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # Create a DataLoader object for the valdation data
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        # Load the neural network model with a set random seed
        torch.manual_seed(seed)
        model = CNN(num_classes, L, dropout_prob)

        # Define the loss function and optimizer
        criterion = nn.BCELoss() # Could try nn.CrossEntropyLoss() instead
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        
        # Shceduler reduces the learning rate by (factor) when the validation loss stops decreasing (min mode) after (patience) epochs.
        # Scheduler is linked with the stopping criterion. After (stopping criterion) epochs without improvement, 
        # The lr would have already decreased at least once. The training then stops early.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=patience)

        # Train the model with a maximum of (num_epochs) epochs and stop training after (stopping_criterion) epochs without improvement
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(w, a, L, seed, sID,
                                                                                 tc_folder, output_folder,
                                                                                 model, criterion, optimizer, scheduler, train_loader, val_loader,
                                                                                 num_epochs, stopping_criterion)
        
        # Save the losses and accuracies
        np.save(f'{output_folder}/{tc_folder}/train_losses_ac{tc:.4f}_w{w:.2f}_a{a:.4f}_L{L}-{sID}.npy', train_losses)
        np.save(f'{output_folder}/{tc_folder}/val_losses_ac{tc:.4f}_w{w:.2f}_a{a:.4f}_L{L}-{sID}.npy', val_losses)
        np.save(f'{output_folder}/{tc_folder}/train_accuracies_ac{tc:.4f}_w{w:.2f}_a{a:.4f}_L{L}-{sID}.npy', train_accuracies)
        np.save(f'{output_folder}/{tc_folder}/val_accuracies_ac{tc:.4f}_w{w:.2f}_a{a:.4f}_L{L}-{sID}.npy', val_accuracies)
        
        # Load the best model to test it
        model.load_state_dict(
            torch.load(
                f'{output_folder}/{tc_folder}/best_CNN_w{w:.2f}_a{a:.4f}_L{L}-{sID}.pt', weights_only=True))
        
        # Set the model to evaluation mode
        model.eval()
        
        # Compute the outputs of the testing data directly without looping over the DataLoader
        outputs = model(X_test)[:, 0]
        
        # Compute the test accuracy
        is_correct = ((outputs >= 0.5).float() == y_test).float()
        test_accuracy = (is_correct.sum() / is_correct.shape[0]).item()
        
        # Add the value to the list
        test_accuracies.append(test_accuracy)
        
        # Compute the test loss
        test_loss = criterion(outputs, y_test).item()
        
        # Print the test accuracy and loss
        print('-' * 50)
        print('-' * 11 + f' Test Accuracy: {test_accuracy:.8f} ' + '-' * 11)
        print('-' * 11 + f' Test Loss:     {test_loss:.8f} ' + '-' * 11)
        print('-' * 50 + '\n')
        
    # Save the test accuracy
    np.savetxt(f'{output_folder}/accs_w{w:.2f}_a{a:.4f}_L{L}_seed{seed}-{sID}.csv', test_accuracies)
 
    
if __name__ == "__main__":        
 
    # Run the simulation with the following parameters
    
    # ML parameters:
    num_classes = 1;    batch_size = 32;    num_epochs = 150;   initial_lr = 1E-03;     weight_decay = 1E-02;   dropout_prob = 0.2

    # Patience is the number of epochs without improvement to change the learning rate
    # Stopping criterion is the number of epochs without improvement to stop the training

    patience = 10;      stopping_criterion = 20
    
    # Special ID for this simulation
    sID = 1

    # Bandwidth
    W = 8

    # Phonon frequency
    w = 1.0

    # Inverse temperature
    β_array = np.linspace(0.5, 3.0, 26)

    # e-ph coupling constant
    λ_inv = 5

    λ = 1 / λ_inv

    λ = 2 * λ # factor of 2 for Charles' definition of lambda
    a = np.sqrt(((w ** 2) * W * λ) / 8)

    # Lattice size
    L = 12
    
    # Seed for random number generator
    seeds = [2000]
        
    b_min = np.min(β_array);   b_max = np.max(β_array);   num_b = len(β_array)

    # Define the main path where the data is stored
    CSV_path = '/nfs/home/gissa/Bond_Holstein/LBC'
    input_data = f'{CSV_path}/electron_densities_w{w:.2f}_a{a:.4f}_L{L}_csv-{sID}/electron_densities_w{w:.2f}_a{a:.4f}_L{L}-{sID}.csv'

    for seed in seeds:
        
        output_folder = f'Bond_Holstein_LBC_w{w:.2f}_a{a:.4f}_L{L}_seed{seed}-{sID}'
        
        CNN_Bond_Holstein_simulation(w=w, a=a, L=L, seed=seed, sID=sID, t_min=b_min, t_max=b_max, num_t=num_b,
        CSV_path=CSV_path, input_data=input_data, output_folder=output_folder, 
        num_classes=num_classes, batch_size=batch_size, num_epochs=num_epochs, dropout_prob=dropout_prob,
        initial_lr=initial_lr, weight_decay=weight_decay, patience=patience, stopping_criterion=stopping_criterion)