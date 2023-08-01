import numpy as np
import json
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model import GraphVAE

from tqdm import tqdm
import sys, os
import time
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
import torch.optim as optim
from FetterGrad import FetterGrad

"""Calculate the loss Using the GraphVAE loss function."""
def loss_f(x, x_hat, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean') # reconstruction loss using mean squared error (MSE)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # Kullback-Leibler divergence (KLD) loss
    beta = 0.001 # Regularization term Beta
    total_loss = recon_loss + beta * kld_loss # total loss is a weighted sum of the reconstruction and KLD losses
    return total_loss

"""Train the GraphVAE model using the specified data and hyperparameters."""
def train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS):
    model.train()
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for i, data in enumerate(t):
            optimizer.zero_grad() # Set the graident of model parameters to zero before each batch processing

            x = data.edge_index.to(device) # Move data to GPU if available
            batch = data.batch.to(device) #....
            adj_matrix = to_dense_adj(x.long(), batch) # Convert Sparse Adjacency to Dense Adjacency matrix
            prid, x_hat, mu, log_var = model(data.to(device)) # Move model and data to GPU
            # Loss Calculation by different Matrices
            loss = loss_f(adj_matrix, x_hat, mu, log_var) # Calculate the reconstruction loss of generated Drug
            mse_loss = mse_f(prid, data.y.view(-1, 1).float().to(device)) # Calculate mean squared error (MSE) loss for the predicted values Vs Actual Affinity
            train_ci = get_cindex(prid.cpu().detach().numpy(), data.y.view(-1, 1).float().cpu().detach().numpy()) # Calculate the concordance index (c-index) for the predicted values

            losses = [loss, mse_loss] # store two losses into one list
            optimizer.ft_backward(losses) # perform back propegation using Fetter Grad 
            optimizer.step() # Update the model parameter based on computed gradient during the backword pass

            t.set_postfix(Recon_loss=loss.item(), MSE=mse_loss.item(), Train_cindex=train_ci) # Update the progress bar for each iteration

    return model



def test(model, device, test_loader, dataset, FLAGS):
    """Test the GraphVAE model on the specified data and report the results.""" 
    print('Testing on {} samples...'.format(len(test_loader.dataset)))  # Print the number of samples in test data set during testing
    model.eval() # Set the model to evaluation mode

    # Initialize variables for storing true and predicted values, as well as the total loss
    total_true = torch.Tensor() # Empty tensor for all actual labels
    total_predict = torch.Tensor() # Empty tensor for all pridected labels
    total_loss = 0 

    if dataset == "kiba":         # Determine which threshold values to use based on the dataset
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  

    with torch.no_grad(): # Disable gradient calculation during testing
        for i, data in enumerate(tqdm(test_loader)): # Loop through each batch in the test loader        
            
            x = data.edge_index.to(device) # Move data to the specified device (GPU or CPU)
            batch = data.batch.to(device)
            adj_matrix = to_dense_adj(x.long(), batch) # Convert Sparse Adjacency to Dense Adjacency matrix
        
            prid, x_hat, mu, log_var = model(data.to(device)) # Run the forward pass through the model to generate predictions and reconstructed graphs

            loss = loss_f(adj_matrix, x_hat, mu, log_var) # Calculate the loss between the actual and reconstructed graphs  
  
            total_true = torch.cat((total_true, data.y.view(-1, 1).cpu()), 0) # Concatenate the true labels from the current batch to the existing 'total_true' tensor.
            total_predict = torch.cat((total_predict, prid.cpu()), 0) #......   
            G = total_true.numpy().flatten() # Flatten the true and predicted values for calculating MSE, c-index, RM2 and AUC
            P = total_predict.numpy().flatten() #.......
            mse_loss = mse(G, P) # Calculate the MSE for current batch of predictions
            test_ci = get_cindex(G, P) # Calculate the c-index for current batch of predictions      
            rm2 = get_rm2(G, P) # Calculate the R^2m for current batch of predictions    

            auc_values = [] # create empty list for AUC values.
            for t in thresholds:
                auc = get_aupr(np.int32(G > t), P) # Calculate the AUC values for each threshold
                auc_values.append(auc) # Store the all calculated AUC values for each threshold  
            # Add the loss from this batch to the running total
            total_loss += loss.item() * data.num_graphs # store and update the total loss by adding the contribution of the current batch's loss to the overall loss
            msg = f"Test Batch: loss={loss.item()}, MSE={mse_loss.item()}, Test c-index={test_ci}, AUCs={auc_values}"
            logging(msg, FLAGS) # Log the test results for this batch
    return total_loss, mse_loss, test_ci, rm2, auc_values, G, P, x_hat.cpu().numpy()

def experiment(FLAGS):
    logging('Starting program', FLAGS)
    # **Define dataset and CUDA device**
    datasets = ['davis', 'kiba'] # Define the list of Dataset

    dataset_idx = int(sys.argv[1]) #sys.argv[1] represents the first command-line argument after the script name.
    dataset = datasets[dataset_idx] # If dataset_idx is 0, 'davis' will be selected. If dataset_idx is 1, 'kiba' will be selected.
    device = torch.device("cuda:" + str(int(sys.argv[2])) if len(sys.argv) > 2 and torch.cuda.is_available() else "cpu") #the command-line arguments have at least three elements (sys.argv[2] exists) 0-3

    # Define hyperparameters

    BATCH_SIZE = 512
    LR = 0.0002
    NUM_EPOCHS = 1000

    # Print hyperparameters
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    msg = f"Dataset {dataset}, Device {device}, batch size {BATCH_SIZE}, learning rate {LR}, epochs {NUM_EPOCHS}"
    logging(msg, FLAGS)


    # **Load processed data**
    processed_data_file_train = f"data/processed/{dataset}_train.pt"
    processed_data_file_test = f"data/processed/{dataset}_test.pt"
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
    else:
        train_data = TestbedDataset(root="data", dataset=f"{dataset}_train")
        test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")

        # **Prepare PyTorch mini-batches**
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        # **Create model and optimizer**
        model = GraphVAE().to(device)
        optimizer = FetterGrad(optim.Adam(model.parameters(), lr=LR)) 
        mse_f = nn.MSELoss()

        # **Train model**
        best_mse = float('inf')  # Initialize with infinity or a large value
        for epoch in range(NUM_EPOCHS):
            model = train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS)

            if (epoch + 1) % 50 == 0:
                # **Test model**
                total_loss, mse_loss, test_ci, rm2, auc_values, G, P, x_hats = test(model, device, test_loader, dataset, FLAGS)
                filename = f"saved_models/graphvae_model_{dataset}.pth"
                if mse_loss < best_mse:
                    best_mse = mse_loss
                    torch.save(model.state_dict(), filename) # save_best_model(mse_loss, model, best_mse, filename)
                    print('model saved')

                # **Print results**
                print(f"Test Loss: {total_loss / len(test_data):.4f}")
                print(f"MSE: {mse_loss:.4f}")
                print(f"CI: {test_ci:.4f}")
                print(f"RM2: {rm2:.4f}")
                print(f"AUCs: {auc_values}")

        folder_path = "Affinities/"
        np.savetxt(folder_path + "estimated_labels_{}.txt".format(dataset), P)
        np.savetxt(folder_path + "true_labels_{}.txt".format(dataset), G)

        output_folder = "Generated_Adjacency_Matrix/"
        output_file = os.path.join(output_folder, "x_hats.npy")
        x_hats = np.array(x_hats)
        np.save(output_file, x_hats)
        logging('Program finished', FLAGS)


if __name__ == "__main__":

    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = 'dataset_{}'.format(int(time.time()))
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    
    if not os.path.exists('Affinities'):
            os.mkdir('Affinities')
    
    if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
    
    if not os.path.exists('Generated_Adjacency_Matrix'):
            os.makedirs('Generated_Adjacency_Matrix')

    experiment(FLAGS)