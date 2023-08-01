import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch

# Define the graph encoder

'''
The Drug Encoder, denoted as q(Z|X,A), is designed to process 
graph data represented by node feature vectors X and adjacency 
matrix A. The input data is organized in mini-batchesof size 
[batch_size, Drug_features], where each drug is characterized 
by its feature vector.The goal of the Drug Encoder is to transform 
this high-dimensional input into a lower-dimensional representation.

Typically, the Drug Encoder employs a multivariate Gaussian distribution
to map the input data points (X, A) to a continuous range of possible 
values between 0 and 1. This results in novel features that are derived 
from the original drug features, providing a new representation of each drug.

However, when dealing with affinity prediction, it is necessary to keep
the actual representation of the input drug to make accurate predictions.
Thus, we utilized the Drug Encoder to yield a pair of outputs as follows

(1):For novel drug generation, we utilize the feature obtained after 
performing the mean and log variance operation (AMVO). This feature 
captures the Underlying of the input drug and is suitable 
for generating new drug compounds.

(2): for the affinity prediction task, we use the features obtained prior
to the mean and log variance operation (PMVO). These features are more 
appropriate for predicting drug affinity, as they retain the original 
characteristics of the input drug without being altered by the AMVO process.
'''
class Encoder(torch.nn.Module):
    def __init__(self, Drug_Features, dropout, Final_dim):
        super(Encoder, self).__init__()
        self.GraphConv1 = GCNConv(Drug_Features, Drug_Features * 2)
        self.GraphConv2 = GCNConv(Drug_Features * 2, Drug_Features * 3)
        self.GraphConv3 = GCNConv(Drug_Features * 3, Drug_Features * 4)
        self.Mu_layer = GCNConv(Drug_Features * 4, Drug_Features * 4)
        self.Logvar_layer = GCNConv(Drug_Features * 4, Drug_Features * 4)
        self.Drug_FCs = nn.Sequential(
            nn.Linear(Drug_Features * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, Final_dim)
        )
        self.Relu_activation = nn.ReLU()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, data):
       x, edge_index, batch = data.x, data.edge_index, data.batch # This line of code extract the (X,Edge index and mini bacth) from data object data
       GCNConv = self.GraphConv1(x, edge_index) # the 1st GCNConv layer takes (node faature X and adjacancey matrix A) in Shape of [mini_batch, node_features], [A]
       GCNConv = self.Relu_activation(GCNConv) # Apply ReLU activation
       GCNConv = self.GraphConv2(GCNConv, edge_index) # the 2nd GCNConv layer takes feature from 1st layer and adjacency matrix
       GCNConv = self.Relu_activation(GCNConv) # Apply ReLU activation
       PMVO = self.GraphConv3(GCNConv, edge_index) # the 3rd layer takes feature from 2nd layer and adjacency matrix
       x = self.Relu_activation(PMVO) # Apply ReLU activation
       mu = self.Mu_layer(x, edge_index)  # Calculate the mean and log variance using respective layers
       logvar = self.Logvar_layer(x, edge_index) #....

       AMVO = self.reparameterize(mu, logvar) #Reparameterization trick to sample from the Gaussian distribution
                                              #Simply the reparameterization trickenables the backpropagation of gradients through stochastic sampling  
       x2 = gmp(x, batch)  # Global max pooling operation along the batch dimension
       PMVO = self.Drug_FCs(x2) # Passed through the Dense layers

       return AMVO, mu, logvar, PMVO


# Define the graph decoder

'''
The Drug Decoder p(Drug|Z_i,Z_j) uses latent space (AMVO)
and generates the the probabality of adjacency matrix for 
newly generated drug.
'''
class Decoder(nn.Module):
    def __init__(self, Drug_Features=94):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x, data):
        batch = data.batch # This line of code extract the the mini batch from data object.
        x_dense, mask = to_dense_batch(x, batch) # Convert the input data of newly generated drug to mini batch.
        adj = torch.matmul(x_dense, x_dense.transpose(1, 2)) # calculates the pubabality of connectivity among the newly generated drug
        adj_prob = self.sigmoid(adj) # passed through the sigmoid activation function to normalize it
        return adj_prob

class GatedCNN(nn.Module):
    def __init__(self, Protein_Features, Num_Filters, Embed_dim, Final_dim, K_size):
        super(GatedCNN, self).__init__()
        # 1D convolution on protein sequence
        self.Protein_Embed = nn.Embedding(Protein_Features + 1, Embed_dim)
        self.Protein_Conv1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Gate1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Conv2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Gate2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Conv3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)
        self.Protein_Gate3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)
        self.relu = nn.ReLU()
        self.Protein_FC = nn.Linear(96 * 107, Final_dim)

    def forward(self, data):
        # Protein input feed-forward:
        target = data.target # This line of code extract the the protein sequences from data object.
        #EMBEDDING LAYER
        Embed = self.Protein_Embed(target) # Pass the protein sequence to Embbeding layer for assigning a dense feature vector to 
        # GATED CNN 1ST LAYER              # and generate the output in shape [batch_size, sequence_length, embed_dim]
        conv1 = self.Protein_Conv1(Embed)  
        gate1 = torch.sigmoid(self.Protein_Gate1(Embed))  # takes the word embbeding matrix and generates two on the convalution value and another is gate value 
        GCNN1_Output = conv1 * gate1                      # element wise product of convalution value [batch_size, num_filtters ,sequence_length] and gated value [batch_size, num_filtters ,sequence_length]
        GCNN1_Output = self.relu(GCNN1_Output)
        #GATED CNN 2ND LAYER
        conv2 = self.Protein_Conv2(GCNN1_Output)
        gate2 = torch.sigmoid(self.Protein_Gate2(GCNN1_Output)) #....
        GCNN2_Output = conv2 * gate2                            #....
        GCNN2_Output = self.relu(GCNN2_Output)
        #GATED CNN 3RD LAYER
        conv3 = self.Protein_Conv3(GCNN2_Output)
        gate3 = torch.sigmoid(self.Protein_Gate3(GCNN2_Output)) #....
        GCNN3_Output = conv3 * gate3                            #....
        GCNN3_Output = self.relu(GCNN3_Output)
        #FLAT TENSOR
        xt = GCNN3_Output.view(-1, 96 * 107) # Flat the output tensor from 3rd gated cnn layer. to pass it from dense layer 
        #PROTEIN FULLY CONNECTED LAYER
        xt = self.Protein_FC(xt)
        return xt

class FC(torch.nn.Module):
    def __init__(self, output_dim, n_output, dropout):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(output_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output)
        )

    def forward(self, Drug_Features, Protein_Features):
        Combined = torch.cat((Drug_Features, Protein_Features), 1) # concatenate the [Drug feature] and [Protein feature] 
                                                                   # along the second dimension (dimension 1), which means the tensors are stacked side by side horizontally.
        Pridection = self.FC_layers(Combined) # Takes the combined feature and pass through the couple of (dense) layers
                                              # These dense layers are responsible for learning complex patterns in the combined features and making predictions based on the learned representations.
        return Pridection

# MAin CLass
class GraphVAE(torch.nn.Module):
    def __init__(self):
        super(GraphVAE, self).__init__()
        self.encoder = Encoder(Drug_Features=94, dropout=0.2, Final_dim = 128)
        self.decoder = Decoder(Drug_Features=94)
        self.cnn = GatedCNN(Protein_Features=25, Num_Filters=32, Embed_dim=128, Final_dim=128, K_size=8)
        self.fc = FC(output_dim=128, n_output=1, dropout=0.3)

    def forward(self,data):
        AMVO, mu, logvar, PMVO = self.encoder(data)
        Generated_drugs = self.decoder(AMVO, data)
        Protein_vector = self.cnn(data)
        Pridection = self.fc(PMVO, Protein_vector)
        return Pridection, Generated_drugs, mu, logvar