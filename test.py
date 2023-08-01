import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *
from model import GraphVAE
from FetterGrad import FetterGrad

class GraphVAETester:
    def __init__(self, model_path, dataset='kiba', test_batch_size=512):
        self.model_path = model_path
        self.dataset = dataset
        self.TEST_BATCH_SIZE = test_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]  # list of threshold values for kiba
        # self.thresholds = [5.0, 5.50, 6, 6.50, 7.0, 7.50, 8.0, 8.50] # list  of  trash hold values. for davis

    def load_model(self):
        model = GraphVAE()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()
        self.model = model

    def load_test_data(self):
        self.test_data = TestbedDataset(root='data', dataset=self.dataset + '_test')
        self.test_loader = DataLoader(self.test_data, batch_size=self.TEST_BATCH_SIZE, shuffle=False)

    def calculate_metrics(self):
        total_predict = torch.Tensor().to(self.device)
        total_true = torch.Tensor().to(self.device)
        total_loss = 0
        total_mse = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                data = data.to(self.device) 
                x = data.edge_index
                batch = data.batch
                adj_matrix = to_dense_adj(x.long(), batch)
                prid, x_hat, mu, log_var = self.model(data)
                prid = prid.to(self.device)
                total_true = torch.cat((total_true, data.y.view(-1, 1)), 0) 
                total_predict = torch.cat((total_predict, prid), 0)
                G = total_true.cpu().numpy().flatten()
                P = total_predict.cpu().numpy().flatten()
                mse_loss = mse(G, P)
                test_ci = get_cindex(G, P)
                rm2 = get_rm2(G, P)
                rms = rmse(G, P)

                auc_values = []
                for t in self.thresholds:
                    auc = get_aupr(np.int32(prid.cpu() > t), data.y.view(-1, 1).float().cpu())
                    auc_values.append(auc)

        return mse_loss, test_ci, rm2, auc_values


if __name__ == "__main__":
    dataset_name = 'kiba'
    model_file = f'graphvae_model_{dataset_name}.pth'
    model_path = f'saved_models/{model_file}'
    tester = GraphVAETester(model_path, dataset=dataset_name)  
    tester.load_model()
    tester.load_test_data()
    mse_loss, test_ci, rm2, auc_values = tester.calculate_metrics()
    print('MSE: {:.4f}, CI: {:.4f}, RM2: {:.4f}, auc: {}'.format(mse_loss, test_ci, rm2, auc_values))
