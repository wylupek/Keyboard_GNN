import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split
import torch_geometric.loader as torchLoader

import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.data_loader as dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LetterGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=2):
        super(LetterGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = x.float()
            edge_index = edge_index.long()
            x = conv.forward(x, edge_index)
            x = F.relu(x)
        
        # idk maybe use a different pool method 
        x = pyg_nn.global_mean_pool(x, batch)

        # classify
        x = self.fc(x)

        return x
        
class SimpleGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(SimpleGraphDataset, self).__init__('.', None, None, None)
        self.data, self.slices = self.collate(data_list)  # Collate all data objects

    def __len__(self):
        return len(self.data.y)  # Number of graphs in the dataset


df_body = []

test_train_split = 0.25 # 20% into test
NUM_HIDDEN_DIMS = 64
NUM_EPOCHS = 400

file_names = ["../datasets/user1.tsv", "../datasets/user2.tsv", "../datasets/user4.tsv", "../datasets/user3.tsv"]


# stuff we want to change
modes = [dl.LoadMode.ONE_HOT]
rows_per_example_options = [25, 30, 35]

for rows_per_example in rows_per_example_options:
    for mode in modes:
        # this is a list of list of Data objects - datasets
        # each dataset comes from a different input file - a different user
        list_of_datasets = [dl.load_from_file(filename, mode=mode, y=torch.tensor([i]), rows_per_example=rows_per_example)
                            for i, filename in enumerate(file_names)]

        num_features = list_of_datasets[0][0].x.shape[1]

        for positive_index in range(len(file_names)):
            print(f"Rows per example: {rows_per_example}, Mode: {mode}, Positive index: {positive_index}")

            training_dataset_pos = []
            testing_dataset_pos = []
            training_dataset_neg = []
            testing_dataset_neg = []

            # relabel the datasets
            for i, dataset in enumerate(list_of_datasets):
                for data_obj in dataset:
                    if i == positive_index:
                        data_obj.y = torch.tensor([1])
                    else:
                        data_obj.y = torch.tensor([0])

                train, test = train_test_split(dataset, test_size=test_train_split)

                if i == positive_index:
                    training_dataset_pos.extend(train)
                    testing_dataset_pos.extend(test)
                else:
                    training_dataset_neg.extend(train)
                    testing_dataset_neg.extend(test)

            train = [e.to(device) for e in (training_dataset_pos + training_dataset_neg)]
            test = [e.to(device) for e in (testing_dataset_pos + testing_dataset_neg)]
            dataset = SimpleGraphDataset(train)
            test_dataset = SimpleGraphDataset(test)


            model = LetterGNN(num_node_features=dataset.num_node_features, hidden_dim=NUM_HIDDEN_DIMS, num_classes=dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            # Train loop
            def train(model, data_loader):
                model.train()
                total_loss = 0
                for data in data_loader:
                    optimizer.zero_grad()
                    output = model(data.x, data.edge_index, data.batch)  # Forward pass
                    loss = criterion(output, data.y)  # Compute the loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update the model parameters
                    total_loss += loss.item()
                return total_loss / len(data_loader)


            # Training over epochs
            data_loader = torchLoader.DataLoader(dataset, batch_size=32, shuffle=True)
            prev_loss = 100000
            for epoch in range(1, NUM_EPOCHS):
                loss = train(model, data_loader)
                if epoch % 50 == 0 or epoch == NUM_EPOCHS-1:
                    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
                
                if round(loss, 3) == 0 and round(prev_loss-loss, 2) == 0:
                    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
                    break
            
                prev_loss = loss

            def test(model, data_loader):
                model.eval()
                correct = 0
                for data in data_loader:
                    output = model(data.x, data.edge_index, data.batch)
                    pred = output.argmax(dim=1)  # Get the index of the max log-probability
                    # print(f"Correct {data.y[0]} Pred {pred[0]}")
                    if pred != data.y:
                        print(output[0])
                    correct += (pred == data.y).sum().item()
                return correct / len(data_loader.dataset)

            def test_acc():
                # Test the model
                test_loader = torchLoader.DataLoader(test_dataset, batch_size=1, shuffle=False)
                accuracy = test(model, test_loader)
                print(f"Test Accuracy: {accuracy:.4f}")
                return accuracy 

            acc = test_acc()
            df_body.append(
                (rows_per_example, mode, positive_index, acc)
            )
            with open("zz.txt", "a") as f:
                f.writelines(
                    [", ".join((str(rows_per_example), str(mode), str(positive_index), str(acc))), "\n"]
                )
            

df = pd.DataFrame(df_body, columns=["rows_per_example", "mode", "pos_index", "acc"])
print(df)
df.to_csv("res.csv")