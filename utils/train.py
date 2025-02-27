import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
import torch_geometric.loader as torchLoader
from collections import Counter

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import load_from_db, LoadMode
from sklearn.metrics import precision_score, recall_score, f1_score


# Define a simple GCN model
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
        self._data, self.slices = self.collate(data_list)  # Collate all data objects

    def __len__(self):
        return len(self._data.y)  # Number of graphs in the dataset

    def statistics(self) -> str:
        class_counts = Counter(self._data.y.cpu().numpy())
        return " | ".join([f"{item_class}: {count}" for item_class, count in
                           class_counts.items()]) + f" | Total: {sum(class_counts.values())}"


def train(database_path: str, user_id: str, model_path='', mode=LoadMode.ONE_HOT,
          test_train_split=0.2, hidden_dim=128, epochs_num=1000,
          rows_per_example=50, positive_negative_ratio=0.5, offset=10) -> float:
    """
    Train and save the model
    :param database_path: Path to database with key presses
    :param user_id: user_id for positive labels
    :param model_path: Path to save the model. Leave default to save at ./model/<user_id>.pth
    :param mode: Mode for processing node attributes
    :param test_train_split: test to all examples proportion, set 0 for training only
    :param hidden_dim: hidden dimension
    :param epochs_num: number of epochs for training loop
    :param rows_per_example: number of key presses per example
    :param positive_negative_ratio: positive to negative class ratio, set 0 to load all examples,
        but take care of class imbalance.
    :param offset: Number of rows between beginning of each example
    :return: accuracy of the model
    """
    if model_path == '':
        model_path = f'models/{user_id}.pth'

    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    examples_pos, examples_neg_list = load_from_db(
        database_path=database_path, user_id=user_id, positive_negative_ratio=positive_negative_ratio,
        mode=mode, rows_per_example=rows_per_example, offset=offset
    )

    if test_train_split > 0.0:
        train_neg = []
        test_neg = []
        for examples_neg in examples_neg_list:
            # we could use a smaller buffer than rows_per_example since we already skip some
            # example in the data loading stage, but let's just be safe
            tr = examples_neg [
                0:round((1-test_train_split)*len(examples_neg)) - rows_per_example
            ]
            ts = examples_neg[
                round((1-test_train_split)*len(examples_neg)):
            ]
            train_neg.extend(tr)
            test_neg.extend(ts)

        train_pos = examples_pos[
            0:round((1-test_train_split)*len(examples_pos)) - rows_per_example
        ]
        test_pos = examples_pos[
            round((1-test_train_split)*len(examples_pos)):
        ]
        
        train_examples = train_pos + train_neg
        test_examples = test_pos + test_neg
        print(f"train_examples {len(train_examples)}")
        print(f"test_examples {len(test_examples)}")

        train_examples = SimpleGraphDataset([e.to(device) for e in train_examples])
        test_examples = SimpleGraphDataset([e.to(device) for e in test_examples])
        print("Train dataset statistics: ", train_examples.statistics())
        print("Test dataset statistics:  ", test_examples.statistics())
    else:
        train_examples = examples_pos # + examples_neg
        for example in examples_neg_list:
            train_examples.extend(example)
        test_examples = []
        train_examples = SimpleGraphDataset([e.to(device) for e in train_examples])
        print("Train dataset statistics: ", train_examples.statistics())


    model = LetterGNN(num_node_features=train_examples.num_node_features, hidden_dim=hidden_dim,
                      num_classes=train_examples.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training over epochs
    data_loader = torchLoader.DataLoader(train_examples, batch_size=32, shuffle=True)
    for epoch in range(1, epochs_num):
        model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)  # Forward pass
            loss = criterion(output, data.y)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters
            total_loss += loss.item()
        loss = total_loss / len(data_loader)
        if epoch % 50 == 0 or epoch == epochs_num - 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        if round(loss, 3) == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            break

    torch.save(model.state_dict(), model_path)

    if len(test_examples):
        test_loader = torchLoader.DataLoader(test_examples, batch_size=1, shuffle=False)
        model.eval()
        preds = []
        bases = []
        for data in test_loader:
            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)  # Get the index of the max log-probability
            bases.append(data.y[0].item())
            preds.append(pred[0].item())

        precision = precision_score(bases, preds)
        recall = recall_score(bases, preds)
        f1 = f1_score(bases, preds)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return precision
    return 0.0


def train_with_crossvalidation(database_path: str, user_id: str, model_path='', mode=LoadMode.ONE_HOT,
          test_train_split=0.2, hidden_dim=128, epochs_num=1000,
          rows_per_example=50, positive_negative_ratio=0.5, offset=10) -> float:
    """
    copy of the above func just training the model multiple times and performing cross validation
    """

    def lower_upper_split(lower, upper, l, skip_boundry):
        from math import floor
        
        bigger_left = l[
            0 : 
            floor(max(0, lower*len(l)-skip_boundry))
        ]
        middle = l[
            floor(lower*len(l)) : floor(upper*len(l))]
        bigger_right = l[ floor(upper*len(l)) + skip_boundry: ]

        return bigger_left+bigger_right, middle


    if model_path == '':
        model_path = f'models/{user_id}.pth'

    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    examples_pos, examples_neg_list = load_from_db(
        database_path=database_path, user_id=user_id, positive_negative_ratio=positive_negative_ratio,
        mode=mode, rows_per_example=rows_per_example, offset=1
    )

    k = 1/test_train_split
    # choose the part of the dataset that will be user for testing
    lowers = [(1/k)*i for i in range(5)] 
    uppers = [(1/k)*(i+1) for i in range(5)] 
    
    for lower,upper in zip(lowers, uppers):

        if test_train_split > 0.0:
            train_neg = []
            test_neg = []
            for examples_neg in examples_neg_list:
                # we could user a smaller buffer than rows_per_example since we already skip some
                # example in the data loading stage, but lets just be safe
                
                # take lower split of data from 
                tr, ts = lower_upper_split(lower, upper, examples_neg, rows_per_example)

                train_neg.extend(tr)
                test_neg.extend(ts)


            train_pos, test_pos = lower_upper_split(lower, upper, examples_pos, rows_per_example)
            

            train_examples = train_pos + train_neg
            test_examples = test_pos + test_neg
            print(f"train_examples {len(train_examples)}")
            print(f"test_examples {len(test_examples)}")

            train_examples = SimpleGraphDataset([e.to(device) for e in train_examples])
            test_examples = SimpleGraphDataset([e.to(device) for e in test_examples])

            print("Train dataset statistics: ", train_examples.statistics())
            print("Test dataset statistics:  ", test_examples.statistics())
        else:
            train_examples = examples_pos + examples_neg
            test_examples = []
            train_examples = SimpleGraphDataset([e.to(device) for e in train_examples])
            print("Train dataset statistics: ", train_examples.statistics())


        model = LetterGNN(num_node_features=train_examples.num_node_features, hidden_dim=hidden_dim,
                        num_classes=train_examples.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Training over epochs
        data_loader = torchLoader.DataLoader(train_examples, batch_size=32, shuffle=True)
        best_model = None
        smallest_loss = 1000
        for epoch in range(1, epochs_num):
            model.train()
            total_loss = 0
            for data in data_loader:
                optimizer.zero_grad()
                output = model(data.x, data.edge_index, data.batch)  # Forward pass
                loss = criterion(output, data.y)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the model parameters
                total_loss += loss.item()
            loss = total_loss / len(data_loader)

            # # Validate the model
            # model.eval()
            # validate_loader = torchLoader.DataLoader(validate_examples, batch_size=1, shuffle=False)
            # validate_preds = []
            # validate_bases = []
            # for data in validate_loader:
            #     output = model(data.x, data.edge_index, data.batch)
            #     pred = output.argmax(dim=1)
            #     validate_bases.append(data.y[0].item())
            #     validate_preds.append(pred[0].item())
            # validate_precision = precision_score(validate_bases, validate_preds)
            if loss < smallest_loss:
                smallest_loss = loss
                best_model = model.state_dict()

            if epoch % 50 == 0 or epoch == epochs_num - 1:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            if round(loss, 3) == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
                break

        torch.save(best_model, model_path)

        if len(test_examples):
            test_loader = torchLoader.DataLoader(test_examples, batch_size=1, shuffle=False)
            model.eval()
            preds = []
            bases = []
            for data in test_loader:
                output = model(data.x, data.edge_index, data.batch)
                pred = output.argmax(dim=1)  # Get the index of the max log-probability
                bases.append(data.y[0].item())
                preds.append(pred[0].item())

            # precision = precision_score(bases, preds)
            # recall = recall_score(bases, preds)
            # f1 = f1_score(bases, preds)
            tp = sum((p == 1 and b == 1) for p, b in zip(preds, bases))
            fp = sum((p == 1 and b == 0) for p, b in zip(preds, bases))
            tn = sum((p == 0 and b == 0) for p, b in zip(preds, bases))
            fn = sum((p == 0 and b == 1) for p, b in zip(preds, bases))

            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            with open(f"dupa_{user_id}.txt", "a") as f:
                f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")

    return 0.0
   

if __name__ == '__main__':
    user = "user2"
    if len(sys.argv) > 1:
        user = sys.argv[1]

    for dims in [256]:
        for mode in [LoadMode.ONE_HOT, LoadMode.DROP]:
            for row_per_example in [35, 40, 50]:
                with open(f"dupa_{user}.txt", "a") as f:
                    f.write(f"dims: {dims}, rows: {row_per_example}, mode: {str(mode)}\n")

                train_with_crossvalidation("../keystroke_data.sqlite", user,
                    model_path='../models/test.pth', test_train_split=0.2, positive_negative_ratio=1,
                    mode=mode, epochs_num=500, hidden_dim=dims, rows_per_example=row_per_example)

