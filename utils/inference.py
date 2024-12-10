from torch_geometric.loader import DataLoader
import torch

try:
    from utils.data_loader import LoadMode, load_from_str
    from utils.train import SimpleGraphDataset, LetterGNN
except:
    from data_loader import LoadMode, load_from_str
    from train import SimpleGraphDataset, LetterGNN


from current_setup import kwargs as current_kwargs

def inference(user_id: int, content: str, model_path='', threshold=0.7) -> tuple[float, int]:
    """
    Train and save the model
    :param user_id: user_id for positive labels
    :param content: .tsv content
    :param model_path: Path to save the model. Leave default to save at ./models/<user_id>.pth
    :param threshold: Threshold for positive prediction
    :return: (accuracy, prediction)

    All model parameters are taken from current_setup.py
    """
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    inference_dataset = load_from_str(content, y=torch.tensor([1]),
                                      mode=current_kwargs["mode"], rows_per_example=current_kwargs["rows_per_example"], offset=10)
    inference_dataset = [e.to(device) for e in inference_dataset]
    inference_dataset = SimpleGraphDataset(inference_dataset)

    if model_path == '':
        model_path = f'models/{user_id}.pth'

    loaded_model = LetterGNN(num_node_features=inference_dataset.num_node_features, hidden_conv_dim=current_kwargs["hidden_conv_dim"],
                             use_fc_before=current_kwargs["use_fc_before"], hidden_ff_dim=current_kwargs["hidden_ff_dim"], 
                             num_layers=current_kwargs["num_layers"], num_classes=1).to(device)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.eval()


    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    total_positives = 0
    with torch.no_grad():
        for data in inference_loader:
            data = data.to(device)
            output = loaded_model(data.x, data.edge_index, data.batch)
            prob = torch.sigmoid(output)  # Convert logits to probabilities
            pred = (prob > current_kwargs["threshold"]).float()  # Threshold at 0.5 to get binary predictions
            
            total_positives += pred
    accuracy = float(total_positives / len(inference_dataset))

    if accuracy < threshold:
        return accuracy, 0
    return accuracy, 1


if __name__ == '__main__':
    import sys 
    user = int(sys.argv[1])
    
    model_path = f"../models/models_5_len35/{user}.pth"
    test_file_path = f"../datasets/inference/key_presses_{user}.-1.tsv"
    with open(test_file_path, "r", encoding="utf-8") as file:
        tsv_content = file.read()
    print(inference(user, tsv_content, model_path=f"../models/models_5_len35/{user}.pth", threshold=0.5))
