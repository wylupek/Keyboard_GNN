import sqlite3
import os
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch_geometric.loader as torchLoader
import utils.data_loader as loader
import utils.train as training
from utils import current_setup

# Database interaction to get user IDs
def get_user_ids(database_path):
    """
    Get all the users from the database.
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"The database file '{database_path}' does not exist.")

    try:
        conn = sqlite3.connect(database_path)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"An error occurred while connecting to the database: {e}")

    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT user_id
        FROM key_press
    """)
    return [i[0] for i in cursor.fetchall()]

positive_threshold = 0.55
# predefined modes and values

kwargs = current_setup.kwargs

def main():
    mode = kwargs.get('mode')
    rows_per_example = kwargs.get('rows_per_example')
    folder_name = f"benchmark_{mode}_{rows_per_example}_{positive_threshold*100}"
    os.makedirs(folder_name, exist_ok=True)



    model_path_prefix = "models/no_acc_and_cap/"
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    database_path = "../keystroke_data.sqlite"
    user_ids = get_user_ids(database_path)

    print(f"Found {len(user_ids)} user IDs in the database.")

    # Prepare training tasks
    succ = True
    for user in user_ids:
        model_path = model_path_prefix + f"{user}.pth"
        if not os.path.exists(model_path):
            print(f"Missing model for user {user}, go generate it by running 'python utlis/train.py {user}' ")
            succ=False
    
    if not succ:
        exit()

    user_test_examples = {}
    # mat[i][j] contains the number of times model i answered true for user j.
    # note that we want this number to be large for when i==j but small otherwise
    positive_ans_count_mat = [[0 for _ in user_ids] for _ in user_ids]
    positive_ans_percent_mat = [[0 for _ in user_ids] for _ in user_ids]
    # rows are models, columns are users
    user_models = []
    examples_per_user = {}

    # LOAD MODELS
    for user in user_ids:
        inference_path = f"datasets/inference/key_presses_{user}.-1.tsv"
        model_path = model_path_prefix + f"{user}.pth"

        if not os.path.exists(inference_path):
            print(f"Missing inference data for user {user}. ERROR")
            exit()

        examples = loader.load_from_file(inference_path, 0, mode=kwargs["mode"], rows_per_example=kwargs["rows_per_example"], offset=1, agg_time=True)
        examples_per_user[user] = len(examples)
        dataset = training.SimpleGraphDataset([e.to(device) for e in examples])
        user_test_examples[user] = dataset

        model = training.LetterGNN(num_classes=1, num_node_features=dataset.num_node_features, num_layers=kwargs["num_layers"], hidden_conv_dim=kwargs["hidden_conv_dim"],
                                   use_fc_before=kwargs["use_fc_before"], hidden_ff_dim=kwargs["hidden_ff_dim"], ).to(device)


        model.load_state_dict(torch.load(model_path))
        model.eval()
        user_models.append(model)

    # Create evals 
    for i, predicting_model_id in enumerate(user_ids):
        for j, actual_use_id in enumerate(user_ids):
            test_loader = torchLoader.DataLoader(user_test_examples[actual_use_id], batch_size=1, shuffle=False)
            ones_counter = 0
            for data in test_loader:
                output = user_models[i](data.x, data.edge_index, data.batch)
                prob = torch.sigmoid(output)  # Convert logits to probabilities
                pred = (prob > positive_threshold).float()  # Threshold at 0.5 to get binary predictions
                if pred[0].item() == 1: # the predicted class is 1
                    ones_counter += 1

            positive_ans_count_mat[i][j] = ones_counter
            positive_ans_percent_mat[i][j] = int(ones_counter/len(test_loader)*100)

            

    # Convert the data to a NumPy array for compatibility with matplotlib
    results_array = np.array(positive_ans_percent_mat)

    # Create the plot
    plt.figure(figsize=(8, 6))
    cax = plt.imshow(results_array, cmap="Blues", interpolation="nearest")

    # Add color bar for reference
    plt.colorbar(cax, label="Number of Positive Predictions")

    # Add labels to axes
    plt.xticks(ticks=np.arange(len(user_ids)), labels=user_ids, fontsize=10)
    plt.yticks(ticks=np.arange(len(user_ids)), labels=user_ids, fontsize=10)
    plt.xlabel("Actual User ID", fontsize=12)
    plt.ylabel("Predicting Model ID", fontsize=12)

    # Add text annotations inside each cell
    for i in range(results_array.shape[0]):
        for j in range(results_array.shape[1]):
            plt.text(j, i, str(results_array[i, j]),
                    ha="center", va="center", color="black", fontsize=10)

    # Add title
    plt.title(f"Confusion Matrix Heatmap", fontsize=14)

    # Add kwargs as a footnote
    plt.figtext(0.5, -0.1, f"kwargs: {kwargs}", wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{folder_name}/len40_{model_path_prefix.split("/")[-2]}.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()

    TPRs = []
    FPRs = []

    # Create evals, user i is positive, user j is negative 
    for i, predicting_model_id in enumerate(user_ids):
        TP = 0 
        FP = 0 
        TN = 0 
        FN = 0 
        for j, actual_use_id in enumerate(user_ids):
            if i == j:
                TP += positive_ans_count_mat[i][i]
                FN += examples_per_user[actual_use_id] - positive_ans_count_mat[i][i]
            else:
                FP += positive_ans_count_mat[i][j]
                TN += examples_per_user[actual_use_id] - positive_ans_count_mat[i][j] # all examples not labeled 1 are TN

        # Calculate true positive rate (TPR) and false positive rate (FPR)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        print(f"Model {predicting_model_id} - TPR: {TPR:.4f}, FPR: {FPR:.4f}")

        # Append TPR and FPR to lists for later use
        TPRs.append(TPR)
        FPRs.append(FPR)

        print(predicting_model_id)            
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print("num positive exmples: ", examples_per_user[predicting_model_id])

        # Create confusion matrix
        confusion_matrix = np.array([[TP, FN], [FP, TN]])

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        cax = plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")

        # Add color bar for reference
        plt.colorbar(cax, label="Count")

        # Add labels to axes
        plt.xticks(ticks=[0, 1], labels=["Positive", "Negative"], fontsize=10)
        plt.yticks(ticks=[0, 1], labels=["Positive", "Negative"], fontsize=10)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)

        # Add text annotations inside each cell
        for m in range(confusion_matrix.shape[0]):
            for n in range(confusion_matrix.shape[1]):
                plt.text(n, m, str(confusion_matrix[m, n]),
                         ha="center", va="center", color="black", fontsize=10)

        # Add title
        plt.title(f"Confusion Matrix for Model {predicting_model_id}", fontsize=14)

        # Save the plot as a PDF
        plt.tight_layout()
        plt.savefig(f"{folder_name}/confusion_matrix_{predicting_model_id}.pdf", format="pdf")

        # Close the figure to free resources
        plt.close()

    # Plot ROC curve with labeled points for each user
    plt.figure(figsize=(8, 6))
    for i, (fpr, tpr) in enumerate(zip(FPRs, TPRs)):
        plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
        plt.text(fpr, tpr, user_ids[i], fontsize=9, ha='right')

    # Calculate and plot the average TPR and FPR
    avg_TPR = sum(TPRs) / len(TPRs)
    avg_FPR = sum(FPRs) / len(FPRs)
    plt.plot(avg_FPR, avg_TPR, marker='x', color='g', markersize=10, label='Average')

    # Annotate the average point
    plt.text(avg_FPR, avg_TPR, 'Average', fontsize=9, ha='right')

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')

    # Add labels and title
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')

    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(f"{folder_name}/roc_curve.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()
    

    # Create evals for every user vs user pair
    for i, predicting_model_id in enumerate(user_ids):
        for j, negative_user_id in enumerate(user_ids):
            if i == j:
                continue
            
            TP = positive_ans_count_mat[i][i]
            FN = examples_per_user[predicting_model_id] - positive_ans_count_mat[i][i]
            FP = positive_ans_count_mat[i][j]
            TN = examples_per_user[negative_user_id] - positive_ans_count_mat[i][j] # all examples not labeled 1 are TN
            
            # Create confusion matrix
            confusion_matrix = np.array([[TP, FN], [FP, TN]])

            # Plot confusion matrix
            plt.figure(figsize=(6, 4))
            cax = plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")

            # Add color bar for reference
            plt.colorbar(cax, label="Count")

            # Add labels to axes
            plt.xticks(ticks=[0, 1], labels=["Positive", "Negative"], fontsize=10)
            plt.yticks(ticks=[0, 1], labels=["Positive", "Negative"], fontsize=10)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)

            # Add text annotations inside each cell
            for m in range(confusion_matrix.shape[0]):
                for n in range(confusion_matrix.shape[1]):
                    plt.text(n, m, str(confusion_matrix[m, n]),
                            ha="center", va="center", color="black", fontsize=10)

            # Add title
            plt.title(f"Confusion Matrix for Model {predicting_model_id} vs User {negative_user_id}", fontsize=14)

            # Save the plot as a PDF
            plt.tight_layout()
            plt.savefig(f"{folder_name}/user_{predicting_model_id}_{negative_user_id}.pdf", format="pdf")

            # Close the figure to free resources
            plt.close()


    # Write results to res.txt
    with open(f"{folder_name}/res.txt", "w") as f:
        for i, model_id in enumerate(user_ids):
            f.write(f"Model {model_id} - true_positive_rate: {TPRs[i]:.4f}, false_positive_rate: {FPRs[i]:.4f}\n")
        
        avg_true_positive_rate = sum(TPRs) / len(TPRs)
        avg_false_positive_rate = sum(FPRs) / len(FPRs)
        
        f.write(f"\nAverage true_positive_rate: {avg_true_positive_rate:.4f}\n")
        f.write(f"Average false_positive_rate: {avg_false_positive_rate:.4f}\n")

if __name__ == "__main__":
    main()
