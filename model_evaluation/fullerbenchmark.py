from adjustText import adjust_text
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

def plot_confusion_matrix(folder_name, predicting_model_id, confusion_matrix):
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

    # # Add title
    # plt.title(f"Confusion Matrix for Model {predicting_model_id}", fontsize=14)

    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(f"{folder_name}/confusion_matrix_{predicting_model_id}.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()

def plot_heatmap(folder_name, results_array, user_ids, model_path_prefix, kwargs):
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
    # plt.title(f"Confusion Matrix Heatmap", fontsize=14)

    # Add kwargs as a footnote
    plt.figtext(0.5, -0.1, f"kwargs: {kwargs}", wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{folder_name}/len40_{model_path_prefix.split('/')[-2]}.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()

import numpy as np

def plot_roc_curves(user_models, proba_mat, user_ids, folder_name):
    # Define threshold values to iterate over
    thresholds = [i * 0.05 for i in range(0, 21)]

    # Initialize lists to store TPR and FPR for each model and threshold
    all_tprs = []
    all_fprs = []
    all_fars = []  # FAR values for EER calculation
    all_frrs = []  # FRR values for EER calculation

    eer_results = []  # To store EER results for each model

    # Iterate over each model
    for i, model in enumerate(user_models):
        print("user_num ", i)
        model_tprs = []
        model_fprs = []
        model_fars = []  # FAR values for EER calculation
        model_frrs = []  # FRR values for EER calculation

        # Iterate over each threshold
        for threshold in thresholds:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for j, user_id in enumerate(user_ids):
                proba_vec = proba_mat[i][j]
                for prob in proba_vec:
                    pred = (prob > threshold).float()  # Apply threshold
                    if i == j:  # Positive class (same user)
                        for x in pred:
                            if x.item() == 1:
                                TP += 1
                            else:
                                FN += 1
                    else:  # Negative class (different user)
                        for x in pred:
                            if x.item() == 1:
                                FP += 1
                            else:
                                TN += 1

            # Calculate TPR and FPR for current threshold
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            model_tprs.append(TPR)
            model_fprs.append(FPR)

            # Calculate FAR and FRR for EER
            FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FRR = FN / (TP + FN) if (TP + FN) > 0 else 0
            model_fars.append(FAR)
            model_frrs.append(FRR)

        # Compute EER for this model
        fars = np.array(model_fars)
        frrs = np.array(model_frrs)
        eer_threshold_idx = np.argmin(np.abs(fars - frrs))  # Find where FAR and FRR are closest
        eer = (fars[eer_threshold_idx] + frrs[eer_threshold_idx]) / 2
        eer_threshold = thresholds[eer_threshold_idx]
        eer_results.append((user_ids[i], eer, eer_threshold))  # Store EER and corresponding threshold

        all_tprs.append(model_tprs)
        all_fprs.append(model_fprs)
        all_fars.append(model_fars)
        all_frrs.append(model_frrs)

    # Write EER results to file
    with open(f"{folder_name}/eer.txt", "w") as eer_file:
        total_eer = 0
        for user_id, eer, threshold in eer_results:
            eer_file.write(f"model: {user_id} eer: {eer:.4f} threshold: {threshold:.2f}\n")
            total_eer += eer
        average_eer = total_eer / len(eer_results)
        eer_file.write(f"Average EER: {average_eer:.4f}\n")

    # Plot ROC curves on a grid of subplots (4x4)
    num_models = len(user_models)
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_models:
            ax = axes[i]
            ax.plot(all_fprs[i], all_tprs[i], marker='o', label=f'Model {user_ids[i]}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
            ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
            ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
            # ax.set_title(f'ROC Curve for Model {user_ids[i]}', fontsize=14)
            ax.legend(loc='lower right')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder_name}/roc_curves_all_models_subplots.pdf", format="pdf")
    plt.close()


    # Plot FAR and FRR curves on a grid of subplots (4x4)
    num_models = len(user_models)
    rows = 6
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 20))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_models:
            ax = axes[i]
            ax.plot(thresholds, all_fars[i], marker='o', label='FAR', color='r')
            ax.plot(thresholds, all_frrs[i], marker='o', label='FRR', color='b')
            ax.set_xlabel('Threshold', fontsize=12)
            ax.set_ylabel('FRR and FRR Rates', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right')
            ax.set_title(f'Model {user_ids[i]}', fontsize=14)
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder_name}/far_frr_curves_all_models_subplots.pdf", format="pdf")
    plt.close()


def plot_user_vs_user(folder_name, user_ids, positive_ans_count_mat, examples_per_user):
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
            # plt.title(f"Confusion Matrix for Model {predicting_model_id} vs User {negative_user_id}", fontsize=14)

            # Save the plot as a PDF
            plt.tight_layout()
            plt.savefig(f"{folder_name}/user_{predicting_model_id}_{negative_user_id}.pdf", format="pdf")

            # Close the figure to free resources
            plt.close()

def plot_precision_recall_curves(user_models, proba_mat, user_ids, folder_name):
    # Define threshold values to iterate over
    thresholds = [i * 0.05 for i in range(0, 21)]

    # Initialize lists to store Precision and Recall for each model and threshold
    all_precisions = []
    all_recalls = []

    # Iterate over each model
    for i, model in enumerate(user_models):
        print("user_num ", i)
        model_precisions = []
        model_recalls = []

        # Iterate over each threshold
        for threshold in thresholds:
            TP = 0
            FP = 0
            FN = 0

            for j, user_id in enumerate(user_ids):
                for prob in proba_mat[i][j]:
                    pred = (prob > threshold).float()  # Apply threshold
                    if i == j:  # Positive class (same user)
                        for x in pred:
                            if x.item() == 1:
                                TP += 1
                            else:
                                FN += 1
                    else:  # Negative class (different user)
                        for x in pred:
                            if x.item() == 1:
                                FP += 1

            # Calculate Precision and Recall for current threshold
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            model_precisions.append(precision)
            model_recalls.append(recall)

        all_precisions.append(model_precisions)
        all_recalls.append(model_recalls)

    # Plot Precision-Recall curves on a grid of subplots (4x4)
    num_models = len(user_models)
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_models:
            ax = axes[i]
            ax.plot(all_recalls[i], all_precisions[i], marker='o', label=f'Model {user_ids[i]}')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            # ax.set_title(f'Precision-Recall Curve for Model {user_ids[i]}', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower left')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder_name}/precision_recall_curves_all_models_subplots.pdf", format="pdf")
    plt.close()


# predefined modes and values
kwargs = current_setup.kwargs
positive_threshold = kwargs['threshold']

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def main():
    mode = kwargs.get('mode')
    rows_per_example = kwargs.get('rows_per_example')
    folder_name = f"full_benchmark_half_neg_example_{mode}_{rows_per_example}_{positive_threshold*100}"
    os.makedirs(folder_name, exist_ok=True)

    model_path_prefix = "models/model_5_50_half_neg_but_some_more/"
    if not model_path_prefix.endswith("/"):
        model_path_prefix = model_path_prefix + "/"

    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    database_path = "../keystroke_data.sqlite"
    user_ids = get_user_ids(database_path)

    print(f"Found {len(user_ids)} user IDs in the database.")

    # Prepare training tasks
    succ = True
    for user in user_ids:
        model_path = model_path_prefix + f"{user}.pth"
        if not os.path.exists(model_path):
            print(f"Missing model for user {user}, go generate it by running 'python utils/train.py {user}' ")
            succ = False

    if not succ:
        exit()

    user_test_examples = {}
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
                                   use_fc_before=kwargs["use_fc_before"], hidden_ff_dim=kwargs["hidden_ff_dim"] ).to(device)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        user_models.append(model)

    # Create evals
    positive_ans_count_mat = [[0 for _ in user_ids] for _ in user_ids]
    positive_ans_percent_mat = [[0 for _ in user_ids] for _ in user_ids]

    proba_mat = [[[] for _ in user_ids] for _ in user_ids]

    for i, predicting_model_id in enumerate(user_ids):
        for j, actual_use_id in enumerate(user_ids):
            test_loader = torchLoader.DataLoader(user_test_examples[actual_use_id], batch_size=1, shuffle=False)
            ones_counter = 0
            for data in test_loader:
                output = user_models[i](data.x, data.edge_index, data.batch)
                prob = torch.sigmoid(output)  # Convert logits to probabilities
                proba_mat[i][j].append( prob )
                pred = (prob > positive_threshold).float()  # Threshold at 0.5 to get binary predictions
                if pred[0].item() == 1:  # the predicted class is 1
                    ones_counter += 1

            positive_ans_count_mat[i][j] = ones_counter
            positive_ans_percent_mat[i][j] = int(ones_counter / len(test_loader) * 100)

    # Convert the data to a NumPy array for compatibility with matplotlib
    results_array = np.array(positive_ans_percent_mat)

    # Plot heatmap
    plot_heatmap(folder_name, results_array, user_ids, model_path_prefix, kwargs)

    TPRs = []
    FPRs = []
    precisions = []
    recalls = []
    accs = []
    frrs = []
    fars = []

    f = open(f"{folder_name}/confusion_matrix_{predicting_model_id}.txt", "w")

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
                TN += examples_per_user[actual_use_id] - positive_ans_count_mat[i][j]  # all examples not labeled 1 are TN

        # Calculate true positive rate (TPR) and false positive rate (FPR)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (TP + FN) if (TP + FN) > 0 else 0
        frrs.append(FRR)
        fars.append(FAR)


        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Append precision and recall to lists for later use
        precisions.append(precision)
        recalls.append(recall)

        # Calculate accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        accs.append(accuracy)
        print(f"Model {predicting_model_id} - Accuracy: {accuracy:.4f}")

        # Print precision and recall
        print(f"Model {predicting_model_id} - Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Save confusion matrix to a text file
        f.write(f"Model:{predicting_model_id} TP:{TP} FP:{FP} TN:{TN} FN:{FN}\n")

        print(f"Model {predicting_model_id} - TPR: {TPR:.4f}, FPR: {FPR:.4f}")

        # Append TPR and FPR to lists for later use
        TPRs.append(TPR)
        FPRs.append(FPR)

        print(predicting_model_id)
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print("num positive examples: ", examples_per_user[predicting_model_id])

        # Create confusion matrix
        confusion_matrix = np.array([[TP, FN], [FP, TN]])

        # Plot confusion matrix
        plot_confusion_matrix(folder_name, predicting_model_id, confusion_matrix)

    # Plot FAR vs FRR for each model
    plt.figure(figsize=(8, 6))
    texts = []
    for i, (far, frr) in enumerate(zip(fars, frrs)):
        plt.plot(frr, far, marker='o', linestyle='-', color='b')
        texts.append(plt.text(frr, far, user_ids[i], fontsize=9))


    # Calculate and plot the average FAR and FRR
    avg_FRR = sum(frrs) / len(frrs)
    avg_FAR = sum(fars) / len(fars)
    plt.axvline(x=avg_FRR, color='r', linestyle='--', label='Average FRR')
    plt.axhline(y=avg_FAR, color='g', linestyle='--', label='Average FAR')

    # Add labels and title
    plt.xlabel('False Rejection Rate (FRR)', fontsize=12)
    plt.ylabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')


    # Save the plot as a PDF
    plt.tight_layout()
   
    # Adjust text positions to prevent overlapping with a larger offset
    adjust_text(texts, expand_points=(2, 2))
   
    plt.savefig(f"{folder_name}/far_vs_frr.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()

    f.close()

    plt.figure(figsize=(8, 6))
    texts = []
    for i, (fpr, tpr) in enumerate(zip(FPRs, TPRs)):
        plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
        texts.append(plt.text(fpr, tpr, user_ids[i], fontsize=9))

    # Adjust text positions to prevent overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

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
    plt.legend(loc='lower right')

    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(f"{folder_name}/roc_curve.pdf", format="pdf")

    # Close the figure to free resources
    plt.close()

    # Create ROC curves for all models
    plot_roc_curves(user_models, proba_mat, user_ids, folder_name)

    # Create evals for every user vs user pair
    # plot_user_vs_user(folder_name, user_ids, positive_ans_count_mat, examples_per_user)

    

    # Write results to res.txt
    with open(f"{folder_name}/res.txt", "w") as f:
        for i, model_id in enumerate(user_ids):
            f.write(f"Model {model_id} - true_positive_rate: {TPRs[i]:.4f}, false_positive_rate: {FPRs[i]:.4f}\n")
            f.write(f"Precision: {precisions[i]:.4f}, Recall: {recalls[i]:.4f}\n")
            f.write(f"Accuracy: {accs[i]:.4f}\n")

        avg_true_positive_rate = sum(TPRs) / len(TPRs)
        avg_false_positive_rate = sum(FPRs) / len(FPRs)

        f.write(f"\nAverage true_positive_rate: {avg_true_positive_rate:.4f}\n")
        f.write(f"Average false_positive_rate: {avg_false_positive_rate:.4f}\n")
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)

        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")

        avg_accuracy = sum(accs) / len(accs)
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")

    # Add the new function call to the main function
    # Create Precision-Recall curves for all models
    plot_precision_recall_curves(user_models, proba_mat, user_ids, folder_name)


    # Find the threshold for which average FAR and FRR are equal or within 0.05 of each other
    def find_eer_threshold(proba_mat, user_ids, thresholds):
        all_fars = []
        all_frrs = []

        for i, user_id in enumerate(user_ids):
            model_fars = []
            model_frrs = []

            for threshold in thresholds:
                TP = 0
                FP = 0
                TN = 0
                FN = 0

                for j, actual_user_id in enumerate(user_ids):
                    proba_vec = proba_mat[i][j]
                    for prob in proba_vec:
                        pred = (prob > threshold).float()
                        if i == j:
                            for x in pred:
                                if x.item() == 1:
                                    TP += 1
                                else:
                                    FN += 1
                        else:
                            for x in pred:
                                if x.item() == 1:
                                    FP += 1
                                else:
                                    TN += 1

                FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
                FRR = FN / (TP + FN) if (TP + FN) > 0 else 0
                model_fars.append(FAR)
                model_frrs.append(FRR)

            all_fars.append(model_fars)
            all_frrs.append(model_frrs)

        avg_fars = np.mean(all_fars, axis=0)
        avg_frrs = np.mean(all_frrs, axis=0)
        eer_threshold_idx = np.argmin(np.abs(avg_fars - avg_frrs))
        eer_threshold = thresholds[eer_threshold_idx]
        eer = (avg_fars[eer_threshold_idx] + avg_frrs[eer_threshold_idx]) / 2


        return eer_threshold, eer

    # Define thresholds to iterate over
    thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]

    # Find the EER threshold and EER value
    eer_threshold, eer = find_eer_threshold(proba_mat, user_ids, thresholds)

    # Save the EER threshold and EER value to a file
    with open(f"{folder_name}/all_model_eer.txt", "w") as eer_file:
        eer_file.write(f"EER Threshold: {eer_threshold:.2f}\n")
        eer_file.write(f"EER: {eer:.4f}\n")

if __name__ == "__main__":
    main()