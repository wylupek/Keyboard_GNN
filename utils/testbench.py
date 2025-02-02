import multiprocessing as mp
import sys, os, copy
import pandas as pd
import train

from current_setup import kwargs

def run_single_benchmark(user:int, benchmark_file_path):
    """"
    workflow goes like this:
        A model will be trained for a specified user using cross validation, the results will be written to an intermediate file?
    """
    kwargs2 = copy.deepcopy(kwargs)
    kwargs2["user_id"] = user 
    kwargs2.pop("threshold", None) # remove the threshold

    args = pd.DataFrame([kwargs2])

    l, model_summary_df = train.train_with_crossvalidation(
        "../keystroke_data.sqlite", model_path=f"../models/model_5_60_half_neg/{user}.pth", test_train_split=0.2, offset=10, positive_negative_ratio=2,
            **kwargs2)
    

    # aggregate the precission and recall
    precision = sum([i[0] for i in l])/len(l)
    recall = sum([i[1] for i in l])/len(l)
    min_prec = min([i[0] for i in l])
    min_recall = min([i[1] for i in l])
    df2 = pd.DataFrame([(precision, recall, min_prec, min_recall)], columns=["precision", "recall", "min_prec", "min_recall"])

    
    full_report_df = pd.concat([args, model_summary_df, df2], axis=1)

    if os.path.exists(benchmark_file_path):
        full_report_df.to_csv(benchmark_file_path,  mode='a', header=False, index=False)
    else:
        full_report_df.to_csv(benchmark_file_path, header='column_names', index=False)


if __name__ == '__main__':
    users = sys.argv[1:]
    for user in users:
        user = int(user)
        print(f"\n----- Training user {user} --------")
        run_single_benchmark(user, f"user{user}.csv")
        print(f"----- Done user {user} --------\n\n")


    # kwargs = {
    #     "mode": train.LoadMode.MOST_COMMON_ONLY,
    #     "epochs_num": 700,
    #     "rows_per_example": 50,
    #     "hidden_conv_dim" :64, 
    #     "hidden_ff_dim": 128,
    #     "user_id": user,
    #     "num_layers": 2,
    #     "use_fc_before": True
    # }

    # # train the proper model
    # l, model_summary_df = train.train(
    #     "../keystroke_data.sqlite", test_train_split=0, positive_negative_ratio=1, offset=1,
    #         **kwargs)
 
