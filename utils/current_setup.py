try: 
    import data_loader as loader
except:
    from utils import data_loader as loader

kwargs = {
    "mode": loader.LoadMode.ACCENT_AND_CAPITAL_FLAG,
    "epochs_num": 700,
    "rows_per_example": 60,
    "hidden_conv_dim" :64, 
    "hidden_ff_dim": 128,
    "num_layers": 2,
    "use_fc_before": True,
    "threshold": 0.8
}
