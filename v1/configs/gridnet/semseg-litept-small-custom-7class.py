_base_ = ["semseg-litept-small-gridnet-stage-a.py"]

save_path = __import__("os").environ.get("PCDSEG_SAVE_PATH", "exp/litept_small_custom_7class")
weight = __import__("os").environ.get("PCDSEG_INIT_WEIGHT", "")
epoch = int(__import__("os").environ.get("PCDSEG_EPOCHS", "100"))
eval_epoch = 5
batch_size = int(__import__("os").environ.get("PCDSEG_BATCH_SIZE", "8"))
num_worker = int(__import__("os").environ.get("PCDSEG_WORKERS", "4"))

data_root = __import__("os").environ.get("PCDSEG_DATA", "data/custom_7class")
data = dict(
    train=dict(split="train_final.json", data_root=data_root, loop=1),
    val=dict(split="val_final.json", data_root=data_root),
    test=dict(split="val_final.json", data_root=data_root),
)
