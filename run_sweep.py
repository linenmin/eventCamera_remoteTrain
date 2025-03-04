import wandb
from tf2_mbNetV2_train import train

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "data_name": {
            "values": ["timeStack1281281"]
        },
        "learning_rate": {
            "values": [1e-3]
        },
        "epochs": {
            "value": 100
        },
        "batch_size": {
            "values": [16, 12, 64, 32,  8]
        },
        "patience": {
            "value": 100
        },
        "min_delta": {
            "value": 0.01
        },
        "dense_dropout": {
            "values": [0.5]
        },
        "dense_l2": {
            "values": [1e-3]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="event_MT_tf2mobileNetV2_sweep")
print("Created sweep with ID:", sweep_id)
wandb.agent(sweep_id, function=train, count=24)
