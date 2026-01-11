## Offline Mode
```
export WANDB_MODE=offline
export WANDB_DIR=/path/to/wandb_logs   # 선택 (기본은 ./wandb)
```
or

```
import wandb

wandb.init(
    project="my_project",
    name="exp_001",
    mode="offline"
)
```

## Check in Local Dashboard
```
wandb sync --sync-all --no-upload wandb_logs
http://localhost:8080

```