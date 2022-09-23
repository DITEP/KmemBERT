# Results folder

Here, results are saved after you trained a model, so you can reuse the model and keep its results.

No result folder is pushed on Github.

## Example

This is a content of a result folder

```
.
├── args.json            - args from the command line
├── correlations.png     - graph of label-predictions correlation
├── correlations_distributions.png - same but considering the variance
├── loss.png             - loss evolution
├── losses.json          - all losses
├── mae_distribution.png - MAE distribution over labels
├── results.json         - result metrics
├── checkpoint.pth       - PyTorch state dict
├── stds.png             - std distribution
└── test.json            - predictions and labels on the test set
```

## Checkpoint

Use `checkpoint = torch.load(<path_checkpoint>, map_location=<device>)` to load a checkpoint on a given device.

The checkpoints are composed of the following items.
```
{
    'model': model.state_dict(),   - model state dict
    'best_loss': model.best_loss,  - model best loss on the validation dataset
    'epoch': epoch,                - at which epoch it was saved
    'tokenizer': model.tokenizer   - model tokenizer (could be None)
    'optimizer': model.optimizer.state_dict(), - model optimizer
    'scheduler': model.scheduler.state_dict()  - model scheduler
}
```