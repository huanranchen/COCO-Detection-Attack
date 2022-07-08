import torch
import torch.nn.functional as F

def get_loss(x: torch.tensor, model: torch.nn.Module):
    criterion = lambda x: F.mse_loss(x, torch.tensor([0.0]).cuda())
    predictions = model(x)
    # print(predictions)
    if len(x) == 0:
        return x.detach()
    pred = predictions[0]
    scores = pred["scores"]
    mask = scores > 0.5
    scores = scores[mask]
    loss = criterion(scores)
    return loss.item()


