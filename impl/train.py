import torch


def train(optimizer, modelx, dl, loss_fn):
    modelx.train()
    total_loss = []
    for batch in dl:
        optimizer.zero_grad()
        pred = modelx(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dl, metrics, loss_fn=None):
    model.eval()
    preds = []
    ys = []
    for batch in dl:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
