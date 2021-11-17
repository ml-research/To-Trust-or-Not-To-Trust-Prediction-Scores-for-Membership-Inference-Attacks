import torch


def get_model_prediction_scores(model, apply_softmax, dataset, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prediction_scores = []
    with torch.no_grad():
        model.to(device)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if apply_softmax:
                output = model(x).softmax(dim=1)
            else:
                output = model(x)
            prediction_scores.append(output.cpu())

    prediction_scores = torch.cat(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
    return prediction_scores
