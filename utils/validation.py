import torch


def evaluate(model, dataset, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_corrects = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model.to(device)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            num_corrects += (torch.argmax(output, dim=1) == y).float().sum()
    return num_corrects / len(dataset)


def expected_calibration_error(model, dataset, num_bins, apply_softmax, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_score_vectors = []
    targets = []
    for x, y in dataloader:
        with torch.no_grad():
            conf = model(x.cuda())
            if apply_softmax:
                conf = conf.softmax(1)
            pred_score_vectors.append(conf.cpu())
            targets.append(y)

    pred_score_vectors = torch.cat(pred_score_vectors, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    bounds = torch.linspace(0, 1, num_bins + 1)
    lower_bounds = bounds[:-1]
    upper_bounds = bounds[1:]

    max_pred_scores, y_pred = torch.max(pred_score_vectors, 1)
    accuracies = y_pred.eq(targets)

    ece = torch.zeros(1, device='cpu')
    for l_bin, u_bin in zip(lower_bounds, upper_bounds):
        is_in_bin = max_pred_scores.gt(l_bin.cpu().item()) * max_pred_scores.le(u_bin.cpu().item())
        prop = is_in_bin.float().mean()
        if prop.cpu().item() > 0:
            acc = accuracies[is_in_bin].float().mean()
            avg = max_pred_scores[is_in_bin].mean()
            ece += prop * torch.abs(avg - acc)
    return ece.cpu().item()



def overconfidence_error(model, dataset, num_bins, apply_softmax, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_score_vectors = []
    targets = []
    for x, y in dataloader:
        with torch.no_grad():
            conf = model(x.cuda())
            if apply_softmax:
                conf = conf.softmax(1)
            pred_score_vectors.append(conf.cpu())
            targets.append(y)

    pred_score_vectors = torch.cat(pred_score_vectors, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    bounds = torch.linspace(0, 1, num_bins + 1)
    lower_bounds = bounds[:-1]
    upper_bounds = bounds[1:]

    max_pred_scores, y_pred = torch.max(pred_score_vectors, 1)
    accuracies = y_pred.eq(targets)

    oe = torch.zeros(1, device='cpu')
    for l_bin, u_bin in zip(lower_bounds, upper_bounds):
        is_in_bin = max_pred_scores.gt(l_bin.cpu().item()) * max_pred_scores.le(u_bin.cpu().item())
        prop = is_in_bin.float().mean()
        if prop.cpu().item() > 0:
            acc = accuracies[is_in_bin].float().mean()
            conf = max_pred_scores[is_in_bin].mean()
            if conf - acc > 0.0:
                oe += prop * conf * (conf - acc)
    return oe.cpu().item()

