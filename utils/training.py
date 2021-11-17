import torch
from utils.validation import evaluate
from tqdm.autonotebook import tqdm
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def train_model(
    model,
    dataset,
    optimizer,
    num_epochs,
    batch_size,
    loss_fkt=torch.nn.CrossEntropyLoss(),
    val_dataset=None,
    num_workers=4,
    filename=None,
    lr_scheduler=None,
    rtpt=None,
    wandb=None
):
    model = model.cuda()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        num_correct = 0
        for x, y in tqdm(trainloader, total=len(trainloader), desc=f"Epoch {epoch}", leave=False):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fkt(output, y)
            loss.backward()
            optimizer.step()
            num_correct += (torch.argmax(output.softmax(dim=1), dim=1) == y).sum()

        if lr_scheduler:
            lr_scheduler.step()

        model.eval()
        if wandb:
            wandb.log({"Training Acc": num_correct / len(dataset), "epoch": epoch})

        val_acc = 0
        if val_dataset:
            model.eval()
            val_acc = evaluate(model, val_dataset, batch_size=batch_size)

            if wandb:
                wandb.log({"Test Acc": val_acc, "epoch": epoch})

        print(f'Epoch {epoch}: Training Acc={num_correct/len(dataset):.2f} \t Validation Acc={val_acc:.2f}')

        if rtpt:
            rtpt.step()

    if filename:
        if len(filename.split(os.sep)) > 1 and not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(model.state_dict(), filename)


class EarlyStopper:
    def __init__(self, window, min_diff=0.005):
        self.window = window
        self.best_value = np.inf
        self.current_count = 0
        self.min_diff = min_diff

    def stop_early(self, value):
        if self.best_value <= (value + self.min_diff) and self.current_count >= self.window:
            self.current_count = 0
            return True

        if value < self.best_value and (self.best_value - value) >= self.min_diff:
            self.current_count = 0
            self.best_value = value

        self.current_count += 1

        return False


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Source: https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, output, target):
        pred_score = 1. - self.smoothing
        logprobs = F.log_softmax(output, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = pred_score * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
