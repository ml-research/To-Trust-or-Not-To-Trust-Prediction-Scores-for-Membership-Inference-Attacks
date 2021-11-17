import torch
import torch.nn as nn


class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_calibrated = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temperature = 0

    def calibrate(self, validation_data, batch_size=128):
        self.model.eval()
        temperature = nn.Parameter(torch.tensor(1.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        args = {'temperature': temperature}
        logits = []
        labels = []
        temperature_values = []
        loss_values = []

        data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits.append(self.model(x))
                labels.append(y)

        logits = torch.cat(logits, dim=0).to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)

        def T_scaling(logits, args):
            temperature = args.get('temperature', None)
            return torch.div(logits, temperature)

        def _eval():
            loss = criterion(T_scaling(logits, args), labels)
            loss.backward()
            temperature_values.append(temperature.item())
            loss_values.append(loss)
            return loss

        optimizer.step(_eval)
        self.temperature = temperature
        print('Best temperature value: {:.2f}'.format(self.temperature.item()))

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x) / self.temperature
            pred_scores = torch.softmax(logits, dim=1)
        return pred_scores
