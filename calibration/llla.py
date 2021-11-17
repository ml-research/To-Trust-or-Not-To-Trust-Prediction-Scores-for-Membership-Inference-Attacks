from typing import Tuple
import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import KFAC
from tqdm import tqdm
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader


class LLLA(nn.Module):
    """
    """
    def __init__(self, model: nn.Module, last_layer: nn.Module):
        """
        Source: https://github.com/wiseodd/last_layer_laplace
        Some modifications were applied to implement LLLA as nn.Module subclass

        BSD 3-Clause License

        Copyright (c) 2020, Agustinus Kristiadi
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """

        super().__init__()
        self.model = model
        self.last_layer = last_layer
        self.loss_fkt = nn.CrossEntropyLoss()
        self.n_classes = last_layer.out_features

    def forward(
        self,
        x: torch.Tensor,
        M_W: torch.Tensor = None,
        M_b: torch.Tensor = None,
        U: torch.Tensor = None,
        V: torch.Tensor = None,
        B: torch.Tensor = None,
        n_samples: int = 100,
        delta: float = 1.0,
        apply_softmax: bool = True
    ):
        """Compute Bayesian prediction following algorithm 2 lines 12-21.
        """
        if M_W is None:
            M_W = self.M_W
        if M_b is None:
            M_b = self.M_b
        if U is None:
            U = self.U
        if V is None:
            V = self.V
        if B is None:
            B = self.B

        py = []

        # Compute input of the final linear layer using a forward hook
        phi_container = {}

        def phi_hook_fn(module, input, output):
            phi_container[0] = input[0]

        handle = self.last_layer.register_forward_hook(phi_hook_fn)
        self.model.forward(x)
        handle.remove()
        phi = phi_container[0]
        # Build posterior Gaussian distribution p(f|D)
        mu_pred = phi @ M_W + M_b
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0) + B.unsqueeze(0)

        scale_tril = torch.cholesky(Cov_pred)
        post_pred = MultivariateNormal(mu_pred, scale_tril=scale_tril)

        # Monte-Carlo integration with n_samples posterior samples for predictive distribution over f_mu.
        py_ = 0
        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1) if apply_softmax else f_s
        py_ /= n_samples
        py.append(py_)

        return torch.cat(py, dim=0)

    def estimate_parameters(
        self, train_loader, val_loader, ood_loader=None, interval=torch.logspace(-4, 2, 100), lam=0.25, var0=None
    ):
        """Estimates the parameters of the Laplacian approximation.
        Args:
            train_loader (DataLoader): Training data used to compute hessians.
            val_loader (DataLoader): Validation data used to estimate variance.
            ood_loader (DataLoader): OOD data used to estimate variance.
            interval (torch.Tensor, optional): Search interval for var0. Defaults to torch.logspace(-4, 2, 100).
            lam (float, optional): Balancing factor. Defaults to 0.25.
        """
        # set the model to eval
        was_training = self.model.training
        self.model.eval()

        # Compute hessians using training data
        hessians = self.get_hessian(train_loader)

        # Estimate var0 using validation and OOD data
        if var0:
            self.var0 = torch.tensor(var0)
        else:
            self.var0 = self.gridsearch_var0(hessians, val_loader, ood_loader, interval, lam)
        print(f'Estimated var0={self.var0}')

        # Compute Kronecker factors with estimated var0
        self.M_W, self.M_b, self.U, self.V, self.B = self.estimate_variance(self.var0, hessians)

        # restore the state of the model
        if was_training:
            self.model.train()

    def get_hessian(self, train_loader):
        """Computes hessians on a dataset.
        Args:
            train_loader (DataLoader): Data to compute hessians
        Returns:
            list: List of tensors with computed hessian
        """
        W = list(self.model.parameters())[-2]
        b = list(self.model.parameters())[-1]
        m, n = W.shape

        extend(self.loss_fkt, debug=False)
        extend(self.last_layer, debug=False)

        with backpack(KFAC()):
            U, V = torch.zeros(m, m, device=self.model.device), torch.zeros(n, n, device=self.model.device)
            B = torch.zeros(m, m, device=self.model.device)

            for i, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(self.model.device), y.to(self.model.device)

                self.model.zero_grad()
                loss = self.loss_fkt(self.model(x), y)
                loss.backward()

                with torch.no_grad():
                    # Hessian of weight
                    U_, V_ = W.kfac
                    B_ = b.kfac[0]
                    # Running average weighting
                    rho = min(1 - 1 / (i + 1), 0.95)
                    # Take running average of the Kronecker factors
                    U = rho * U + (1 - rho) * U_
                    V = rho * V + (1 - rho) * V_
                    B = rho * B + (1 - rho) * B_

        n_data = len(train_loader.dataset)

        M_W = W.t()
        M_b = b
        U = math.sqrt(n_data) * U
        V = math.sqrt(n_data) * V
        B = n_data * B

        return [M_W, M_b, U, V, B]

    def estimate_variance(self, var0, hessians, invert=True):
        """Compute Kronecker factors following algorithm 2.
        Args:
            var0 (torch.Tensor): Estimated Var0
            hessians (list of torch.Tensor): Computed hessians
            invert (bool, optional): Should hessians be inverted. Defaults to True.
        Returns:
            [type]: [description]
        """
        device = hessians[0].device
        if not invert:
            return hessians

        tau = 1 / var0  # Prior precision tau_0

        with torch.no_grad():
            M_W, M_b, U, V, B = hessians

            m, n = U.shape[0], V.shape[0]

            # Add priors to Kronecker factors
            U_ = U + torch.sqrt(tau) * torch.eye(m, device=device)
            V_ = V + torch.sqrt(tau) * torch.eye(n, device=device)
            B_ = B + tau * torch.eye(m, device=device)

            # Covariances for Laplace
            U_inv = torch.inverse(V_)  # Interchanged since W is transposed
            V_inv = torch.inverse(U_)
            B_inv = torch.inverse(B_)

        return [M_W, M_b, U_inv, V_inv, B_inv]

    def gridsearch_var0(
        self,
        hessians: torch.tensor,
        val_loader: torch.utils.data.DataLoader,
        ood_loader: torch.utils.data.DataLoader,
        interval: torch.tensor = torch.logspace(-4, 2, 100),
        lam: float = 0.25
    ):
        """One-parameter optimization for var0. Corresponds to equation 8 in the paper.
        Args:
            hessians (torch.tensor): Computed hessians.
            val_loader (torch.utils.data.DataLoader): Validation data
            ood_loader (torch.utils.data.DataLoader): OOD data
            interval (torch.tensor, optional): Search interval for var0. Defaults to torch.logspace(-4, 2, 100).
            lam (float, optional): Balancing factor. Defaults to 0.25.
        Returns:
            torch.Tensor: Estimation for var0
        """
        targets = torch.cat([y for x, y in val_loader], dim=0).to(self.model.device)
        vals, var0s = [], []
        pbar = tqdm(interval)

        for var0 in pbar:
            with torch.no_grad():
                M_W, M_b, U, V, B = self.estimate_variance(var0, hessians)

                preds = self.predict(val_loader, M_W, M_b, U, V, B, n_samples=10)
                preds_out = self.predict(ood_loader, M_W, M_b, U, V, B, n_samples=10)

                # Standard cross-entropy loss over validation set
                loss_in = F.nll_loss(torch.log(preds + 1e-8), targets)
                # Negative prediction entropy over OOD distribution
                loss_out = -torch.log(preds_out + 1e-8).mean()
                loss = loss_in + lam * loss_out

                vals.append(loss)
                var0s.append(var0)

                pbar.set_description(
                    f'var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}'
                )

        best_var0 = var0s[torch.tensor(vals).argmin()].cpu()

        return best_var0

    @torch.no_grad()
    def predict(
        self,
        dataloader,
        M_W: torch.Tensor = None,
        M_b: torch.Tensor = None,
        U: torch.Tensor = None,
        V: torch.Tensor = None,
        B: torch.Tensor = None,
        n_samples: int = 100,
        delta: float = 1.0,
        apply_softmax=True
    ):
        """Compute Bayesian prediction following algorithm 2 lines 12-21
        """
        predictions = []
        for x, y in dataloader:
            x = x.to(self.model.device)
            predictions.append(self.forward(x, M_W, M_b, U, V, B, n_samples, delta, apply_softmax=True))
        predictions = torch.cat(predictions, dim=0)

        return predictions

    @staticmethod
    def build_ood_loader(sample_shape: Tuple, size: int, batch_size: int, delta: float = 1.0, distr: str = 'uniform'):
        """Creates a data loader with OOD samples
        Args:
            sample_shape (tuple): Shape of each OOD sample.
            size (int): Size of the dataset.
            batch_size (int): Batch size.
            delta (float): Scaling factor. Defaults to 1.0.
            distr (str, optional): Uniform to sample from. Defaults to uniform.
        Returns:
            DataLoader: OOD data loader
        """
        if distr == 'uniform':
            data = delta * torch.rand((size, ) + sample_shape)
        elif distr == 'normal':
            data = delta * torch.randn((size, ) + sample_shape)
        else:
            raise RuntimeError(f'Invalid distribution {distr}. Choose one of \'uniform\' or \'normal\'.')
        dataset = torch.utils.data.TensorDataset(data, torch.zeros_like(data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    def evaluate(self, evaluation_data, batch_size=128, criterion=nn.CrossEntropyLoss(), dataloader_num_workers=3):
        evalloader = DataLoader(
            evaluation_data, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers, pin_memory=True
        )
        self.eval()
        with torch.no_grad():
            running_loss = torch.tensor(0.0)
            running_mean_max_pred_score = 0
            num_corrects = 0
            for inputs, labels in tqdm(evalloader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                model_output = self.forward(inputs)
                num_corrects += (torch.argmax(model_output, dim=1).squeeze() == labels.squeeze()).sum()
                running_loss += criterion(model_output, labels).cpu()

                running_mean_max_pred_score += self.get_mean_max_prediction_score(model_output)

            acc = num_corrects / len(evaluation_data)

            return acc, running_loss.item() / len(evaluation_data), running_mean_max_pred_score / len(evalloader)

    @staticmethod
    def get_mean_max_prediction_score(logits, reduce_mean=True):
        # calculate the mean maximum prediction score (MMPS) of the validation batch
        prediction_scores = torch.nn.functional.softmax(logits, dim=1)
        max_prediction_scores, indices = torch.max(prediction_scores, dim=1)

        result = max_prediction_scores.cpu()

        if reduce_mean:
            return result.mean().item()
        else:
            return result
