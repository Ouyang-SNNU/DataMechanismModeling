import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.integrate import odeint


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float64)
torch.manual_seed(6689)
np.random.seed(6689)


class Compartment(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[128, 128, 128, 128, 128, 128, 128, 128, 128], output_size=4):
        super(Compartment, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[3], hidden_sizes[4]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[4], hidden_sizes[5]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[5], hidden_sizes[6]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[6], hidden_sizes[7]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[7], hidden_sizes[8]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[8], output_size),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()
        self.resol = resol
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float64), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class SEIR_Net(nn.Module):
    def __init__(self):
        super(SEIR_Net, self).__init__()
        self._all_layers = []
        self.Compartment = Compartment()

    def forward(self, x):
        x = self.Compartment(x)
        return x


class loss_generator(nn.Module):
    def __init__(self):
        super(loss_generator, self).__init__()
        self.sigma = torch.tensor(0.2, dtype=torch.float64)
        self.gamma = torch.tensor(0.162 + 4.816e-3, dtype=torch.float64)
        self.beta_a = torch.tensor(0.1, dtype=torch.float64)
        self.beta_n = torch.tensor(0.5, dtype=torch.float64)
        self.dt = 1
        self.derivative_t = Conv1dDerivative(DerFilter=[[[-1, 0, 1]]],
                                             resol=2 * self.dt,
                                             kernel_size=3,
                                             name='partial_t').cuda()

    def get_Loss(self, gt, Compartment):
        mse_loss = nn.MSELoss()
        p = Compartment[:, 3:4]
        Compartment = Compartment[:, 0:3]

        I_new = gt[:, 1:2]
        I_cum = gt[:, 0:1]

        R = 1 - Compartment[:, 0:1] - Compartment[:, 1:2] - Compartment[:, 2:3]
        R[R < 0] = 0
        Compartment = torch.cat((Compartment, R), dim=1)
        Compartment_perm = Compartment.permute(1, 0).unsqueeze(0)
        dS_dt = self.derivative_t(Compartment_perm[:, 0:1, :])
        dE_dt = self.derivative_t(Compartment_perm[:, 1:2, :])
        dI_dt = self.derivative_t(Compartment_perm[:, 2:3, :])
        dR_dt = self.derivative_t(Compartment_perm[:, 3:4, :])
        dCompartment_dt = torch.cat((dS_dt, dE_dt, dI_dt, dR_dt), dim=1)
        dCompartment_dt = dCompartment_dt.squeeze(0).permute(1, 0)

        S = Compartment[:, 0:1]
        E = Compartment[:, 1:2]
        I = Compartment[:, 2:3]
        R = Compartment[:, 3:4]

        S = S[1:-1, :]
        E = E[1:-1, :]
        I = I[1:-1, :]
        R = R[1:-1, :]
        p = p[1:-1, :]

        # data_loss
        I_new_nn = self.gamma * Compartment[:, 2:3]
        I_cum_nn = torch.cumsum(I_new_nn, dim=0)
        data_loss = mse_loss(I_new, I_new_nn) + mse_loss(I_cum_nn, I_cum)

        # phy_loss
        f_S = - dCompartment_dt[:, 0:1] - (p * self.beta_n + (1 - p) * self.beta_a) * (1 - E - I - R) * I
        f_E = - dCompartment_dt[:, 1:2] + (p * self.beta_n + (1 - p) * self.beta_a) * (1 - E - I - R) * I - self.sigma * E
        f_I = - dCompartment_dt[:, 2:3] + self.sigma * E - self.gamma * (1 - E - S - R)
        f_R = - dCompartment_dt[:, 3:4] + self.gamma * I
        f_N = 1 - S - E - I - R

        phy_loss = mse_loss(f_S, torch.zeros_like(f_S).cuda()) + \
                   mse_loss(f_E, torch.zeros_like(f_E).cuda()) + \
                   mse_loss(f_I, torch.zeros_like(f_I).cuda()) + \
                   mse_loss(f_R, torch.zeros_like(f_R).cuda()) + \
                   mse_loss(f_N, torch.zeros_like(f_N).cuda())

        return data_loss, phy_loss


# train
def train(model, gt, n_iters, learning_rate, save_path, cont):
    loss_func = loss_generator()
    if cont:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.99)

    for epoch in range(n_iters):
        optimizer.zero_grad()
        Compartment = model(gt)
        loss_data, loss_phy = loss_func.get_Loss(gt, Compartment)
        loss = 1 * loss_data + 5000 * loss_phy
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        loss_tol = loss.item()
        data_loss = loss_data.item()
        phy_loss = loss_phy.item()
        # train_loss_list.append([loss, data_loss, phy_loss])
        print(f'[{epoch + 1:4d}/{n_iters:4d}] '
              f'Loss: {loss_tol: .7e}, '
              f'Data Loss: {data_loss: .7e}, '
              f'Phy Loss: {phy_loss: .7e} '
              )
        if (epoch + 1) % 1000 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), }, save_path + 'Compartment_Net_US.pt')


# load
def load_model(model):
    checkpoint = torch.load('../Model/Compartment_Net_US.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(list(model.parameters()), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.99)
    return model, optimizer, scheduler


if __name__ == '__main__':
    # load data
    df = pd.read_excel('../Data/smoothed data/US_smoothed.xlsx')
    Data = df.to_numpy()
    gt_train = Data[0:900, 3:]
    if gt_train.dtype == np.object_:
        gt_train = gt_train.astype(float)
    gt_train = torch.tensor(gt_train).cuda()

    # parameters of differential equations
    N = 331002647                  # total population
    gamma = 0.162 + 4.816e-3       # remove rate
    sigma = 0.2                    # incubation rate
    beta_a = 0.1                   # baseline transmission rate

    # parameters of training
    n_iters = 100000               # training iterations
    lr = 1e-3                      # initial learning rate
    save_path = '../Model/'

    # scaling: normalize values for model training
    gt_train[:, 0:2] = gt_train[:, 0:2] / N
    gt_train[:, 2:3] = gt_train[:, 2:3] / 900

    SEIR_Net = SEIR_Net().cuda()

    start = time.time()
    train(SEIR_Net, gt_train, n_iters, lr, save_path, False)
    # additional training epochs on a previously trained model, though not used here
    # train(SEIR_Net, gt_train, n_iters, lr, save_path, True)
    end = time.time()
    print('The training time is: ', (end - start))

    Compartment = SEIR_Net(gt_train)

    p = Compartment[:, 3:4]
    Compartment = Compartment[:, 0:3]

    Compartment = Compartment.detach().cpu().numpy()
    gt_train = gt_train.detach().cpu().numpy()

    S = Compartment[:, 0]
    E = Compartment[:, 1]
    I = Compartment[:, 2]
    R = 1 - S - E - I
    R[R < 0] = 0

    # inverse scaling: convert normalized outputs back to original scale
    I_new_nn = gamma * I
    I_new_nn = N * I_new_nn
    I_cum_nn = np.cumsum(I_new_nn)
    gt_train = gt_train * N
    I_cum_real = gt_train[:, 0]
    I_new_real = gt_train[:, 1]

    loss_func = loss_generator()
    p_perm = p.permute(1, 0).unsqueeze(0)
    dp_dt = loss_func.derivative_t(p_perm)
    dp_dt = dp_dt.squeeze(0).permute(1, 0)
    p = p.detach().cpu().numpy()
    dp_dt = dp_dt.detach().cpu().numpy()
    dp_dt = np.concatenate((p[1:2] - p[0:1], dp_dt, p[-1:] - p[-2:-1]), axis=0)
    f = dp_dt / (p * (1 - p))

    # save data
    data = np.column_stack((S, E, I, R, p, I_new_nn, I_cum_nn, I_new_real, I_cum_real, f))
    np.savetxt('../Data/Compartment_US.csv', data, delimiter=',', header='S,E,I,R,p,I_new_nn,I_cum_nn,I_new_real,I_cum_real,f', comments='', fmt='%.6f')


    # fine-tuning of beta_n
    def p(t):
        idx = min(int(round(t)), len(p_values) - 1)
        return p_values[idx]


    def SEIR(y, t, beta_n, beta_a, sigma, gamma):
        S, E, I, R = y
        dS = -(p(t) * beta_n + (1 - p(t)) * beta_a) * S * I / N
        dE = (p(t) * beta_n + (1 - p(t)) * beta_a) * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]


    data = pd.read_csv('../Data/Compartment_US.csv', header=0).values
    S, E, I, R, p_values, I_new_nn, I_cum_nn, I_new_real, I_cum_real, f = data.T

    # initial conditions
    S0, E0, I0, R0 = S[0], E[0], I[0], 1 - S[0] - E[0] - I[0]
    initial_conditions = [S0 * N, E0 * N, I0 * N, R0 * N]

    tspan = np.arange(len(S))


    def objective(beta_n):
        def SEIR_variable_beta(y, t):
            return SEIR(y, t, beta_n, beta_a, sigma, gamma)

        try:
            sol = odeint(SEIR_variable_beta, initial_conditions, tspan)
        except ValueError as e:
            raise ValueError(f"ODE integration failed: {e}")

        I_new_ODE = gamma * sol[:, 2]

        mse = np.mean((I_new_ODE - I_new_real) ** 2)
        return mse


    beta_n_values = np.linspace(0.45, 0.55, 100)
    best_beta_n = None
    best_mse = float('inf')

    for beta_n in beta_n_values:
        mse = objective(beta_n)
        if mse < best_mse:
            best_mse = mse
            best_beta_n = beta_n

    print(f"best_beta_n: {best_beta_n}")