import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import scipy.io
import time
import matplotlib
matplotlib.use('Agg')
import random
def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):
    def __init__(self, X_initial, X_f_train, X_lb, X_ub, Y_lb, Y_ub, Z_lb, Z_ub, layers, lb, ub, model_para):
        super(PINN, self).__init__()
        
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.x0 = torch.tensor(X_initial[:, 0:1], requires_grad=True).float().to(device)
        self.y0 = torch.tensor(X_initial[:, 1:2], requires_grad=True).float().to(device)
        self.z0 = torch.tensor(X_initial[:, 2:3], requires_grad=True).float().to(device)
        self.t0 = torch.tensor(X_initial[:, 3:4], requires_grad=True).float().to(device)
        self.u0 = torch.tensor(X_initial[:, 4:5]).float().to(device)

        self.x_lb_x = torch.tensor(X_lb[:, 0:1], requires_grad=True).float().to(device)
        self.x_lb_y = torch.tensor(X_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_lb_z = torch.tensor(X_lb[:, 2:3], requires_grad=True).float().to(device)
        self.x_lb_t = torch.tensor(X_lb[:, 3:4], requires_grad=True).float().to(device)
        self.x_lb_u = torch.tensor(X_lb[:, 4:5]).float().to(device)

        self.x_ub_x = torch.tensor(X_ub[:, 0:1], requires_grad=True).float().to(device)
        self.x_ub_y = torch.tensor(X_ub[:, 1:2], requires_grad=True).float().to(device)
        self.x_ub_z = torch.tensor(X_ub[:, 2:3], requires_grad=True).float().to(device)
        self.x_ub_t = torch.tensor(X_ub[:, 3:4], requires_grad=True).float().to(device)       
        self.x_ub_u = torch.tensor(X_ub[:, 4:5]).float().to(device)

        self.y_lb_x = torch.tensor(Y_lb[:, 0:1], requires_grad=True).float().to(device)
        self.y_lb_y = torch.tensor(Y_lb[:, 1:2], requires_grad=True).float().to(device)
        self.y_lb_z = torch.tensor(Y_lb[:, 2:3], requires_grad=True).float().to(device)
        self.y_lb_t = torch.tensor(Y_lb[:, 3:4], requires_grad=True).float().to(device)
        self.y_lb_u = torch.tensor(Y_lb[:, 4:5]).float().to(device)
        
        self.y_ub_x = torch.tensor(Y_ub[:, 0:1], requires_grad=True).float().to(device)
        self.y_ub_y = torch.tensor(Y_ub[:, 1:2], requires_grad=True).float().to(device)
        self.y_ub_z = torch.tensor(Y_ub[:, 2:3], requires_grad=True).float().to(device)
        self.y_ub_t = torch.tensor(Y_ub[:, 3:4], requires_grad=True).float().to(device)
        self.y_ub_u = torch.tensor(Y_ub[:, 4:5]).float().to(device)
        
        self.z_lb_x = torch.tensor(Z_lb[:, 0:1], requires_grad=True).float().to(device)
        self.z_lb_y = torch.tensor(Z_lb[:, 1:2], requires_grad=True).float().to(device)
        self.z_lb_z = torch.tensor(Z_lb[:, 2:3], requires_grad=True).float().to(device)
        self.z_lb_t = torch.tensor(Z_lb[:, 3:4], requires_grad=True).float().to(device)
        self.z_lb_u = torch.tensor(Z_lb[:, 4:5]).float().to(device)

        self.z_ub_x = torch.tensor(Z_ub[:, 0:1], requires_grad=True).float().to(device)
        self.z_ub_y = torch.tensor(Z_ub[:, 1:2], requires_grad=True).float().to(device)
        self.z_ub_z = torch.tensor(Z_ub[:, 2:3], requires_grad=True).float().to(device)
        self.z_ub_t = torch.tensor(Z_ub[:, 3:4], requires_grad=True).float().to(device)
        self.z_ub_u = torch.tensor(Z_ub[:, 4:5]).float().to(device)
        
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.z_f = torch.tensor(X_f_train[:, 2:3], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 3:4], requires_grad=True).float().to(device)
        
        self.act = nn.Tanh()
        
        self.linears = nn.ModuleList([nn.Linear(layers[l],layers[l+1]) for l in range(len(layers)-1)])
        
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        if model_para != "test":
            for i in range(len(layers)-1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1)
                nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self,x):
        if torch.is_tensor(x) !=True:
            x = torch.from_numpy(x)

        x = 2.0 * (x-self.lb)/(self.ub-self.lb) - torch.tensor(1.0).to(device)

        for linear in self.linears[:-1]:
            x = self.act(linear(x))

        x = self.linears[-1](x)

        return x

    def Net_u(self, x, y, z, t):

        u = self.forward(torch.cat([x, y, z, t],dim=1))

        return u

    def Net_f(self,x, y, z, t):
        u = self.Net_u(x, y, z, t)

        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]
        
        u_xt = torch.autograd.grad(
            u_x, t, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]
        
        u_xxx = torch.autograd.grad(
            u_xx, x, 
            grad_outputs=torch.ones_like(u_xx),
            retain_graph=True,
            create_graph=True)[0]
        u_xxxx = torch.autograd.grad(
            u_xxx, x, 
            grad_outputs=torch.ones_like(u_xxx),
            retain_graph=True,
            create_graph=True)[0]
        
        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True)[0]

        u_z = torch.autograd.grad(
            u, z, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        u_zz = torch.autograd.grad(
            u_z, z, 
            grad_outputs=torch.ones_like(u_z),
            retain_graph=True,
            create_graph=True)[0]
              
        alpha = -0.1
        beta = -4
        r = 0.5
        
        f = u_xt + beta*u_x**2 + alpha*u_xxxx + beta*u*u_xx + r*u_yy + r*u_zz
        
        return f
    
    def loss(self):

        u0_pred = self.Net_u(self.x0, self.y0, self.z0, self.t0)
        x_lb_u_pred = self.Net_u(self.x_lb_x, self.x_lb_y, self.x_lb_z, self.x_lb_t)
        x_ub_u_pred = self.Net_u(self.x_ub_x, self.x_ub_y, self.x_ub_z, self.x_ub_t)
        y_lb_u_pred = self.Net_u(self.y_lb_x, self.y_lb_y, self.y_lb_z, self.y_lb_t)
        y_ub_u_pred = self.Net_u(self.y_ub_x, self.y_ub_y, self.y_ub_z, self.y_ub_t)
        z_lb_u_pred = self.Net_u(self.z_lb_x, self.z_lb_y, self.z_lb_z, self.z_lb_t)
        z_ub_u_pred = self.Net_u(self.z_ub_x, self.z_ub_y, self.z_ub_z, self.z_ub_t)
        f_pred = self.Net_f(self.x_f, self.y_f, self.z_f, self.t_f)
        
        loss_0 = torch.mean((u0_pred - self.u0)**2)
        
        loss_b = torch.mean((x_lb_u_pred-self.x_lb_u)**2) +\
                 torch.mean((x_ub_u_pred-self.x_ub_u)**2) +\
                 torch.mean((y_lb_u_pred-self.y_lb_u)**2) +\
                 torch.mean((y_ub_u_pred-self.y_ub_u)**2) +\
                 torch.mean((z_lb_u_pred-self.z_lb_u)**2) +\
                 torch.mean((z_ub_u_pred-self.z_ub_u)**2)
        
        loss_f = torch.mean(f_pred**2)

        return loss_0, loss_b, loss_f

    def predict(self, X_star):

        x = torch.tensor(X_star[:,0:1]).float().to(device)
        y = torch.tensor(X_star[:,1:2]).float().to(device)
        z = torch.tensor(X_star[:,2:3]).float().to(device)
        t = torch.tensor(X_star[:,3:4]).float().to(device)

        u = self.Net_u(x,y,z,t)
        
        u = u.cpu().detach().numpy()

        return u
    
if __name__ == "__main__":

    # Domain bounds
    lb = np.array([-5.0, -5.0, -0.5, 0.0])#x,y,z,t
    ub = np.array([5.0, 5.0, 0.5, 1.0])

    N0 = 3000
    N_b = 2000
    N_f = 20000
    layers = [4, 100, 100, 1]

    data = scipy.io.loadmat('Data_KP.mat')

    tt = data['tnew'].flatten()[:, None]
    xx = data['xnew'].flatten()[:, None]
    yy = data['ynew'].flatten()[:, None]
    zz = data['znew'].flatten()[:, None]
    Exact = data['unew']
    
    X_f_train = lb + (ub - lb) * lhs(4, N_f)

    X, Y, Z, T = np.meshgrid(xx, yy, zz, tt)
    Exact = np.float32(Exact)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.transpose(1, 0, 2, 3).flatten()[:, None]
    
    # initialize
    t = np.ones(len(xx) * len(yy) * len(zz)) * lb[3]
    X1, Y1, Z1 = np.meshgrid(xx, yy, zz)
    x_initial = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None], Z1.flatten()[:, None], t.flatten()[:, None]))
    u0 = Exact[:, :, :, 0].transpose(1, 0, 2).flatten()[:, None]

    idx0 = np.random.choice(x_initial.shape[0], N0, replace=False)
    x_initial = x_initial[idx0, :]
    U0 = u0[idx0, :]

    X_initial = np.hstack((x_initial, U0))
    # x low boundary
    x_lb_x = np.ones(len(yy) * len(zz) * len(tt)) * lb[0]
    x_lb_y, x_lb_z, x_lb_t = np.meshgrid(yy, zz, tt)
    x_lb = np.hstack((x_lb_x.flatten()[:, None], x_lb_y.flatten()[:, None], x_lb_z.flatten()[:, None], x_lb_t.flatten()[:, None]))
    x_lb_u = Exact[0, :, :, :].transpose(1, 0, 2).flatten()[:, None]

    idx1 = np.random.choice(x_lb.shape[0], N_b, replace=False)
    xx_lb = x_lb[idx1, :]
    xx_lb_u = x_lb_u[idx1, :]
    
    X_lb = np.hstack((xx_lb, xx_lb_u))
    # x upper boundary
    x_ub_x = np.ones(len(yy) * len(zz) * len(tt)) * ub[0]
    x_ub_y, x_ub_z, x_ub_t = np.meshgrid(yy, zz, tt)
    x_ub = np.hstack((x_ub_x.flatten()[:, None], x_ub_y.flatten()[:, None], x_ub_z.flatten()[:, None], x_ub_t.flatten()[:, None]))
    x_ub_u = Exact[-1, :, :, :].transpose(1, 0, 2).flatten()[:, None]

    idx2 = np.random.choice(x_ub.shape[0], N_b, replace=False)
    xx_ub = x_ub[idx2, :]
    xx_ub_u = x_ub_u[idx2, :]
    
    X_ub = np.hstack((xx_ub, xx_ub_u))
        
    # y low boundary
    y_lb_y = np.ones(len(xx) * len(zz) * len(tt)) * lb[1]
    y_lb_x, y_lb_z, y_lb_t = np.meshgrid(xx, zz, tt)
    y_lb = np.hstack((y_lb_x.flatten()[:, None], y_lb_y.flatten()[:, None], y_lb_z.flatten()[:, None], y_lb_t.flatten()[:, None]))
    y_lb_u = Exact[:, 0, :, :].transpose(1, 0, 2).flatten()[:, None]

    idx3 = np.random.choice(x_ub.shape[0], N_b, replace=False)
    yy_lb = y_lb[idx3, :]
    yy_lb_u = y_lb_u[idx3, :]
    
    Y_lb = np.hstack((yy_lb, yy_lb_u))

    # y upper boundary
    y_ub_y = np.ones(len(xx) * len(zz) * len(tt)) * ub[1]
    y_ub_x, y_ub_z, y_ub_t = np.meshgrid(xx, zz, tt)
    y_ub = np.hstack((y_ub_x.flatten()[:, None], y_ub_y.flatten()[:, None], y_ub_z.flatten()[:, None], y_ub_t.flatten()[:, None]))
    y_ub_u = Exact[:, -1, :, :].transpose(1, 0, 2).flatten()[:, None]

    idx4 = np.random.choice(y_ub.shape[0], N_b, replace=False)
    yy_ub = y_ub[idx4, :]
    yy_ub_u = y_ub_u[idx4, :]
    
    Y_ub = np.hstack((yy_ub, yy_ub_u))

    # z low boundary
    z_lb_z = np.ones(len(xx) * len(yy) * len(tt)) * lb[2]
    z_lb_x, z_lb_y, z_lb_t = np.meshgrid(xx, yy, tt)
    z_lb = np.hstack((z_lb_x.flatten()[:, None], z_lb_y.flatten()[:, None], z_lb_z.flatten()[:, None], z_lb_t.flatten()[:, None]))
    z_lb_u = Exact[:, :, 0, :].transpose(1, 0, 2).flatten()[:, None]

    idx5 = np.random.choice(z_lb.shape[0], N_b, replace=False)
    zz_lb = z_lb[idx5, :]
    zz_lb_u = z_lb_u[idx5, :]
    
    Z_lb = np.hstack((zz_lb, zz_lb_u))

    # z upper boundary
    z_ub_z = np.ones(len(xx) * len(yy) * len(tt)) * ub[2]
    z_ub_x, z_ub_y, z_ub_t = np.meshgrid(xx, yy, tt)
    z_ub = np.hstack((z_ub_x.flatten()[:, None], z_ub_y.flatten()[:, None], z_ub_z.flatten()[:, None], z_ub_t.flatten()[:, None]))
    z_ub_u = Exact[:, :, -1, :].transpose(1, 0, 2).flatten()[:, None]

    idx6 = np.random.choice(z_ub.shape[0], N_b, replace=False)
    zz_ub = z_ub[idx6, :]
    zz_ub_u = z_ub_u[idx6, :]
    
    Z_ub = np.hstack((zz_ub, zz_ub_u))
    
    model_para = "train"
    max_iter = 20001
    
    Rates = ['cons', 'piece', 'exp', 'cos']
    weights = ['lossw']

    loss_mat = [[i for i in range(len(Rates))] for i in range(len(weights))] 
    time_mat = [[i for i in range(len(Rates))] for i in range(len(weights))]  
    erru_mat = [[i for i in range(len(Rates))] for i in range(len(weights))]  
    LosswLam_mat = [[i for i in range(len(Rates))] for i in range(1)] 

    for weight in weights:
        row = weights.index(weight)

        for rate in Rates:
            col = Rates.index(rate)
            set_random_seed(1)
            models = PINN(X_initial, X_f_train, X_lb, X_ub, Y_lb, Y_ub, Z_lb, Z_ub, layers, lb, ub, model_para).to(device)
            optimizer = torch.optim.Adam(models.parameters(), lr=0.001)  

            if rate=="cons":
                lambda1 = lambda epoch: 0.1
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            elif rate=="piece":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.4, last_epoch=-1)
            elif rate=="exp":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997)
            elif rate=="cos":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5000, eta_min=0)
            
            loss_w = 1.0
            alpha = 0.9

            start_time = time.time()

            loss_w_rec = []

            loss_rec = []

            for epoch in range(max_iter):
                
                optimizer.zero_grad()     

                loss0, lossb, lossf = models.loss()

                loss = loss0 + loss_w*lossb + lossf
                if epoch%10 == 0:
                    w = lossb.item()/loss0.item()

                    loss_w_rec.append(loss_w)

                    loss_w = w*(1-alpha) + loss_w*alpha

                    loss_rec.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if epoch%5000 == 0:
                    print('epoch: {}, Weight: {}, Rate: {}'.format(epoch, weight, rate))
                
            elapsed = time.time() - start_time
            time_mat[row][col] = elapsed
            loss_mat[row][col] = loss_rec
            
            torch.save(models.state_dict(),'{}{}_model_parameters.pth'.format(weight,rate))

            ##predicted solution
            models.eval()
            with torch.no_grad():
                u_pre = models.predict(X_star)
                
            error_u = np.linalg.norm(u_star - u_pre, 2) / np.linalg.norm(u_star, 2)
            
            erru_mat[row][col] = error_u
            
            print('Relative L2: {:.4e}, Time: {:.3f}'.format(error_u, elapsed))

