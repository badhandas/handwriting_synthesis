import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import argparse
from utilz import decay_learning_rate, save_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint

class Window(pl.LightningModule):
    def __init__(self, padded_text_len, cell_size, K):
        super(Window, self).__init__()
        self.linear = nn.Linear(cell_size, 3 * K)
        self.padded_text_len = padded_text_len

    def forward(self, x, kappa_old, onehots, text_lens):
        params = self.linear(x).exp()

        alpha, beta, pre_kappa = params.chunk(3, dim=-1)
        kappa = kappa_old + pre_kappa

        indices = torch.from_numpy(np.array(range(self.padded_text_len + 1))).type(torch.FloatTensor)
        # if cuda:
        #     indices = indices.cuda()
        indices = Variable(indices, requires_grad=False).cuda()
        gravity = -beta.unsqueeze(2) * (kappa.unsqueeze(2).repeat(1, 1, self.padded_text_len + 1) - indices) ** 2
        phi = (alpha.unsqueeze(2) * gravity.exp()).sum(dim=1) * (self.padded_text_len / text_lens)

        w = (phi.narrow(-1, 0, self.padded_text_len).unsqueeze(2) * onehots).sum(dim=1)
        return w, kappa, phi


class LSTM1(pl.LightningModule):
    def __init__(self, padded_text_len, vocab_len, cell_size, K):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTMCell(input_size=3 + vocab_len, hidden_size=cell_size)
        self.window = Window(padded_text_len, cell_size, K)

    def forward(self, x, onehots, text_lens, w_old, kappa_old, prev):
        h1s = []
        ws = []
        phis = []
        for _ in range(x.size()[1]):
            cell_input = torch.cat([x.narrow(1, _, 1).squeeze(1), w_old], dim=-1)
            prev = self.lstm(cell_input, prev)

            # attention window parameters
            w_old, kappa_old, old_phi = self.window(prev[0], kappa_old, onehots, text_lens)

            # concatenate for single pass through the next layer
            h1s.append(prev[0])
            ws.append(w_old)

        return torch.stack(ws, dim=0).permute(1, 0, 2), torch.stack(h1s, dim=0).permute(1, 0, 2), \
               prev, w_old, kappa_old, old_phi


class LSTM2(pl.LightningModule):
    def __init__(self, vocab_len, cell_size):
        super(LSTM2, self).__init__()
        self.lstm = nn.LSTM(input_size=3 + vocab_len + cell_size,
                            hidden_size=cell_size, num_layers=1, batch_first=True)

    def forward(self, x, ws, h1s, prev2):
        lstm_input = torch.cat([x, ws, h1s], -1)
        h2s, prev2 = self.lstm(lstm_input, prev2)
        return h2s, prev2


def log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, \
                   y, masks):
    # targets
    y_0 = y.narrow(-1, 0, 1)
    y_1 = y.narrow(-1, 1, 1)
    y_2 = y.narrow(-1, 2, 1)

    # end of stroke prediction
    end_loglik = (y_0 * end + (1 - y_0) * (1 - end)).log().squeeze()

    # new stroke point prediction
    const = 1E-20  # to prevent numerical error
    pi_term = torch.Tensor([2 * np.pi])
    if cuda:
        pi_term = pi_term.cuda()
    pi_term = -Variable(pi_term, requires_grad=False).log()

    z = (y_1 - mu_1) ** 2 / (log_sigma_1.exp() ** 2) \
        + ((y_2 - mu_2) ** 2 / (log_sigma_2.exp() ** 2)) \
        - 2 * rho * (y_1 - mu_1) * (y_2 - mu_2) / ((log_sigma_1 + log_sigma_2).exp())
    mog_lik1 = pi_term - log_sigma_1 - log_sigma_2 - 0.5 * ((1 - rho ** 2).log())
    mog_lik2 = z / (2 * (1 - rho ** 2))
    mog_loglik = ((weights.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1) + const).log()

    return (end_loglik * masks).sum() + ((mog_loglik) * masks).sum()


class LSTMSynthesis(pl.LightningModule):
    def __init__(self, padded_text_len, vocab_len, cell_size, num_clusters, K):
        super().__init__()
        self.lstm1 = LSTM1(padded_text_len, vocab_len, cell_size, K)
        self.lstm2 = LSTM2(vocab_len, cell_size)
        self.linear = nn.Linear(cell_size * 2, 1 + num_clusters * 6)
        self.tanh = nn.Tanh()
        self.learning_rate = 8E-4
        self.h1_init = self.c1_init = torch.zeros((args.batch_size, args.cell_size))
        self.h2_init = self.c2_init = torch.zeros((1, args.batch_size, args.cell_size))
        self.kappa_old = torch.zeros(args.batch_size, args.K)
        cuda = torch.cuda.is_available()
        if cuda:
            self.h1_init, self.c1_init = self.h1_init.cuda(), self.c1_init.cuda()
            self.h2_init, self.c2_init = self.h2_init.cuda(), self.c2_init.cuda()
            self.kappa_old = self.kappa_old.cuda()

    def forward(self, x, onehots, text_lens, w_old, kappa_old, prev, prev2, bias=0.):
        ws, h1s, prev, w_old, kappa_old, old_phi = self.lstm1(x, onehots, text_lens, w_old, kappa_old, prev)
        h2s, prev2 = self.lstm2(x, ws, h1s, prev2)
        params = self.linear(torch.cat([h1s, h2s], dim=-1))
        mog_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights * (1 + bias), dim=-1)
        rho = self.tanh(pre_rho)
        end = torch.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
        return end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w_old, kappa_old, prev, prev2, old_phi

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.h1_init, self.c1_init = Variable(self.h1_init, requires_grad=False), Variable(self.c1_init, requires_grad=False)
        self.h2_init, self.c2_init = Variable(self.h2_init, requires_grad=False), Variable(self.c2_init, requires_grad=False)
        self.kappa_old = Variable(self.kappa_old, requires_grad=False)
        data, masks, onehots, text_lens = train_batch
        step_back = data.narrow(1, 0, args.timesteps)
        x = Variable(step_back, requires_grad=False)
        onehots = Variable(onehots, requires_grad=False)
        masks = Variable(masks, requires_grad=False)
        masks = masks.narrow(1, 0, args.timesteps)
        text_lens = Variable(text_lens, requires_grad=False)

        # focus window weight on first text char
        w_old = onehots.narrow(1, 0, 1).squeeze()
        # feed forward
        outputs = self.forward(x, onehots, text_lens, w_old, self.kappa_old, (self.h1_init, self.c1_init), (
            self.h2_init, self.c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs
        data = data.narrow(1, 1, args.timesteps)
        y = Variable(data, requires_grad=False)
        loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks) / torch.sum(masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        validation_samples, masks, onehots, text_lens = val_batch
        step_back = validation_samples.narrow(1, 0, args.timesteps)
        masks = Variable(masks, requires_grad=False)
        masks = masks.narrow(1, 0, args.timesteps)
        onehots = Variable(onehots, requires_grad=False)
        text_lens = Variable(text_lens, requires_grad=False)
        w_old = onehots.narrow(1, 0, 1).squeeze()
        x = Variable(step_back, requires_grad=False)
        validation_samples = validation_samples.narrow(1, 1, args.timesteps)
        y = Variable(validation_samples, requires_grad=False)
        outputs = self.forward(x, onehots, text_lens, w_old, self.kappa_old, (self.h1_init, self.c1_init), (
            self.h2_init, self.c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs
        loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks) / torch.sum(masks)
        self.log('val_loss', loss)
        # filename = args.task + '_epoch_{}.pt'.format(epoch+1)
        # save_checkpoint(epoch, model, loss, optimizer, args.model_dir, filename)
        return loss


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synthesis',
                    help='"rand_write" or "synthesis"')
parser.add_argument('--cell_size', type=int, default=400,
                    help='size of LSTM hidden state')
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--timesteps', type=int, default=800,
                    help='LSTM sequence length')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--model_dir', type=str, default='/tmp/dontdelete/',
                    help='directory to save model to')
parser.add_argument('--learning_rate', type=float, default=1E-4,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.99,
                    help='lr decay rate for adam optimizer per epoch')
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of gaussian mixture clusters for stroke prediction')
parser.add_argument('--K', type=int, default=10,
                    help='number of attention clusters on text input')
args = parser.parse_args()

# Prepare train data
train_data = [np.load('data/train_strokes_800.npy'), np.load('data/train_masks_800.npy'),
              np.load('data/train_onehot_800.npy'),
              np.load('data/train_text_lens.npy')]
for _ in range(len(train_data)):
    train_data[_] = torch.from_numpy(train_data[_]).type(torch.FloatTensor)
train_data = [(train_data[0][i], train_data[1][i],
               train_data[2][i], train_data[3][i]) for i in range(len(train_data[0]))]
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
validation_data = [np.load('data/validation_strokes_800.npy'), np.load('data/validation_masks_800.npy'),
                   np.load('data/validation_onehot_800.npy'), np.load('data/validation_text_lens.npy')]
for _ in range(len(validation_data)):
    validation_data[_] = torch.from_numpy(validation_data[_]).type(torch.FloatTensor)
validation_data = [(validation_data[0][i], validation_data[1][i], validation_data[2][i], validation_data[3][i])
                   for i in range(len(validation_data[0]))]
validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

padded_text_len, vocab_len = train_loader.dataset[0][2].size()
model = LSTMSynthesis(padded_text_len, vocab_len, args.cell_size, args.num_clusters, args.K)
cuda = torch.cuda.is_available()
epoch = 0
#trainer uncomment below lines for training and comment for checking
# #trainer = pl.Trainer(gpus=1,max_epochs= 30)
# trainer = pl.Trainer(gpus=1,max_epochs=93, resume_from_checkpoint ='/tmp/dontdelete/checkpoint93.ckpt')
# trainer.fit(model, train_loader, validation_loader)
# #trainer.tune(model)
# trainer.save_checkpoint("/tmp/dontdelete/checkpoint.ckpt")
# filename = args.task + '_epoch_{}.pt'.format(epoch+1)
# save_checkpoint(epoch, model, args.model_dir, filename)


