import numpy as np
import torch
from torch.autograd import Variable
from utilz import plot_stroke
# from model import LSTMRandWriter, LSTMSynthesis
from lit import LSTMSynthesis
import matplotlib.pyplot as plt
#import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# find gpu 
cuda = torch.cuda.is_available()


def generate_conditionally(text, cell_size=400, num_clusters=20, K=10, random_state=700, \
                           bias=3., bias2=1.,
                           state_dict_file='/tmp/dontdelete/synthesis_epoch_91.pt'):
    char_to_code = torch.load('char_to_code.pt')
    np.random.seed(random_state)
    text = text + ' '

    model = LSTMSynthesis(len(text), len(char_to_code)+1, cell_size, num_clusters, K)
    model.load_state_dict(torch.load(state_dict_file)['model'])
    #model_test = LSTMSynthesis.load_from_checkpoint(state_dict_file)
    # model = torch.load(state_dict_file)
    #model = ModelCheckpoint('lightning_logs/version_16/checkpoints/epoch=8-step=890.ckpt')
    onehots = np.zeros((len(text), len(char_to_code) + 1))
    for _ in range(len(text)):
        try:
            onehots[_][char_to_code[text[_]]] = 1
        except:
            onehots[_][-1] = 1

    zero_tensor = torch.zeros((1, 1, 3))
    h1_init, c1_init = torch.zeros((1, cell_size)), torch.zeros((1, cell_size))
    h2_init, c2_init = torch.zeros((1, 1, cell_size)), torch.zeros((1, 1, cell_size))
    kappa_old = torch.zeros(1, K)
    onehots = torch.from_numpy(onehots).type(torch.FloatTensor)
    text_len = torch.from_numpy(np.array([[len(text)]])).type(torch.FloatTensor)

    if cuda:
        model.cuda()
        zero_tensor = zero_tensor.cuda()
        h1_init, c1_init = h1_init.cuda(), c1_init.cuda()
        h2_init, c2_init = h2_init.cuda(), c2_init.cuda()
        kappa_old = kappa_old.cuda()
        onehots = onehots.cuda()
        text_len = text_len.cuda()

    x = Variable(zero_tensor)
    h1_init, c1_init = Variable(h1_init), Variable(c1_init)
    h2_init, c2_init = Variable(h2_init), Variable(c2_init)
    prev = (h1_init, c1_init)
    prev2 = (h2_init, c2_init)
    kappa_old = Variable(kappa_old)
    onehots = Variable(onehots, requires_grad=False)
    w_old = onehots.narrow(0, 0, 1)  # attention on the first input text char
    text_len = Variable(text_len)

    record = [np.zeros(3)]
    phis = []
    stop = False
    count = 0
    while not stop:
        outputs = model(x, onehots, text_len, w_old, kappa_old, prev, prev2, bias)
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w_old, kappa_old, prev, prev2, old_phi = outputs

        # bernoulli sample
        prob_end = end.data[0][0][0].cpu()
        sample_end = np.random.binomial(1, prob_end)

        sample_index = np.random.choice(range(20), p=weights.data[0][0].cpu().numpy())
        mu = np.array([mu_1.data[0][0][sample_index], mu_2.data[0][0][sample_index]], dtype=float)
        log_sigma_1 = log_sigma_1 - bias2
        log_sigma_2 = log_sigma_2 - bias2
        v1 = (log_sigma_1).exp().data[0][0][sample_index] ** 2
        v2 = (log_sigma_2).exp().data[0][0][sample_index] ** 2
        c = rho.data[0][0][sample_index] * log_sigma_1.exp().data[0][0][sample_index] \
            * log_sigma_2.exp().data[0][0][sample_index]
        cov = np.array([[v1, c], [c, v2]], dtype=float)
        sample_point = np.random.multivariate_normal(mu, cov)

        out = np.insert(sample_point, 0, sample_end)
        record.append(out)
        x = torch.from_numpy(out).type(torch.FloatTensor)
        if cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=False)
        x = x.view(1, 1, 3)

        # attention
        old_phi = old_phi.squeeze(0)
        phis.append(old_phi)
        old_phi = old_phi.data.cpu().numpy()

        # hack to prevent early exit (attention is unstable at the beginning)
        if count >= 20 and np.max(old_phi) == old_phi[-1]:
            stop = True
        count += 1

    phis = torch.stack(phis).data.cpu().numpy().T

    plot_stroke(np.array(record))
    attention_plot(phis)


def attention_plot(phis):
    plt.rcParams["figure.figsize"] = (12, 6)
    phis = phis / (np.sum(phis, axis=0, keepdims=True))
    plt.xlabel('handwriting generation')
    plt.ylabel('text scanning')
    plt.imshow(phis, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()
