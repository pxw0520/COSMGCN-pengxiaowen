import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import IEMOCAPRobertaCometDataset
from model import MaskedNLLLoss
# from commonsense_model import CommonsenseGRUModel
# from commonsense_model import CommonsenseGCN
# from COMSGCN_v2_model import CommonsenseGCN
# from COMSGCN_v3_model import CommonsenseGCN
from COMSGCN_v4_model import CommonsenseGCN
from sklearn.metrics import f1_score, accuracy_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_IEMOCAP_loaders(batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPRobertaCometDataset('train')
    validset = IEMOCAPRobertaCometDataset('valid')
    testset = IEMOCAPRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=True)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, [avg_fscore], vids, ei, et, en, el

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general2', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=3, help='normalization strategy')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    emo_gru = True
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    global D_s

    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100

    D_e = D_p + D_r + D_i

    global seed
    seed = args.seed
    # seed_everything(seed)

    model = CommonsenseGCN(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual)

    print('IEMOCAP COSMGCN Model.')

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([1 / 0.086747,
                                      1 / 0.144406,
                                      1 / 0.227883,
                                      1 / 0.160585,
                                      1 / 0.127711,
                                      1 / 0.252668])

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        # loss_function = MaskedNLLLoss()   # fixme
        loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lf = open('logs/cosmgcn_iemocap_logs.txt', 'a')

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(batch_size=batch_size,
                                                                  num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_model(model, loss_function, train_loader, e,
                                                                              optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_model(model,
                                                                                                             loss_function,
                                                                                                             test_loader,
                                                                                                             e)

        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(
            e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
            test_fscore, round(time.time() - start_time, 2))

        print(x)
        lf.write(x + '\n')

    if args.tensorboard:
        writer.close()

    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]

    print('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))

    scores_best = np.max(test_fscores[0])
    print('@Best F1: {}'.format(scores_best))

    scores = [score1, score2, scores_best]
    scores = [str(item) for item in scores]

    rf = open('results/cosmgcn_iemocap_results.txt', 'a')
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()