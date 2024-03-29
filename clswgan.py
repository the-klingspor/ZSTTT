import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import classifier
import classifier2
import model
import logger

from extract_features import get_zsl_data_collection
from utils import util, visualization
from utils.rotation_net import RotationNet
from test_time_training.rotation_ttt import rotation_ttt_loop
from test_time_training.utils import ttt_epoch

parser = argparse.ArgumentParser()

# Dataset and paths
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--extract_features', action='store_true', default=False, help='extract features using a given backbone')
parser.add_argument('--backbone_path', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/resnet50_cub.pth')
parser.add_argument('--data_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/')
parser.add_argument('--splitdir', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/')
parser.add_argument('--class_txt', type=str, default='trainvalclasses.txt')
parser.add_argument('--attribute_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/attributes/class_attribute_labels_continuous.txt')
parser.add_argument('--include_txt', type=str, default='unseen_train.txt')

# Model parameters
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discrimindator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--image_size', type=int, default=336)
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--architecture', type=str, default="resnet50")
parser.add_argument('--ttt_n_loops', type=int, default=1, help='the number of test-time iterations on each sample')
parser.add_argument('--ttt_learning_rate', type=float, default=5e-3, help='learning rate for TTT')
parser.add_argument('--ttt_momentum', type=float, default=0, help='momentum for TTT')
parser.add_argument('--ttt_weight_decay', type=float, default=0, help='weight decay for TTT')
parser.add_argument('--ttt_batch_size', type=int, default=16, help='number of augmented copies for TTT')

# Training process
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--ttt', type=str, default=None, help='The type of test-time training to use. If None, use no TTT. '
                                                          'Valid parameters are {None (default), "ttt", "memo" and '
                                                          '"disco"')

# wandb
parser.add_argument('--log_online', action='store_true',
                    help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally '
                         'be set.')
parser.add_argument('--wandb_key', default='65954b19f28cc0f35372188d50be8f11cdb79321', type=str, help='API key for W&B.')
parser.add_argument('--project', default='Sample_Project', type=str,
                    help='Name of the project - relates to W&B project names. In --savename default setting part of '
                         'the savename.')
parser.add_argument('--group', default='', type=str, help='Name of the group - relates to W&B group names - all runs '
                                                          'with same setup but different seeds are logged into one '
                                                          'group. In --savename default setting part of the savename. '
                                                          'Name is created as model_dataset_group')
parser.add_argument('--savename', default='group_plus_seed', type=str, help='Run savename - if default, the savename'
                                                                            ' will comprise the project and group name '
                                                                            '(see wandb_parameters()).')
parser.add_argument('--name_seed', type=str, default=0, help="Randomly generated code as name for the run.")


opt = parser.parse_args()
print(opt)

if opt.log_online:
    logger.setup_logger(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
if opt.extract_features:
    data_collection = get_zsl_data_collection(opt)
    data = util.DATA_LOADER(opt, data=data_collection)
    save_dir = os.path.join(opt.dataroot, opt.dataset, "original_features.pdf")
    visualization.embed_and_plot(data, save_dir=save_dir)
else:
    data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

dict_to_log = {}
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
        errG = G_cost + opt.cls_weight*c_errG
        errG.backward()
        optimizerG.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item()))
    dict_to_log = {"loss_D": D_cost.item(),
                   "loss_G": G_cost.item(),
                   "wasserstein_D": Wasserstein_D.item(),
                   "clf_error_G": c_errG.item()}


    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        dict_to_log["unseen_accuracy"] = cls.acc_unseen
        dict_to_log["seen_accuracy"] = cls.acc_seen
        dict_to_log["harmonic_mean"] = cls.H
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num) 
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls.acc
        print('unseen class accuracy= ', acc)
        dict_to_log["accuracy"] = acc

    if opt.log_online:
        logger.log(dict_to_log)
    # reset G to training mode
    netG.train()

final_dict = {}

if opt.gzsl:
    final_dict["unseen_accuracy_final"] = dict_to_log["unseen_accuracy"]
    final_dict["seen_accuracy_final"] = dict_to_log["seen_accuracy"]
    final_dict["harmonic_mean_final"] = dict_to_log["harmonic_mean"]
else:
    final_dict["accuracy_final"] = dict_to_log["accuracy"]

if opt.log_online:
    print(final_dict)
    logger.log(final_dict)

if opt.ttt and opt.extract_features:
    if opt.ttt == 'ttt':
        ttt_loop = rotation_ttt_loop
    elif opt.ttt == 'memo':
        ttt_loop = None
        raise NotImplementedError()
    elif opt.ttt == 'disco':
        ttt_loop = None
        raise NotImplementedError()
    else:
        assert False, "Wrong TTT mode chosen - choose one of {ttt, memo, disco}"

    model = RotationNet(num_classes=opt.nclass_all, architecture=opt.architecture)
    model.load_state_dict(torch.load(opt.backbone_path))
    # Apply rotation TTT to each image
    ttt_features, ttt_losses = ttt_epoch(ttt_loop, model, data, opt)
    save_dir = os.path.join(opt.dataroot, opt.dataset, opt.ttt + "_features.pdf")
    visualization.embed_and_plot(data, save_dir=save_dir)

    data_collection['features'] = ttt_features
    ttt_data = util.DATA_LOADER(opt, data=data_collection)

    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, ttt_data.unseenclasses, ttt_data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(train_X, train_Y, ttt_data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        final_dict["unseen_accuracy_final"] = cls.acc_unseen
        final_dict["seen_accuracy_final"] = cls.acc_seen
        final_dict["harmonic_mean_final"] = cls.H
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, ttt_data.unseenclasses, ttt_data.attribute, opt.syn_num)
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, ttt_data.unseenclasses), ttt_data,
                                     ttt_data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls.acc
        print('unseen class accuracy= ', acc)
        final_dict["accuracy_final"] = acc

    if opt.log_online:
        print(final_dict)
        logger.log(final_dict)
