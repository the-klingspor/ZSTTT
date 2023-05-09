import wandb
import os

from torch import save as t_save


def setup_logger(opt):
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    save_path = os.path.join(opt.outname, 'CheckPoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    group_name = make_group_name(opt)
    if opt.savename == 'group_plus_seed':
        opt.savename = group_name + '_{}'.format(opt.name_seed)
    wandb.init(project=opt.project, group=group_name, name=opt.savename, dir=save_path,
               settings=wandb.Settings(start_method='fork'))
    wandb.config.update(vars(opt))


def make_group_name(opt):
    if opt.dataset != '':
        group_name = opt.dataset
    else:
        group_name = ''
    # group_name += opt.model
    if opt.group != '':
        group_name += "_" + opt.group
    return group_name


def log(dict_to_log, step=None):
    if step:
        wandb.log(dict_to_log, step)
    else:
        wandb.log(dict_to_log)


def save(model, image_extractor, name, args):
    extr_name = 'img_extr_' + name
    t_save(image_extractor.state_dict(), os.path.join(args.path_aux, extr_name))

    model_name = 'model_' + name
    t_save(model.state_dict(), os.path.join(args.path_aux, model_name))