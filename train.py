import glob
from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models import PAN
import random
import nibabel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    num_class = 57
    train_dir = '/LPBA_path/Train/'
    val_dir = '/LPBA_path/Val/'
    train_imgs = glob.glob(os.path.join(train_dir, "*.nii.gz"))
    val_imgs = glob.glob(os.path.join(val_dir, "*.nii.gz"))

    img_size = nibabel.load(train_imgs[0]).get_fdata().squeeze().shape

    weights = [1, 1, 1, 1]  # loss weights
    lr = 0.0001
    save_dir = 'PAN/'
    save_root = '/LPBA_path/Model/PAN'
    save_exp = save_root + '/experiments/' + save_dir
    save_log = save_root + '/logs/' + save_dir
    if not os.path.exists(save_exp):
        os.makedirs(save_exp)
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    lr = 0.0001
    epoch_start = 0
    max_epoch = 50
    # img_size = (160, 192, 224)
    cont_training = False

    '''
    Initialize model
    '''
    model = PAN(img_size)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    # reg_model_bilin = utils.register_model(img_size, 'bilinear')
    # reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 0
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(save_exp + natsorted(os.listdir(save_exp))[-3])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.int16)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                       ])

    train_set = datasets.LPBA40Dataset(train_imgs, transforms=train_composed)
    val_set = datasets.LPBA40_InferDataset(val_imgs, transforms=val_composed, istest=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # criterion = nn.MSELoss()
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [nn.MSELoss()] * 2
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_in = torch.cat((x, y), dim=1)
            output = model(x, y)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if n < 2:
                    curr_loss = loss_function(output[n], y) * weights[n]
                else:
                    curr_loss = loss_function(output[n], torch.eye(int(output[n].size()[0])).cuda()) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   loss_vals[0].item(),
                                                                                   loss_vals[1].item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data[:4]]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                # x_in = torch.cat((x, y), dim=1)
                # grid_img = mk_grid_img(8, 1, img_size)
                output = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), num_clus=num_class)
                # dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_exp, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        # plt.switch_backend('agg')
        # pred_fig = comput_fig(def_out)
        # grid_fig = comput_fig(def_grid)
        # x_fig = comput_fig(x_seg)
        # tar_fig = comput_fig(y_seg)
        # writer.add_figure('Grid', grid_fig, epoch)
        # plt.close(grid_fig)
        # writer.add_figure('input', x_fig, epoch)
        # plt.close(x_fig)
        # writer.add_figure('ground truth', tar_fig, epoch)
        # plt.close(tar_fig)
        # writer.add_figure('prediction', pred_fig, epoch)
        # plt.close(pred_fig)
        loss_all.reset()
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=5):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()