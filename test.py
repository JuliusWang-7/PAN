import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, datasets_2, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models import PAN
import nibabel
import SimpleITK as sitk
import time
from Evaluation.Eval_metrics import compute_surface_distances, \
    compute_average_surface_distance, compute_robust_hausdorff, compute_dice_coefficient

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def make_one_hot(mask, num_class):
    # 数据转为one hot 类型
    # mask_unique = np.unique(mask)
    mask_unique = [m for m in range(num_class)]
    one_hot_mask = [mask == i for i in mask_unique]
    one_hot_mask = np.stack(one_hot_mask)
    return one_hot_mask

def main():
    num_class = 57
    test_dir = '/LPBA_path/Test/'
    imgs_path = glob.glob(os.path.join(test_dir, "*.nii.gz"))
    imgs_path.sort()
    img_size = (160, 192, 160)
    spacing = (1, 1, 1)

    model_idx = -1
    model_root = '/LPBA_path/Model/PAN'
    model_path = model_root + '/experiments/'
    # log_path = save_root + '/logs' + save_dir

    save_root = '/LPBA_path/Results/PAN'
    save_flow = save_root + '/deformable_field'
    if not os.path.exists(save_flow):
        os.makedirs(save_flow)
    save_label_trans = save_root + '/label_trans'
    if not os.path.exists(save_label_trans):
        os.makedirs(save_label_trans)
    save_moving_trans = save_root + '/moving_trans'
    if not os.path.exists(save_moving_trans):
        os.makedirs(save_moving_trans)
    save_fixed = save_root + '/fixed'
    if not os.path.exists(save_fixed):
        os.makedirs(save_fixed)
    if not os.path.exists(save_fixed +'/img'):
        os.makedirs(save_fixed +'/img')
    if not os.path.exists(save_fixed+'/label'):
        os.makedirs(save_fixed +'/label')
    save_moving = save_root + '/moving'
    if not os.path.exists(save_moving):
        os.makedirs(save_moving)
    if not os.path.exists(save_moving + '/img'):
        os.makedirs(save_moving + '/img')
    if not os.path.exists(save_moving + '/label'):
        os.makedirs(save_moving + '/label')

    model_folder = 'PAN/'
    model_dir = model_path + model_folder


    model = PAN(img_size)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location='cuda:0')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_m = utils.register_model(img_size, 'bilinear')
    reg_model_m.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = datasets_2.LPBA40_InferDataset(imgs_path, transforms=test_composed, istest=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_assd_def = utils.AverageMeter()
    eval_assd_raw = utils.AverageMeter()
    eval_hd95_def = utils.AverageMeter()
    eval_hd95_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_time = utils.AverageMeter()

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            save_name = data[4][0]
            # spacing = data[5][0].detach().cpu().numpy()
            data = [t.cuda() for t in data[:4]]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            time_start = time.time()

            # x_in = torch.cat((x,y),dim=1)
            x_def, flow, _, _ = model(x, y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            out = reg_model_m([x.cuda().float(), flow.cuda()])

            time_end = time.time()
            eval_time.update((time_end - time_start))

            # deformable_field = sitk.GetImageFromArray(flow.detach().cpu().numpy()[0].transpose((3,2,1,0)), isVector=True)
            # sitk.WriteImage(deformable_field, os.path.join(save_flow, save_name))
            label_out = def_out.detach().cpu().numpy()[0, 0, :, :, :].astype('int8')
            nibabel.save(nibabel.Nifti1Image(label_out, np.eye(4)), os.path.join(save_label_trans, save_name))
            out = out.detach().cpu().numpy()[0, 0, :, :, :].astype('float64')
            nibabel.save(nibabel.Nifti1Image(out, np.eye(4)), os.path.join(save_moving_trans, save_name))
            nibabel.save(nibabel.Nifti1Image(y.detach().cpu().numpy()[0, 0, :, :, :].astype('float64'),
                                             np.eye(4)), os.path.join(save_fixed + '/img', save_name))
            nibabel.save(nibabel.Nifti1Image(y_seg.detach().cpu().numpy()[0, 0, :, :, :].astype('int8'),
                                             np.eye(4)), os.path.join(save_fixed + '/label', save_name))
            nibabel.save(nibabel.Nifti1Image(x.detach().cpu().numpy()[0, 0, :, :, :].astype('float64'),
                                             np.eye(4)), os.path.join(save_moving + '/img', save_name))
            nibabel.save(nibabel.Nifti1Image(x_seg.detach().cpu().numpy()[0, 0, :, :, :].astype('int8'),
                                             np.eye(4)), os.path.join(save_moving + '/label', save_name))

            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), num_clus=num_class)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), num_clus=num_class)


            # one-hot calculate distance
            seg_trans = make_one_hot(def_out.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            seg_x = make_one_hot(x_seg.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            seg_y = make_one_hot(y_seg.detach().cpu().numpy()[0, 0, :, :, :], num_class=num_class)
            assd_trans = 0
            assd_raw = 0
            hd95_trans = 0
            hd95_raw = 0
            cal_index = 0

            for i in range(seg_trans.shape[0]):
                if i == 0:
                    continue

                if (seg_trans[i] == False).all() or (seg_y[i] == False).all() or (seg_x[i] == False).all():
                    continue

                sur_dist_trans = compute_surface_distances(seg_trans[i], seg_y[i], spacing_mm=spacing)
                sur_dist_raw = compute_surface_distances(seg_x[i], seg_y[i], spacing_mm=spacing)

                a, b = compute_average_surface_distance(sur_dist_trans)
                assd_trans += (a+b)/2
                c, d = compute_average_surface_distance(sur_dist_raw)
                assd_raw += (c+d)/2
                hd95_trans += compute_robust_hausdorff(sur_dist_trans, 95)
                hd95_raw += compute_robust_hausdorff(sur_dist_raw, 95)
                cal_index += 1


            assd_trans /= cal_index
            assd_raw /= cal_index
            hd95_trans /= cal_index
            hd95_raw /= cal_index
            print(save_name)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            print('Trans assd: {:.4f}, Raw assd: {:.4f}'.format(assd_trans.item(), assd_raw.item()))
            print('Trans hd95: {:.4f}, Raw hd95: {:.4f}'.format(hd95_trans.item(), hd95_raw.item()))

            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            eval_assd_def.update(assd_trans.item(), x.size(0))
            eval_assd_raw.update(assd_raw.item(), x.size(0))
            eval_hd95_def.update(hd95_trans.item(), x.size(0))
            eval_hd95_raw.update(hd95_raw.item(), x.size(0))

            stdy_idx += 1


        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('Deformed ASSD: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_assd_def.avg,
                                                                                    eval_assd_def.std,
                                                                                    eval_assd_raw.avg,
                                                                                    eval_assd_raw.std))
        print('Deformed HD95: {:.3f} +- {:.3f}, Affine hd95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,
                                                                                    eval_hd95_def.std,
                                                                                    eval_hd95_raw.avg,
                                                                                    eval_hd95_raw.std))

        print('deformed det: %.3e, std: %.3e' % (eval_det.avg, eval_det.std))

        import xlwt

        new_book = xlwt.Workbook()
        new_table = new_book.add_sheet('score', cell_overwrite_ok=True)

        new_table.write(1, 1, '{:.3f} ± {:.3f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        new_table.write(1, 2, '{:.3f} ± {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))

        new_table.write(1, 3, '{:.3f} ± {:.3f}'.format(eval_assd_raw.avg, eval_assd_raw.std))
        new_table.write(1, 4, '{:.3f} ± {:.3f}'.format(eval_assd_def.avg, eval_assd_def.std))

        new_table.write(1, 5, '{:.3f} ± {:.3f}'.format(eval_hd95_raw.avg, eval_hd95_raw.std))
        new_table.write(1, 6, '{:.3f} ± {:.3f}'.format(eval_hd95_def.avg, eval_hd95_def.std))
        new_table.write(1, 7, '%.3e ± %.3e' % (eval_det.avg, eval_det.std))
        new_table.write(1, 8, '{:.1f} ± {:.1f}'.format(eval_time.avg, eval_time.std))

        new_book.save("/LPBA_path/Results/PAN/Infer_LPBA.xls")



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