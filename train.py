import torch
from utils.data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from utils.training_logger import TrainingLogger
from model.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from utils.loss import FocalLoss, SSIM
import os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path, args.anomaly_source_path, resize_shape=[args.size,args.size])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=0)

        # Initialize Logger
        logger = TrainingLogger()
        loss_names = ['l2_loss', 'ssim_loss', 'seg_loss', 'total_loss']
        logger.print_header(loss_names)

        n_iter = 0
        for epoch in range(args.epochs):
            # print("Epoch: "+str(epoch)) # Handled by logger
            
            pbar = logger.create_progress_bar(dataloader, len(dataloader))
            
            for i_batch, sample_batched in pbar:
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                
                # Log batch
                current_losses = [l2_loss.item(), ssim_loss.item(), segment_loss.item(), loss.item()]
                logger.log_batch(epoch, args.epochs, current_losses, 
                               targets_shape=(args.bs,), 
                               imgs_shape=gray_batch.shape, 
                               pbar=pbar, 
                               batch_idx=i_batch)
                n_iter +=1

            scheduler.step()
            
            # Log epoch end
            logger.log_epoch_end(epoch, [get_lr(optimizer)], current_losses, results=None)

            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pt"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pt"))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--size', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    obj_batch = [['18.07_20-23defrom']
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['18.07_20-23defrom']
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

"""
python train.py --gpu_id 0 --obj_id 0 \
 --lr 0.0001 --size 256\
 --bs 16 --epochs 30 \
 --checkpoint_path "D:/Documents/DRAEM/checkpoints" \
 --data_path D:\Datasets\18.07_20-23defrom\train\good \
 --anomaly_source_path "D:/Datasets/dtd/images" \
 --log_path "D:/Documents/DRAEM/logs" 
"""