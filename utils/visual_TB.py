import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Visualizer():

    def __init__(self, log_dir):

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def write_lr(self, optim, globaliter):
        for i, param_group in enumerate(optim.param_groups):
            self.summary_writer.add_scalar('learning_rate/lr_' + str(i), param_group['lr'], globaliter)
        self.summary_writer.flush()

    def write_loss_train(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/train', value, globaliter)
        self.summary_writer.flush()

    def write_loss_validation(self, value, globaliter, if_testtimes=False):
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''

        self.summary_writer.add_scalar('Loss/validation' + postfix, value, globaliter)
        self.summary_writer.flush()

    def write_image(self, images, epoch, if_predict=False, if_testtimes=False):

        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''
        # if len(images.shape) == 4:
        if True:
            _, _, row, col = images.shape
            vol_NE_batch = torch.zeros((6, 1, row, col))
            speed_NE_batch = torch.zeros((6, 1, row, col))
            vol_NW_batch = torch.zeros((6, 1, row, col))
            speed_NW_batch = torch.zeros((6, 1, row, col))
            vol_SE_batch = torch.zeros((6, 1, row, col))
            speed_SE_batch = torch.zeros((6, 1, row, col))
            vol_SW_batch = torch.zeros((6, 1, row, col))
            speed_SW_batch = torch.zeros((6, 1, row, col))

            # NE Vol
            vol_NE_batch[0] = images[0, 0, :, :]
            vol_NE_batch[1] = images[0, 8, :, :]
            vol_NE_batch[2] = images[0, 16, :, :]
            vol_NE_batch[3] = images[0, 24, :, :]
            vol_NE_batch[4] = images[0, 32, :, :]
            vol_NE_batch[5] = images[0, 40, :, :]

            # NE Speed
            speed_NE_batch[0] = images[0, 1, :, :]
            speed_NE_batch[1] = images[0, 9, :, :]
            speed_NE_batch[2] = images[0, 17, :, :]
            speed_NE_batch[3] = images[0, 25, :, :]
            speed_NE_batch[4] = images[0, 33, :, :]
            speed_NE_batch[5] = images[0, 41, :, :]

            # NW Vol
            vol_NW_batch[0] = images[0, 2, :, :]
            vol_NW_batch[1] = images[0, 10, :, :]
            vol_NW_batch[2] = images[0, 18, :, :]
            vol_NW_batch[3] = images[0, 26, :, :]
            vol_NW_batch[4] = images[0, 34, :, :]
            vol_NW_batch[5] = images[0, 42, :, :]

            # NW Speed
            speed_NW_batch[0] = images[0, 3, :, :]
            speed_NW_batch[1] = images[0, 11, :, :]
            speed_NW_batch[2] = images[0, 19, :, :]
            speed_NW_batch[3] = images[0, 27, :, :]
            speed_NW_batch[4] = images[0, 35, :, :]
            speed_NW_batch[5] = images[0, 43, :, :]

            # SE Vol
            vol_SE_batch[0] = images[0, 4, :, :]
            vol_SE_batch[1] = images[0, 12, :, :]
            vol_SE_batch[2] = images[0, 20, :, :]
            vol_SE_batch[3] = images[0, 28, :, :]
            vol_SE_batch[4] = images[0, 36, :, :]
            vol_SE_batch[5] = images[0, 44, :, :]

            # SE Speed
            speed_SE_batch[0] = images[0, 5, :, :]
            speed_SE_batch[1] = images[0, 13, :, :]
            speed_SE_batch[2] = images[0, 21, :, :]
            speed_SE_batch[3] = images[0, 29, :, :]
            speed_SE_batch[4] = images[0, 37, :, :]
            speed_SE_batch[5] = images[0, 45, :, :]

            # SW Vol
            vol_SW_batch[0] = images[0, 6, :, :]
            vol_SW_batch[1] = images[0, 14, :, :]
            vol_SW_batch[2] = images[0, 22, :, :]
            vol_SW_batch[3] = images[0, 30, :, :]
            vol_SW_batch[4] = images[0, 38, :, :]
            vol_SW_batch[5] = images[0, 46, :, :]

            # SW Speed
            speed_SW_batch[0] = images[0, 7, :, :]
            speed_SW_batch[1] = images[0, 15, :, :]
            speed_SW_batch[2] = images[0, 23, :, :]
            speed_SW_batch[3] = images[0, 31, :, :]
            speed_SW_batch[4] = images[0, 39, :, :]
            speed_SW_batch[5] = images[0, 47, :, :]

        if if_predict:
            vol_NE_batch = torchvision.utils.make_grid(vol_NE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/volume_ne', vol_NE_batch, epoch)

            speed_NE_batch = torchvision.utils.make_grid(speed_NE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/speed_ne', speed_NE_batch, epoch)

            vol_NW_batch = torchvision.utils.make_grid(vol_NW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/volume_nw', vol_NW_batch, epoch)

            speed_NW_batch = torchvision.utils.make_grid(speed_NW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/speed_nw', speed_NW_batch, epoch)

            vol_SE_batch = torchvision.utils.make_grid(vol_SE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/volume_se', vol_SE_batch, epoch)

            speed_SE_batch = torchvision.utils.make_grid(speed_SE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/speed_se', speed_SE_batch, epoch)

            vol_SW_batch = torchvision.utils.make_grid(vol_SW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/volume_sw', vol_SW_batch, epoch)

            speed_SW_batch = torchvision.utils.make_grid(speed_SW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/speed_sw', speed_SW_batch, epoch)


        else:
            vol_NE_batch = torchvision.utils.make_grid(vol_NE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/volume_ne', vol_NE_batch, epoch)

            speed_NE_batch = torchvision.utils.make_grid(speed_NE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/speed_ne', speed_NE_batch, epoch)

            vol_NW_batch = torchvision.utils.make_grid(vol_NW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/volume_nw', vol_NW_batch, epoch)

            speed_NW_batch = torchvision.utils.make_grid(speed_NW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/speed_nw', speed_NW_batch, epoch)

            vol_SE_batch = torchvision.utils.make_grid(vol_SE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/volume_se', vol_SE_batch, epoch)

            speed_SE_batch = torchvision.utils.make_grid(speed_SE_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/speed_se', speed_SE_batch, epoch)

            vol_SW_batch = torchvision.utils.make_grid(vol_SW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/volume_sw', vol_SW_batch, epoch)

            speed_SW_batch = torchvision.utils.make_grid(speed_SW_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/speed_sw', speed_SW_batch, epoch)

        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()


if __name__ == "__main__":

    import numpy as np

    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16
        print(img_batch[i, 0].shape)

        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16

    print(img_batch.shape)
