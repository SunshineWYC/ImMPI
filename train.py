import configargparse
import random
import time
from utils.utils import *
from models.mpi_generator import MPIGenerator
from models.feature_generator import FeatureGenerator
from datasets import find_dataset_def
from torch.utils.data import DataLoader
from utils.render import renderNovelView
from torch.utils.tensorboard import SummaryWriter
from models.losses import *
import shutil


class ASITrainer():
    def __init__(self, args, device):
        super(ASITrainer, self).__init__()
        self.args = args
        self.logger = SummaryWriter(args.logdir)
        self.device = device

        self.start_epoch = 0
        self.epochs = args.epochs

        self.neighbor_view_num = args.neighbor_view_num

        self.feature_generator, self.mpi_generator = self.model_definition()
        self.optimizer, self.lr_scheduler = self.optimizer_definition()
        self.train_dataloader, self.validate_dataloader = self.dataloader_definition()
        if args.resume:
            self.resume_training()

        self.ssim_calculator = SSIM().cuda()
        self.loss_rgb_weight = args.loss_rgb_weight
        self.loss_ssim_weight = args.loss_ssim_weight

        # copy train config file
        shutil.copy(self.args.config, os.path.join(self.args.logdir, "config.txt"))

    def model_definition(self):
        """
        model definition
        Returns: models
        """
        feature_generator = FeatureGenerator(model_type=self.args.feature_generator_model_type, pretrained=True, device=self.device).to(self.device)
        mpi_generator = MPIGenerator(feature_out_chs=feature_generator.encoder_channels).to(self.device)

        train_params = sum(params.numel() for params in feature_generator.parameters() if params.requires_grad) + \
                       sum(params.numel() for params in mpi_generator.parameters() if params.requires_grad)
        print("Total_paramteters: {}".format(train_params))

        return feature_generator, mpi_generator

    def optimizer_definition(self):
        """
        optimizer definition
        Returns:
        """
        params = [
            {"params": self.feature_generator.parameters(), "lr": self.args.learning_rate},
            {"params": self.mpi_generator.parameters(), "lr": self.args.learning_rate}
        ]
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))

        milestones = [int(epoch_idx) for epoch_idx in self.args.lr_ds_epoch_idx.split(':')[0].split(',')]
        lr_gamma = 1 / float(self.args.lr_ds_epoch_idx.split(':')[1])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=self.start_epoch - 1)
        return optimizer, lr_scheduler

    def dataloader_definition(self):
        # dataset, dataloader definition
        MVSDataset = find_dataset_def(self.args.dataset)
        train_dataset = MVSDataset(self.args.train_dataset_dirpath, self.args.train_list_filepath, neighbor_view_num=self.args.neighbor_view_num)
        validate_dataset = MVSDataset(self.args.validate_dataset_dirpath, self.args.validate_list_filepath, neighbor_view_num=self.args.neighbor_view_num)
        train_dataloader = DataLoader(train_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        validate_dataloader = DataLoader(validate_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        return train_dataloader, validate_dataloader

    def resume_training(self):
        """
        training process resume, load model and optimizer ckpt
        """
        if self.args.loadckpt is None:
            saved_models = [fn for fn in os.listdir(self.args.logdir) if fn.endswith(".ckpt")]
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # use the latest checkpoint file
            loadckpt = os.path.join(self.args.logdir, saved_models[-1])
        else:
            loadckpt = self.args.loadckpt
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt)
        self.start_epoch = state_dict["epoch"]
        self.feature_generator.load_state_dict(state_dict["feature_generator"])
        self.mpi_generator.load_state_dict(state_dict["mpi_generator"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        # self.start_epoch = state_dict["epoch"] + 1
        self.start_epoch = 0    # fine tune from whu_view_syn_small model:799

        # redefine lr_schedular
        milestones = [int(epoch_idx) for epoch_idx in self.args.lr_ds_epoch_idx.split(':')[0].split(',')]
        lr_gamma = 1 / float(self.args.lr_ds_epoch_idx.split(':')[1])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=lr_gamma, last_epoch=self.start_epoch - 1)

    def set_data(self, sample):
        """
        set batch_sample data
        Args:
            sample:

        Returns:
        """
        if self.device == torch.device("cuda"):
            sample = dict2cuda(sample)
        self.image_ref = sample["image_ref"]
        self.depth_min_ref, self.depth_max_ref = sample["depth_min_ref"], sample["depth_max_ref"]
        self.K_ref = sample["K_ref"]
        self.depth_ref = sample["depth_ref"]
        self.images_tgt = sample["images_tgt"]
        self.Ks_tgt, self.Ts_tgt_ref = sample["Ks_tgt"], sample["Ts_tgt_ref"]   # [B, N, 3 ,3], [B, N, 4, 4]
        self.height, self.width = self.image_ref.shape[2], self.image_ref.shape[3]

    def train(self):
        for epoch_idx in range(self.start_epoch, self.epochs):
            print("Training process, Epoch: {}/{}".format(epoch_idx, self.args.epochs))
            for batch_idx, sample in enumerate(self.train_dataloader):
                start_time = time.time()
                global_step = len(self.train_dataloader) * epoch_idx + batch_idx
                self.set_data(sample)
                summary_scalars, summary_images = self.train_sample(self.args.depth_sample_num)
                print("Epoch:{}/{}, Iteration:{}/{}, train loss={:.4f}, time={:.4f}".format(epoch_idx, self.epochs, batch_idx, len(self.train_dataloader), summary_scalars["loss"], time.time() - start_time))
                if global_step % self.args.summary_scalars_freq == 0:
                    save_scalars(self.logger, "Train", summary_scalars, global_step)    # scalars for random sampled tgt-view image
                if global_step % self.args.summary_images_freq == 0:
                    for scale in range(4):
                        save_images(self.logger, "Train_scale_{}".format(scale), summary_images["scale_{}".format(scale)], global_step) # summary images for random sampled tgt-image
            if (epoch_idx+1) % self.args.save_ckpt_freq == 0:
                torch.save({
                            "epoch": epoch_idx,
                            "feature_generator": self.feature_generator.state_dict(),
                            "mpi_generator": self.mpi_generator.state_dict(),
                            "optimizer": self.optimizer.state_dict(),},
                            "{}/mpimodel_{:0>4}.ckpt".format(self.args.logdir, epoch_idx))

            if (epoch_idx+1) % self.args.validate_freq == 0:
                self.validate(epoch_idx, self.args.depth_sample_num)
            self.lr_scheduler.step()

    def train_sample(self, depth_sample_num):
        """
        calculate 4 scale loss, loss backward per tgt image
        Returns: summary_scalars, summary_images
        """
        self.feature_generator.train()
        self.mpi_generator.train()

        # network forward, generate mpi representations
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(self.image_ref)
        mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out], depth_sample_num=depth_sample_num)
        rgb_mpi_ref_dict = {
            "scale_0": mpi_outputs["MPI_{}".format(0)][:, :, :3, :, :],
            "scale_1": mpi_outputs["MPI_{}".format(1)][:, :, :3, :, :],
            "scale_2": mpi_outputs["MPI_{}".format(2)][:, :, :3, :, :],
            "scale_3": mpi_outputs["MPI_{}".format(3)][:, :, :3, :, :],
        }
        sigma_mpi_ref_dict = {
            "scale_0": mpi_outputs["MPI_{}".format(0)][:, :, 3:, :, :],
            "scale_1": mpi_outputs["MPI_{}".format(1)][:, :, 3:, :, :],
            "scale_2": mpi_outputs["MPI_{}".format(2)][:, :, 3:, :, :],
            "scale_3": mpi_outputs["MPI_{}".format(3)][:, :, 3:, :, :],
        }

        neighbor_image_idx = random.randint(0, self.neighbor_view_num-1)
        summary_scalars, summary_images = self.train_per_image(rgb_mpi_ref_dict, sigma_mpi_ref_dict, neighbor_image_idx, depth_sample_num)

        return summary_scalars, summary_images

    def train_per_image(self, rgb_mpi_ref_dict, sigma_mpi_ref_dict, neighbor_image_idx, depth_sample_num):
        with torch.no_grad():
            T_ref_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.args.batch_size, 1, 1)
            T_tgt_ref = self.Ts_tgt_ref[:, neighbor_image_idx, :, :]

        summary_scalars, summary_images = {}, {}
        loss_per_image, loss_rgb_per_image, loss_ssim_per_image = 0.0, 0.0, 0.0
        for scale in range(4):
            with torch.no_grad():
                # rescale intrinsics for ref-view, tgt-views
                K_ref_scaled = self.K_ref / (2 ** scale)
                K_ref_scaled[:, 2, 2] = 1
                K_tgt_scaled = self.Ks_tgt[:, neighbor_image_idx, :, :] / (2 ** scale)
                K_tgt_scaled[:, 2, 2] = 1  # [B, 3, 3]
                height_render, width_render = self.height // 2 ** scale, self.width // 2 ** scale
                # rescale image_ref, depth_ref, images_tgt
                image_ref = F.interpolate(self.image_ref, size=(height_render, width_render), mode="bilinear")  # [B, 3, H//scale, W//scale]
                depth_ref = F.interpolate(self.depth_ref.unsqueeze(1), size=(height_render, width_render), mode="nearest")  # Not for loss, for monitor depth MAE
                image_tgt = F.interpolate(self.images_tgt[:, neighbor_image_idx, :, :, :], size=(height_render, width_render), mode="bilinear")  # [B, 3, H//scale, W//scale]

            # render ref-view syn image
            ref_rgb_syn, ref_depth_syn, ref_mask = renderNovelView(
                rbg_MPI_ref=rgb_mpi_ref_dict["scale_{}".format(scale)],
                sigma_MPI_ref=sigma_mpi_ref_dict["scale_{}".format(scale)],
                depth_min_ref=self.depth_min_ref,
                depth_max_ref=self.depth_max_ref,
                depth_hypothesis_num=depth_sample_num,
                T_tgt_ref=T_ref_ref,
                K_ref=K_ref_scaled,
                K_tgt=K_ref_scaled,
                height_render=height_render,
                width_render=width_render,
            )

            tgt_rgb_syn, tgt_depth_syn, tgt_mask = renderNovelView(
                rbg_MPI_ref=rgb_mpi_ref_dict["scale_{}".format(scale)],
                sigma_MPI_ref=sigma_mpi_ref_dict["scale_{}".format(scale)],
                depth_min_ref=self.depth_min_ref,
                depth_max_ref=self.depth_max_ref,
                depth_hypothesis_num=depth_sample_num,
                T_tgt_ref=T_tgt_ref,
                K_ref=K_ref_scaled,
                K_tgt=K_tgt_scaled,
                height_render=height_render,
                width_render=width_render,
            )

            loss_rgb = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_rgb_weight
            loss_ssim = loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_ssim_weight
            loss = loss_rgb + loss_ssim

            loss_rgb_per_image = loss_rgb_per_image + loss_rgb
            loss_ssim_per_image = loss_ssim_per_image + loss_ssim
            loss_per_image = loss_per_image + loss

            with torch.no_grad():
                summary_images["scale_{}".format(scale)] = {
                    "ref_image": image_ref,
                    "ref_rgb_syn": ref_rgb_syn,
                    "tgt_rgb_syn": tgt_rgb_syn,
                    "ref_depth_syn": ref_depth_syn,
                    "ref_depth": depth_ref,
                    "ref_depth_diff": torch.abs(depth_ref - ref_depth_syn),
                    "tgt_mask": tgt_mask
                }

        self.optimizer.zero_grad()
        loss_per_image.backward()
        self.optimizer.step()

        with torch.no_grad():
            summary_scalars = {
                "loss": loss_per_image.item(),
                "loss_rgb": loss_rgb_per_image.item(),
                "loss_ssim": loss_ssim_per_image.item(),
                # "depth_MAE": torch.mean(torch.abs(summary_images["scale_0"]["ref_depth_syn"] - summary_images["scale_0"]["ref_depth"]))
            }

        return summary_scalars, summary_images


    def validate(self, epoch_idx, depth_sample_num):
        print("Validating process, Epoch: {}/{}".format(epoch_idx, self.epochs))
        average_validate_scalars = ScalarDictMerge()
        for batch_idx, sample in enumerate(self.validate_dataloader):
            self.set_data(sample)
            summary_scalars, summary_images = self.validate_sample(depth_sample_num)
            average_validate_scalars.update(summary_scalars)
        save_scalars(self.logger, "Validate", average_validate_scalars.mean(), epoch_idx)
        save_images(self.logger, "Validate", summary_images["scale_0"], epoch_idx)

    def validate_sample(self, depth_sample_num):
        self.feature_generator.eval()
        self.mpi_generator.eval()
        with torch.no_grad():
            # network forward, generate mpi representations
            conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(self.image_ref)
            mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out], depth_sample_num=depth_sample_num)

            summary_scalars, summary_images = {}, {}  # 0-idx tgt-view summary, scale_0
            for neighbor_image_idx in range(self.neighbor_view_num):  # loss backward and optimizer step neighbor_view_num times
                loss_per_image, loss_rgb_per_image, loss_ssim_per_image = 0.0, 0.0, 0.0
                for scale in range(4):
                    with torch.no_grad():
                        # rescale intrinsics for ref-view, tgt-views
                        K_ref_scaled = self.K_ref / (2 ** scale)
                        K_ref_scaled[:, 2, 2] = 1
                        K_tgt_scaled = self.Ks_tgt[:, neighbor_image_idx, :, :] / (2 ** scale)
                        K_tgt_scaled[:, 2, 2] = 1  # [B, 3, 3]
                        height_render, width_render = self.height // 2 ** scale, self.width // 2 ** scale
                        # rescale image_ref, depth_ref, images_tgt
                        image_ref = F.interpolate(self.image_ref, size=(height_render, width_render), mode="bilinear")  # [B, 3, H//scale, W//scale]
                        depth_ref = F.interpolate(self.depth_ref.unsqueeze(1), size=(height_render, width_render), mode="nearest")  # Not for loss, for monitor depth MAE
                        image_tgt = F.interpolate(self.images_tgt[:, neighbor_image_idx, :, :, :], size=(height_render, width_render), mode="bilinear")  # [B, 3, H//scale, W//scale]

                    rgb_mpi_ref = mpi_outputs["MPI_{}".format(scale)][:, :, :3, :, :]
                    sigma_mpi_ref = mpi_outputs["MPI_{}".format(scale)][:, :, 3:, :, :]

                    # render ref-view syn image
                    T_ref_ref = torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.args.batch_size, 1, 1)
                    ref_rgb_syn, ref_depth_syn, ref_mask = renderNovelView(
                        rbg_MPI_ref=rgb_mpi_ref,
                        sigma_MPI_ref=sigma_mpi_ref,
                        depth_min_ref=self.depth_min_ref,
                        depth_max_ref=self.depth_max_ref,
                        depth_hypothesis_num=depth_sample_num,
                        T_tgt_ref=T_ref_ref,
                        K_ref=K_ref_scaled,
                        K_tgt=K_ref_scaled,
                        height_render=height_render,
                        width_render=width_render,
                    )

                    T_tgt_ref = self.Ts_tgt_ref[:, neighbor_image_idx, :, :]
                    tgt_rgb_syn, tgt_depth_syn, tgt_mask = renderNovelView(
                        rbg_MPI_ref=rgb_mpi_ref,
                        sigma_MPI_ref=sigma_mpi_ref,
                        depth_min_ref=self.depth_min_ref,
                        depth_max_ref=self.depth_max_ref,
                        depth_hypothesis_num=depth_sample_num,
                        T_tgt_ref=T_tgt_ref,
                        K_ref=K_ref_scaled,
                        K_tgt=K_tgt_scaled,
                        height_render=height_render,
                        width_render=width_render,
                    )

                    loss_rgb = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_rgb_weight
                    loss_ssim = loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_ssim_weight
                    loss = loss_rgb + loss_ssim

                    loss_rgb_per_image = loss_rgb_per_image + loss_rgb
                    loss_ssim_per_image = loss_ssim_per_image + loss_ssim
                    loss_per_image = loss_per_image + loss

                    if neighbor_image_idx == 0:
                        with torch.no_grad():
                            summary_images["scale_{}".format(scale)] = {
                                "ref_image": image_ref,
                                "ref_rgb_syn": ref_rgb_syn,
                                "tgt_rgb_syn": tgt_rgb_syn,
                                "ref_depth_syn": ref_depth_syn,
                                "ref_depth": depth_ref,
                                "ref_depth_diff": torch.abs(depth_ref - ref_depth_syn),
                                "tgt_mask": tgt_mask
                            }

                if neighbor_image_idx == 0:
                    with torch.no_grad():
                        summary_scalars = {
                            "loss": loss_per_image.item(),
                            "loss_rgb": loss_rgb_per_image.item(),
                            "loss_ssim": loss_ssim_per_image.item(),
                            # "depth_MAE": torch.mean(
                            #     torch.abs(summary_images["scale_0"]["ref_depth_syn"] - summary_images["scale_0"]["ref_depth"]))
                        }

        return summary_scalars, summary_images


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description="Prior Extractor Training with MVS Dataset")
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # dataset parameters
    parser.add_argument("--dataset", type=str, default="whu_mvs", help="select train mvs dataset")
    parser.add_argument("--train_dataset_dirpath", type=str, default=r"D:\Datasets\WHU\whu_mvs", help="train dataset directory path")
    parser.add_argument("--train_list_filepath", type=str, default="./datasets/datalist/whuViewSyn/train.txt", help="train list filepath")
    parser.add_argument("--validate_dataset_dirpath", type=str, default=r"D:\Datasets\WHU\whu_mvs", help="validate dataset directory path, if None, equal to train dataset")
    parser.add_argument("--validate_list_filepath", type=str, default="./datasets/datalist/whuViewSyn/val.txt", help="validate list filepath")
    # training parameters
    parser.add_argument("--epochs", type=int, default=500, help="train epoch number")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_ds_epoch_idx", type=str, default="100,200,300,400:2", help="epoch ids to downscale lr and the downscale rate")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--loadckpt", default=None, help="load a specific checkpoint")
    parser.add_argument("--logdir", type=str, default="./checkpoints/ASI_training", help="the directory to save checkpoints/logs, tensorboard event log")
    parser.add_argument("--resume", action="store_true", help="continue to train the model")
    # log writer and random seed parameters
    parser.add_argument("--summary_scalars_freq", type=int, default=10, help="save summary scalar frequency")
    parser.add_argument("--summary_images_freq", type=int, default=50, help="save summary images frequency")
    parser.add_argument("--save_ckpt_freq", type=int, default=50, help="save checkpoint frequency, 1 means per epoch")
    parser.add_argument("--validate_freq", type=int, default=10, help="validate frequency")
    parser.add_argument("--seed", type=int, default=28, metavar="S", help="random seed, ensure training can recurrence")
    # model parameters
    parser.add_argument("--depth_sample_num", type=int, default=32, help="depth sample number in decoder")
    parser.add_argument("--feature_generator_model_type", type=str, default="resnet18", help="feature generator model type")
    parser.add_argument("--neighbor_view_num", type=int, default=19, help="neighbor view number")
    # loss weights
    parser.add_argument("--loss_rgb_weight", type=float, default=2.0, help="loss rgb weight")
    parser.add_argument("--loss_ssim_weight", type=float, default=1.0, help="loss depth weight")

    args = parser.parse_args()
    print_args(args)

    # fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ASITrainer(args, device)
    trainer.train()
