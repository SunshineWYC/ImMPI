import configargparse
import datetime
import torch.optim.optimizer
from torch.utils.tensorboard import SummaryWriter
from utils.render import renderNovelView
from utils.utils import *
from models.mpi_generator import MPIGenerator
from models.feature_generator import FeatureGenerator
from datasets import find_scenedata_def
from models.losses import *
import lpips
import shutil


class OptimizePerScene():
    def __init__(self, model, ckpt_filepath, neighbor_view_num, depth_sample_num, epochs, learning_rate, lr_ds_epoch_idx, loss_rgb_weight, loss_ssim_weight, loss_lpips_weight,
                 summary_scalars_freq, summary_images_freq, save_ckpt_freq, device):
        self.feature_generator = model["feature_generator"].to(device)
        self.mpi_generator = model["mpi_generator"].to(device)
        self.load_ckpt(ckpt_filepath)

        self.neighbor_view_num = neighbor_view_num
        self.depth_sample_num = depth_sample_num
        self.epochs = epochs
        self.device = device

        # optimizer setting
        self.optimizer = torch.optim.Adam(self.mpi_generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        milestones = [int(epoch_idx) for epoch_idx in lr_ds_epoch_idx.split(':')[0].split(',')]
        lr_gamma = 1 / float(lr_ds_epoch_idx.split(':')[1])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=lr_gamma, last_epoch=-1)

        self.loss_rgb_weight = loss_rgb_weight
        self.loss_ssim_weight = loss_ssim_weight
        self.loss_lpips_weight = loss_lpips_weight

        self.summary_scalars_freq = summary_scalars_freq
        self.summary_images_freq = summary_images_freq
        self.save_ckpt_freq = save_ckpt_freq

        # loss calculater definition
        self.ssim_calculator = SSIM()
        self.lpips_calculator = lpips.LPIPS(net="vgg").to(device)
        self.lpips_calculator.requires_grad = True

    def load_ckpt(self, ckpt_filepath):
        state_dict = torch.load(ckpt_filepath)
        self.feature_generator.load_state_dict(state_dict["feature_generator"])
        self.mpi_generator.load_state_dict(state_dict["mpi_generator"])

    def set_data(self, scene_data):
        if self.device == torch.device("cuda"):
            scene_data = dict2cuda(scene_data)
        self.image_ref = scene_data["image_ref"]
        self.depth_min_ref, self.depth_max_ref = scene_data["depth_min_ref"], scene_data["depth_max_ref"]
        self.K_ref = scene_data["K_ref"]
        self.E_ref = scene_data["E_ref"]
        self.depth_ref = scene_data["depth_ref"]
        self.images_tgt = scene_data["images_tgt"]
        self.Ks_tgt, self.Ts_tgt_ref = scene_data["Ks_tgt"], scene_data["Ts_tgt_ref"]  # [B, N, 3 ,3], [B, N, 4, 4]
        self.height, self.width = self.image_ref.shape[2], self.image_ref.shape[3]

    def optimize(self, scene_data, logdir, neighbor_view_indices):
        logger = SummaryWriter(logdir)
        self.set_data(scene_data)
        with torch.no_grad():
            conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(self.image_ref)

        self.mpi_generator.train()
        for epoch_idx in range(self.epochs):
            # network forward, generate mpi representations
            summary_scalars_epoch = ScalarDictMerge()
            summary_images_epoch = {}
            for neighbor_image_idx in neighbor_view_indices:
                mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out], depth_sample_num=self.depth_sample_num)
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
                summary_scalars, summary_images = self.optimize_per_image(rgb_mpi_ref_dict, sigma_mpi_ref_dict, neighbor_image_idx)

                summary_scalars_epoch.update(summary_scalars)
                if neighbor_image_idx == 0:
                    summary_images_epoch = summary_images

            if (epoch_idx+1) % self.summary_scalars_freq == 0:
                save_scalars(logger, "Optimize", summary_scalars_epoch.mean(), epoch_idx+1)  # scalars for random sampled tgt-view image
            if (epoch_idx+1) % self.summary_images_freq == 0:
                for scale in range(4):
                    save_images(logger, "Optimize_scale_{}".format(scale), summary_images_epoch["scale_{}".format(scale)], epoch_idx+1)  # summary images for random sampled tgt-image

            print("Optimize, Epoch:{}/{}, loss:{:.4f}".format(epoch_idx, self.epochs, summary_scalars_epoch.mean()["loss"]))

            if (epoch_idx+1) % self.save_ckpt_freq == 0:
                torch.save({
                            "epoch": epoch_idx,
                            "feature_generator": self.feature_generator.state_dict(),
                            "mpi_generator": self.mpi_generator.state_dict(),
                            "optimizer": self.optimizer.state_dict(),},
                            "{}/mpimodel_{:0>4}.ckpt".format(logdir, epoch_idx))
            self.lr_scheduler.step()

    def optimize_per_image(self, rgb_mpi_ref_dict, sigma_mpi_ref_dict, neighbor_image_idx):
        with torch.no_grad():
            T_ref_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],dtype=torch.float32, device=self.device).unsqueeze(0).repeat(1, 1, 1)
            T_tgt_ref = self.Ts_tgt_ref[:, neighbor_image_idx, :, :]

        summary_scalars, summary_images = {}, {}
        loss_per_image, loss_rgb_per_image, loss_ssim_per_image, loss_lpips_per_image = 0.0, 0.0, 0.0, 0.0
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
                depth_hypothesis_num=self.depth_sample_num,
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
                depth_hypothesis_num=self.depth_sample_num,
                T_tgt_ref=T_tgt_ref,
                K_ref=K_ref_scaled,
                K_tgt=K_tgt_scaled,
                height_render=height_render,
                width_render=width_render,
            )

            loss_rgb = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_rgb_weight
            loss_ssim = loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_ssim_weight
            loss_lpips = loss_fcn_rgb_lpips(self.lpips_calculator, tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_lpips_weight
            loss = loss_rgb + loss_ssim + loss_lpips

            loss_rgb_per_image = loss_rgb_per_image + loss_rgb
            loss_ssim_per_image = loss_ssim_per_image + loss_ssim
            loss_lpips_per_image = loss_lpips_per_image + loss_lpips
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
                "loss_lpips": loss_lpips_per_image.item(),
                # "depth_MAE": torch.mean(torch.abs(summary_images["scale_0"]["ref_depth_syn"] - summary_images["scale_0"]["ref_depth"]))
            }

        return summary_scalars, summary_images


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description="Per Scene Optimization")
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # data parameters
    parser.add_argument("--dataset", type=str, default="levir_nvs", help="select dataset")
    parser.add_argument("--dataset_dirpath", type=str, default=r"D:\Datasets\LEVIR_NVS", help="dataset directory path")
    parser.add_argument("--scene_id", type=str, default="scene_005", help="scene id in LEVIR-NVS")
    parser.add_argument("--sample_id", type=str, default="000", help="sample id in LEVIR-NVS per scene")
    # optimization parameters
    parser.add_argument("--epochs", type=int, default=500, help="optimization epoch number")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_ds_epoch_idx", type=str, default="50,100,200,300,400:2", help="epoch ids to downscale lr and the downscale rate")
    parser.add_argument("--ckpt_filepath", type=str, default="./checkpoints/ASI_prior.ckpt",
                        help="load a specific checkpoint")
    # log writer and random seed parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints/optimization/{}_{}", help="the directory to save optimize logs, tensorboard event log")
    parser.add_argument("--summary_scalars_freq", type=int, default=1, help="save summary scalar frequency")
    parser.add_argument("--summary_images_freq", type=int, default=10, help="save summary images frequency")
    parser.add_argument("--save_ckpt_freq", type=int, default=1, help="save checkpoint frequency, 1 means per epoch")
    parser.add_argument("--seed", type=int, default=23, metavar="S", help="random seed, ensure training can recurrence")
    # model parameters
    parser.add_argument("--depth_sample_num", type=int, default=32, help="depth sample number in decoder")
    parser.add_argument("--feature_generator_model_type", type=str, default="resnet18", help="feature generator model type")
    parser.add_argument("--neighbor_view_num", type=int, default=20, help="neighbor view number")
    # loss weights
    parser.add_argument("--loss_rgb_weight", type=float, default=2.0, help="loss rgb weight")
    parser.add_argument("--loss_ssim_weight", type=float, default=1.0, help="loss depth weight")
    parser.add_argument("--loss_lpips_weight", type=float, default=1.0, help="loss depth weight")

    args = parser.parse_args()
    print_args(args)
    # fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SceneData = find_scenedata_def(args.dataset)
    # model, decoder optimizer definition and load init network ckpt
    feature_generator = FeatureGenerator(model_type=args.feature_generator_model_type, pretrained=True, device=device).to(device)
    mpi_generator = MPIGenerator(feature_out_chs=feature_generator.encoder_channels).to(device)
    model = {"feature_generator": feature_generator, "mpi_generator": mpi_generator}

    scene_optimizer = OptimizePerScene(
        model=model,
        ckpt_filepath=args.ckpt_filepath,
        neighbor_view_num=args.neighbor_view_num,
        depth_sample_num=args.depth_sample_num,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_ds_epoch_idx=args.lr_ds_epoch_idx,
        loss_rgb_weight=args.loss_rgb_weight,
        loss_ssim_weight=args.loss_ssim_weight,
        loss_lpips_weight=args.loss_lpips_weight,
        summary_scalars_freq=args.summary_scalars_freq,
        summary_images_freq=args.summary_images_freq,
        save_ckpt_freq=args.save_ckpt_freq,
        device=device
    )

    # Optimization
    current_time = str(datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    print("Start time: ", current_time)

    scene_data = SceneData(args.dataset_dirpath, args.scene_id, args.sample_id, neighbor_view_num=args.neighbor_view_num).loadSceneData()
    neighbor_view_indices = list(range(0, 20, 2))   # training view indices
    logdir = args.logdir.format(scene_data["scene_id"], scene_data["sample_id"])
    # copy optimization config files
    shutil.copy(args.config, os.path.join(logdir, "config.txt"))
    scene_optimizer.optimize(scene_data, logdir, neighbor_view_indices)

    current_time = str(datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S'))
    print("End time: ", current_time)
