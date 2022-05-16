import cv2
import configargparse
from datasets import find_scenedata_def
from models.mpi_generator import MPIGenerator
from models.feature_generator import FeatureGenerator
from utils.utils import *
from utils.render import renderNovelView
from PIL import Image


def readCameraFile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsic = np.fromstring(",".join(lines[1:5]), dtype=np.float32, sep=",").reshape(4, 4)
    intrinsic = np.fromstring(",".join(lines[7:10]), dtype=np.float32, sep=",").reshape(3, 3)
    depth_min, depth_max = 0.0, 0.0
    if len(lines) >= 11:
        depth_min, depth_max = [float(item) for item in lines[11].split(",")]
    return intrinsic, extrinsic, depth_min, depth_max


def getTgtViewsCameraParams(track_dirpath, track_viewpoints_num, E_ref):
    Ks_tgt, Ts_tgt_ref = [], []
    for i in range(1, track_viewpoints_num+1):
        K_tgt, E_tgt, depth_min, depth_max = readCameraFile(os.path.join(track_dirpath, "{:03d}.txt".format(i)))
        Ks_tgt.append(K_tgt)
        Ts_tgt_ref.append(np.matmul(E_tgt, np.linalg.inv(E_ref)))
    Ks_tgt = np.stack(Ks_tgt)
    Ts_tgt_ref = np.stack(Ts_tgt_ref)

    Ks_tgt = torch.from_numpy(Ks_tgt).unsqueeze(0).to(torch.float32)
    Ts_tgt_ref = torch.from_numpy(Ts_tgt_ref).unsqueeze(0)

    return Ks_tgt, Ts_tgt_ref


def VideoSynthetic(ckpt_filepath, ref_data, Ks_tgt, Ts_tgt_ref, device, args):
    images_rendered, masks_rendered = [], []
    with torch.no_grad():
        # model definition and loadckpt
        feature_generator = FeatureGenerator(model_type=args.feature_generator_model_type, pretrained=True, device=device).to(device)
        mpi_generator = MPIGenerator(feature_out_chs=feature_generator.encoder_channels).to(device)
        state_dict = torch.load(ckpt_filepath)
        feature_generator.load_state_dict(state_dict["feature_generator"])
        mpi_generator.load_state_dict(state_dict["mpi_generator"])

        if device == torch.device("cuda"):
            ref_data = dict2cuda(ref_data)
            Ks_tgt = Ks_tgt.to(device)
            Ts_tgt_ref = Ts_tgt_ref.to(device)
        image_ref, depth_min_ref, depth_max_ref, K_ref = ref_data["image_ref"], ref_data["depth_min_ref"], ref_data["depth_max_ref"], ref_data["K_ref"]
        height_render, width_render = image_ref.shape[2], image_ref.shape[3]

        conv1_out, block1_out, block2_out, block3_out, block4_out = feature_generator(image_ref)
        mpi_outputs = mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out], depth_sample_num=args.depth_sample_num)
        rgb_mpi_ref = mpi_outputs["MPI_0"][:, :, :3, :, :]
        sigma_mpi_ref = mpi_outputs["MPI_0"][:, :, 3:, :, :]

        # render neighbour-view syn result
        for i in range(args.track_viewpoint_num):
            T_tgt_ref, K_tgt = Ts_tgt_ref[:, i, :, :], Ks_tgt[:, i, :, :]
            tgt_rgb_syn, _, tgt_mask = renderNovelView(
                rbg_MPI_ref=rgb_mpi_ref,
                sigma_MPI_ref=sigma_mpi_ref,
                depth_min_ref=depth_min_ref,
                depth_max_ref=depth_max_ref,
                depth_hypothesis_num=args.depth_sample_num,
                T_tgt_ref=T_tgt_ref,
                K_ref=K_ref,
                K_tgt=K_tgt,
                height_render=height_render,
                width_render=width_render,
            )

            image_rendered = tgt_rgb_syn.squeeze().permute(1, 2, 0).to("cpu").numpy()
            image_rendered = (image_rendered * 255.).astype(np.uint8)
            image_rendered = cv2.cvtColor(image_rendered, cv2.COLOR_RGB2BGR)
            mask_rendered = tgt_mask.squeeze().to("cpu").numpy()
            mask_rendered = mask_rendered.astype(np.uint8) * 255

            images_rendered.append(image_rendered)
            masks_rendered.append(mask_rendered)

    return images_rendered, masks_rendered


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description="Per Scene Optimization")
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # data parameters
    parser.add_argument("--dataset", type=str, default="levir_nvs", help="select dataset")
    parser.add_argument("--dataset_dirpath", type=str, default=r"D:\Datasets\LEVIR_NVS", help="dataset directory path")
    parser.add_argument("--scene_id", type=str, default="scene_000", help="scene id")
    parser.add_argument("--sample_id", type=str, default="000", help="reference view id")
    # model parameters
    parser.add_argument("--depth_sample_num", type=int, default=32, help="depth sample number in decoder")
    parser.add_argument("--feature_generator_model_type", type=str, default="resnet18", help="feature generator model type")
    parser.add_argument("--neighbor_view_num", type=int, default=20, help="neighbor view number")
    # IO setting
    parser.add_argument("--ckpt_filepath", type=str, default="./checkpoints/optimization", help="load a optimized checkpoint")
    parser.add_argument("--output", type=str, default="./output/track_video", help="video track output dirpath")
    parser.add_argument("--track_viewpoint_num", type=int, default=100, help="video track output dirpath")
    parser.add_argument("--track_filepath", type=str, default="Track", help="track dirpath")

    args = parser.parse_args()
    print_args(args)

    print("Start Rendering Video along Camera Track for {}".format(args.scene_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read reference view data, image_ref, K_ref, E_ref, depth_min, depth_max
    K_ref, E_ref, depth_min_ref, depth_max_ref = readCameraFile(os.path.join(args.dataset_dirpath, args.scene_id, "Cams", "{}.txt".format(args.sample_id)))
    Ks_tgt, Ts_tgt_ref = getTgtViewsCameraParams(os.path.join(args.dataset_dirpath, args.scene_id, args.track_filepath), args.track_viewpoint_num, E_ref)
    image_ref = Image.open(os.path.join(args.dataset_dirpath, args.scene_id, "Images", "{}.png".format(args.sample_id)))
    image_ref = (np.array(image_ref, dtype=np.float32) / 255.).transpose([2, 0, 1])  # CHW
    image_ref = torch.from_numpy(image_ref).unsqueeze(0)
    K_ref = torch.from_numpy(K_ref).unsqueeze(0)
    E_ref = torch.from_numpy(E_ref).unsqueeze(0)
    depth_min_ref = torch.tensor(depth_min_ref, dtype=torch.float32).unsqueeze(0)
    depth_max_ref = torch.tensor(depth_max_ref, dtype=torch.float32).unsqueeze(0)
    ref_data = dict(
        image_ref=image_ref,
        K_ref=K_ref,
        E_ref=E_ref,
        depth_min_ref=depth_min_ref,
        depth_max_ref=depth_max_ref,
    )

    ckpt_filepath = os.path.join(args.ckpt_filepath, "{}_{}/optimizedImMPI.ckpt".format(args.scene_id, args.sample_id))
    images_rendered, masks_rendered = VideoSynthetic(ckpt_filepath, ref_data, Ks_tgt, Ts_tgt_ref, device, args)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    video_writer = cv2.VideoWriter(os.path.join(args.output, "{}_track.mp4".format(args.scene_id)), fourcc, 10, (512, 512))
    for i in range(len(images_rendered)):
        mask = masks_rendered[i]
        ret, mask = cv2.threshold(mask, 100, 1, cv2.THRESH_BINARY)
        mask = np.stack([mask, mask, mask], axis=2)
        image = images_rendered[i]
        image_masked = image * mask
        video_writer.write(image_masked)

    video_writer.release()
    print("Video render finished for {}".format(args.scene_id))
