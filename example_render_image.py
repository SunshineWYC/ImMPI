import cv2
from datasets import find_scenedata_def
from models.mpi_generator import MPIGenerator
from models.feature_generator import FeatureGenerator
from utils.utils import *
from utils.render import renderNovelView


def ViewSynthetic(ckpt_filepath, scene_data, tgt_view_indices, tgt_view_ids, depth_sample_num, output_dirpath, device):
    output_scene_dirpath = os.path.join(output_dirpath, "{}_{}".format(scene_data["scene_id"], scene_data["sample_id"]))
    output_scene_mask_dirpath = os.path.join(output_scene_dirpath, "Masks_SYN")
    output_scene_image_dirpath = os.path.join(output_scene_dirpath, "Images_SYN")
    output_scene_depth_dirpath = os.path.join(output_scene_dirpath, "Depths_SYN")
    if not os.path.exists(output_scene_dirpath):
        os.makedirs(output_scene_dirpath, exist_ok=True)
        os.makedirs(output_scene_mask_dirpath, exist_ok=True)
        os.makedirs(output_scene_image_dirpath, exist_ok=True)
        os.makedirs(output_scene_depth_dirpath, exist_ok=True)

    with torch.no_grad():
        # model definition and load ckpt
        feature_generator = FeatureGenerator(model_type="resnet18", pretrained=True, device=device).to(device)
        mpi_generator = MPIGenerator(feature_out_chs=feature_generator.encoder_channels).to(device)
        state_dict = torch.load(ckpt_filepath)
        feature_generator.load_state_dict(state_dict["feature_generator"])
        mpi_generator.load_state_dict(state_dict["mpi_generator"])

        if device == torch.device("cuda"):
            sample = dict2cuda(scene_data)
        image_ref, depth_min_ref, depth_max_ref, K_ref, depth_ref = sample["image_ref"], sample["depth_min_ref"], sample["depth_max_ref"], sample["K_ref"], sample["depth_ref"].unsqueeze(1)
        images_tgt, Ks_tgt, Ts_tgt_ref = sample["images_tgt"], sample["Ks_tgt"], sample["Ts_tgt_ref"]
        height_render, width_render = image_ref.shape[2], image_ref.shape[3]

        conv1_out, block1_out, block2_out, block3_out, block4_out = feature_generator(image_ref)
        mpi_outputs = mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out], depth_sample_num=depth_sample_num)
        rgb_mpi_ref = mpi_outputs["MPI_0"][:, :, :3, :, :]
        sigma_mpi_ref = mpi_outputs["MPI_0"][:, :, 3:, :, :]

        # render neighbour-view syn result
        for i in tgt_view_indices:
            T_tgt_ref, K_tgt = Ts_tgt_ref[:, i, :, :], Ks_tgt[:, i, :, :]
            tgt_rgb_syn, tgt_depth_syn, tgt_mask = renderNovelView(
                rbg_MPI_ref=rgb_mpi_ref,
                sigma_MPI_ref=sigma_mpi_ref,
                depth_min_ref=depth_min_ref,
                depth_max_ref=depth_max_ref,
                depth_hypothesis_num=depth_sample_num,
                T_tgt_ref=T_tgt_ref,
                K_ref=K_ref,
                K_tgt=K_tgt,
                height_render=height_render,
                width_render=width_render,
            )

            tgt_depth_syn = tgt_depth_syn.squeeze().to("cpu").numpy()
            cv2.imwrite(os.path.join(output_scene_depth_dirpath, "{}.tiff".format(tgt_view_ids[i])), tgt_depth_syn)
            image_rendered = tgt_rgb_syn.squeeze().permute(1, 2, 0).to("cpu").numpy()
            image_rendered = (image_rendered * 255.).astype(np.uint8)
            image_rendered = cv2.cvtColor(image_rendered, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_scene_image_dirpath, "{}.png".format(tgt_view_ids[i])), image_rendered)
            mask_rendered = tgt_mask.squeeze().to("cpu").numpy()
            mask_rendered = mask_rendered.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_scene_mask_dirpath, "{}.png".format(tgt_view_ids[i])), mask_rendered)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Render Train-View and Test-view with optimized model
    SceneData = find_scenedata_def("levir_nvs")
    dataset_dirpath = r"D:\Datasets\LEVIR_NVS"
    depth_sample_num = 32
    neighbor_view_num = 20
    scene_ids = ["scene_{:03d}".format(i) for i in range(0, 16)]
    sample_id = "000"
    for scene_id in scene_ids:
        print("scene id: {}".format(scene_id))
        output_dirpath = "./output/syntheticImage/Levir_NVS"
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath, exist_ok=True)
        ckpt_filepath = r"./checkpoints/optimization/{}_{}/optimizedImMPI.ckpt".format(scene_id, sample_id)
        scene_data = SceneData(dataset_dirpath, scene_id, sample_id, neighbor_view_num).loadSceneData()
        tgt_view_indices = list(range(0, neighbor_view_num))
        tgt_view_ids = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "020"]

        ViewSynthetic(ckpt_filepath, scene_data, tgt_view_indices, tgt_view_ids, depth_sample_num, output_dirpath, device)
