import torch
import torch.nn as nn
import numpy as np


def planeVolumeRendering(rgb_MPI, sigma_MPI, xyz_coor):
    """
    Rendering image, follow the equation of volume rendering process
    Args:
        rgb_MPI: rgb MPI representation, type:torch.Tensor, shape:[B, ndepth, 3, H, W]
        sigma_MPI: sigma MPI representation, type:torch.Tensor, shape:[B, ndepth, 1, H, W]
        xyz_coor: pixel2camera coordinates in camera coordinate, shape:[B, ndepth, 3, H, W]

    Returns:
        rgb_syn: synthetic RGB image, type:torch.Tensor, shape:[B, 3, H, W]
        depth_syn: synthetic depth, type:torch.Tensor, shape:[B, 1, H, W]
        transparency_acc: accumulated transparency, type:torch.Tensor, shape:[B, ndepth, 1, height, width]
        weights: render weights in per plane and per pixel, type:torch.Tensor, shape:[B, ndepth, 1, height, width]

    """
    B, ndepth, _, height, width = sigma_MPI.shape
    xyz_coor_diff = xyz_coor[:, 1:, :, :, :] - xyz_coor[:, :-1, :, :, :]    # [B, ndepth-1, 3, height, width]
    xyz_coor_diff = torch.norm(xyz_coor_diff, dim=2, keepdim=True)  # calculate distance, [B, ndepth-1, 1, height, width]
    xyz_coor_diff = torch.cat((xyz_coor_diff,
                               torch.full((B, 1, 1, height, width), fill_value=1e3, dtype=xyz_coor_diff.dtype, device=xyz_coor_diff.device)),
                              dim=1)    # [B, ndepth, 1, height, width]
    transparency = torch.exp(-sigma_MPI * xyz_coor_diff)    # [B, ndepth, 1, height, width]
    alpha = 1 - transparency    # [B, ndepth, 1, height, width]

    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)    # [B, ndepth, 1, height, width]
    transparency_acc = torch.cat((torch.ones((B, 1, 1, height, width), dtype=transparency_acc.dtype, device=transparency_acc.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1) # [B, ndepth, 1, height, width]

    weights = transparency_acc * alpha  # [B, ndepth, 1, height, width]

    # calculate rgb_syn, depth_syn
    rgb_syn = torch.sum(weights * rgb_MPI, dim=1, keepdim=False)    # [B, 3, height, width]
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # [B, 1, height, width]
    depth_syn = torch.sum(weights * xyz_coor[:, :, 2:, :, :], dim=1, keepdim=False) / (weights_sum + 1e-5)  # [B, 1, height, width]

    return rgb_syn, depth_syn, transparency_acc, weights


# def sampleDepth(depth_min, depth_max, depth_hypothesis_num):
#     """
#     Uniformly sample depth from [depth_min, depth_max]
#     Args:
#         depth_min: min depth value, type:torch.Tensor, shape:[B,]
#         depth_max: max depth value, type:torch.Tensor, shape:[B,]
#         depth_hypothesis_num: depth hypothesis number, type:int
#
#     Returns:
#         depth_sample: depth sample, type:torch.Tensor, shape:[B, ndepth]
#     """
#     depth_samples = []
#     for i in range(depth_min.shape[0]):
#         depth_samples.append(torch.linspace(start=depth_min[i], end=depth_max[i], steps=depth_hypothesis_num, device=depth_min.device))
#     depth_sample = torch.stack(depth_samples, dim=0)    # [B, ndepth]
#     return depth_sample


def sampleDepth(depth_min, depth_max, depth_hypothesis_num):
    """
    Uniformly sample depth from [inversed_depth_max, inversed_depth_max]
    Args:
        depth_min: min depth value, type:torch.Tensor, shape:[B,]
        depth_max: max depth value, type:torch.Tensor, shape:[B,]
        depth_hypothesis_num: depth hypothesis number, type:int

    Returns:
        depth_sample: depth sample, type:torch.Tensor, shape:[B, ndepth]
    """
    depth_samples = []
    for i in range(depth_min.shape[0]):
        depth_samples.append(torch.linspace(start=1.0/depth_min[i], end=1.0/depth_max[i], steps=depth_hypothesis_num, device=depth_min.device))
    depth_sample = torch.stack(depth_samples, dim=0)    # [B, ndepth]
    return 1.0 / depth_sample


def getRefXYZFromDepthSample(height, width, depth_sample, K):
    """
    Generate ref-view planes 3D position XYZ
    Args:
        height: rendered image height, type:int
        width: rendered image width, type:int
        depth_sample: depth sample in ref-view, corresponding to MPI planes' Z, type:torch.Tensor, shape:[B, ndepth]
        K: ref-camera intrinsic, type:torch.Tensor, shape:[B, 3, 3]

    Returns:
        XYZ_ref: 3D position in ref-camera, type:torch.Tensor, shape:[B, ndepth, 3, H, W]

    """
    device = K.device
    batch_size, ndepth = depth_sample.shape
    with torch.no_grad():
        K_inv = torch.inverse(K)    # [B, 3, 3], inversed intrinsics
    K_inv_expand = K_inv.unsqueeze(1).repeat(1, ndepth, 1, 1).reshape(batch_size*ndepth, 3, 3)  # [B*ndepth, 3, 3]

    # generate meshgrid for ref-view.
    x = np.linspace(0, width-1, num=width)
    y = np.linspace(0, height-1, num=height)
    xv, yv = np.meshgrid(x, y)    # [H, W]
    xv = torch.from_numpy(xv.astype(np.float32)).to(device)
    yv = torch.from_numpy(yv.astype(np.float32)).to(device)
    z = torch.ones_like(xv)
    meshgrid = torch.stack((xv, yv, z), dim=2)  # [H, W, 3]
    meshgrid = meshgrid.permute(2, 0, 1).contiguous()   # [3, H, W]
    meshgrid_expand = meshgrid.unsqueeze(0).unsqueeze(1).repeat(batch_size, ndepth, 1, 1, 1).reshape(batch_size*ndepth, 3, -1)  # [B*ndepth, 3, H*W]

    # calculate XYZ_ref
    XYZ_ref = torch.matmul(K_inv_expand, meshgrid_expand)   # [B*ndepth, 3, H*W]
    XYZ_ref = XYZ_ref.reshape(batch_size, ndepth, 3, height*width) * depth_sample.unsqueeze(2).unsqueeze(3) # [B, ndepth, 3, H*W]
    XYZ_ref = XYZ_ref.reshape(batch_size, ndepth, 3, height, width) # [B, ndepth, 3, H, W]

    return XYZ_ref


def transformXYZRef2Tgt(XYZ_ref, T_tgt_ref):
    """
    Transform points XYZ from ref-camera to tgt-camera
    Args:
        XYZ_ref: 3D position in ref-camera, type:torch.Tensor, shape:[B, ndepth, 3, H, W]
        T_tgt_ref: transfrom matrics from ref-camera to tgt-camera, type:torch.Tensor, shape:[B, 4, 4]

    Returns:
        XYZ_tgt: 3D position in tgt-camera, type:torch.Tensor, shape:[B, ndepth, 3, H, W]

    """
    batch_size, ndepth, _, height, width = XYZ_ref.shape
    T_tgt_ref_expand = T_tgt_ref.unsqueeze(1).repeat(1, ndepth, 1, 1).reshape(batch_size*ndepth, 4, 4)  # [B*ndepth, 4, 4]
    XYZ_ref = XYZ_ref.reshape(batch_size*ndepth, 3, -1) # [B*ndepth, 3, H*W]
    XYZ_ref_homogeneous = torch.cat((XYZ_ref, torch.ones_like(XYZ_ref[:, 0:1, :])), dim=1)
    XYZ_tgt_homogeneous = torch.matmul(T_tgt_ref_expand, XYZ_ref_homogeneous)   # [B*ndepth, 4, H*W]
    XYZ_tgt = XYZ_tgt_homogeneous[:, :3, :].reshape(batch_size, ndepth, 3, height, width)
    return XYZ_tgt


def homoWarpSample(MPI_xyz_ref, depth_sample_ref, T_tgt_ref, K_ref, K_tgt, height_render, width_render):
    """
    Homo warp MPI representation to tgt-camera, sample points along ray-marching
    Args:
        MPI_xyz_ref: ref-view MPI and XYZ representation, type:torch.Tensor, shape:[B, ndepth, 7, H, W]
        depth_sample_ref: depth sample in ref-view, corresponding to MPI planes' Z, type:torch.Tensor, shape:[B, ndepth]
        T_tgt_ref: transfrom matrics from ref-camera to tgt-camera, type:torch.Tensor, shape:[B, 4, 4]
        K_ref: ref-camera intrinsic, type:torch.Tensor, shape:[B, 3, 3]
        K_tgt: tgt-camera intrinsic, type:torch.Tensor, shape:[B, 3, 3]
        height_render: rendered image/depth height, type:int
        width_render: rendered image/depth width, type:int

    Returns:
        MPI_xyz_tgt: tgt-view MPI and XYZ representation, type:torch.Tensor, shape:[B, ndepth, 7, H_render, W_render]
        valid_mask: tgt-view homography mask, type:torch.Tensor, bool, shape:[B, ndepth, H_render, W_render]

    """
    device = MPI_xyz_ref.device
    batch_size, ndepth, _, height_mpi, width_mpi = MPI_xyz_ref.shape
    MPI_xyz_ref_reshaped = MPI_xyz_ref.reshape(batch_size*ndepth, 7, height_mpi, width_mpi)

    with torch.no_grad():
        K_ref_inv = torch.inverse(K_ref)
    K_ref_inv_expand = K_ref_inv.unsqueeze(1).repeat(1, ndepth, 1, 1).contiguous().reshape(batch_size*ndepth, 3, 3)  # [B*ndepth, 3, 3]
    K_tgt_expand = K_tgt.unsqueeze(1).repeat(1, ndepth, 1, 1).contiguous().reshape(batch_size*ndepth, 3, 3)  # [B*ndepth, 3, 3]
    T_tgt_ref_expand = T_tgt_ref.unsqueeze(1).repeat(1, ndepth, 1, 1).contiguous().reshape(batch_size*ndepth, 4, 4) # [B*ndepth, 4, 4]

    R_tgt_ref = T_tgt_ref_expand[:, 0:3, 0:3]   # [B*ndepth, 3, 3]
    t_tgt_ref = T_tgt_ref_expand[:, 0:3, 3] # [B*ndepth, 3]

    normal_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    normal_vector_expand = normal_vector.unsqueeze(0).repeat(batch_size*ndepth, 1)  # [B*ndepth, 3]
    # note here use -d_ref, cause the plane function is n^T * X - d_ref = 0
    depth_sample_ref_expand = depth_sample_ref.reshape(batch_size*ndepth, 1, 1).repeat(1, 3, 3) # [B*ndepth, 3, 3]
    R_tnd = R_tgt_ref - torch.matmul(t_tgt_ref.unsqueeze(2), normal_vector_expand.unsqueeze(1)) / (-depth_sample_ref_expand)    # [B*ndepth, 3, 3]
    H_tgt_ref = torch.matmul(K_tgt_expand, torch.matmul(R_tnd, K_ref_inv_expand))   # [B*ndepth, 3, 3]
    with torch.no_grad():
        H_ref_tgt = torch.inverse(H_tgt_ref)

    # generate meshgrid for tgt-view.
    x = np.linspace(0, width_render-1, num=width_render)
    y = np.linspace(0, height_render-1, num=height_render)
    xv, yv = np.meshgrid(x, y)    # [H_render, W_render]
    xv = torch.from_numpy(xv.astype(np.float32)).to(device)
    yv = torch.from_numpy(yv.astype(np.float32)).to(device)
    z = torch.ones_like(xv)
    meshgrid = torch.stack((xv, yv, z), dim=2)  # [H_render, W_render, 3]
    meshgrid = meshgrid.permute(2, 0, 1).contiguous()   # [3, H_render, W_render]
    meshgrid_tgt_homo_expand = meshgrid.unsqueeze(0).unsqueeze(1).repeat(batch_size, ndepth, 1, 1, 1).reshape(batch_size*ndepth, 3, -1)  # [B*ndepth, 3, H_render*W_render]

    # warp meshgrid tgt_homo to meshgrid ref
    meshgrid_ref_homo_expand = torch.matmul(H_ref_tgt, meshgrid_tgt_homo_expand)    # [B*ndepth, 3, H_render*W_render]
    meshgrid_ref_homo = meshgrid_ref_homo_expand.reshape(batch_size*ndepth, 3, height_render, width_render).permute(0, 2, 3, 1)    #[B*ndepth, H_render, W_render, 3]
    meshgrid_ref = meshgrid_ref_homo[:, :, :, 0:2] / meshgrid_ref_homo[:, :, :, 2:] # [B*ndepth, H_render, W_render, 2]

    valid_mask_x = torch.logical_and(meshgrid_ref[:, :, :, 0] < width_mpi, meshgrid_ref[:, :, :, 0] > -1)
    valid_mask_y = torch.logical_and(meshgrid_ref[:, :, :, 1] < height_mpi, meshgrid_ref[:, :, :, 1] > -1)
    valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # [B*ndepth, H_render, W_render]
    valid_mask = valid_mask.reshape(batch_size, ndepth, height_render, width_render)  # [B, ndepth, H_render, W_render]

    # sample from MPI_xyz_ref
    # normalize meshgrid_ref coordinate to [-1, 1]
    meshgrid_ref[:, :, :, 0] = (meshgrid_ref[:, :, :, 0]+0.5) / (width_mpi * 0.5) - 1
    meshgrid_ref[:, :, :, 1] = (meshgrid_ref[:, :, :, 1]+0.5) / (height_mpi * 0.5) - 1
    MPI_xyz_tgt = torch.nn.functional.grid_sample(MPI_xyz_ref_reshaped, grid=meshgrid_ref, padding_mode="border", align_corners=False)   # [B*ndepth, 7, H_render, W_render]
    MPI_xyz_tgt = MPI_xyz_tgt.reshape(batch_size, ndepth, 7, height_render, width_render)   # [B, ndepth, 7, H_render, W_render]

    return MPI_xyz_tgt, valid_mask


def renderNovelView(rbg_MPI_ref, sigma_MPI_ref, depth_min_ref, depth_max_ref, depth_hypothesis_num, T_tgt_ref, K_ref, K_tgt, height_render, width_render):
    """
    Render novel view using decoder output, rgb_MPI, sigma_MPI
    Args:
        rbg_MPI_ref: decoder output, rgb MPI representation in ref-view, type:torch.Tensor, shape:[B, ndepth, 3, height, width]
        sigma_MPI_ref: decoder output, sigma MPI representation in ref-view, type:torch.Tensor, shape:[B, ndepth, 1, height, width]
        depth_min_ref: ref_view depth min, type:torch.Tensor, shape:[B,]
        depth_min_ref: ref_view depth max, type:torch.Tensor, shape:[B,]
        depth_hypothesis_num: depth hypothesis number, type:int
        T_tgt_ref: transform matrix from tgt-camera to ref_camera, type:torch.Tensor, shape:[B, 4, 4]
        K_ref: intrinsic of ref-camera, type:torch.Tensor, shape:[B, 3, 3]
        K_tgt: intrinsic of tgt-camera, type:torch.Tensor, shape:[B, 3, 3]
        height_render: rendered image/depth height, type:int
        width_render: rendered image/depth width, type:int
    Returns:
        rgb_tgt_syn: rgb image rendered in tgt-view, type:torch.Tensor, shape:[B, 3, height, width]
        depth_sample_tgt_syn: tgt depth sample corresponding to depth_hypothesis_ref, type:torch.Tensor, shape:[B, 1, H, W]
        mask_tgt_syn: rendered mask in tgt-view, type:torch.Tensor, shape:[B, 1, height, width]
    """
    device = rbg_MPI_ref.device
    batch_size, ndepth, _, height_mpi, width_mpi = rbg_MPI_ref.shape

    # depth sample
    depth_sample_ref = sampleDepth(depth_min_ref, depth_max_ref, depth_hypothesis_num)  # [B, ndepth]
    # get each MPI 3D position in ref-camera, these points is reconstruction result, and transform these point to tgt-camera
    XYZ_ref = getRefXYZFromDepthSample(height_mpi, width_mpi, depth_sample_ref, K_ref)    # [B, ndepth, 3, H_mpi, W_mpi]
    XYZ_tgt = transformXYZRef2Tgt(XYZ_ref, T_tgt_ref)   # [B, ndepth, 3, H_mpi, W_mpi]

    # calculate MPI representation coordinates in tgt-camera, ray sample XYZ points, get tgt_MPI and tgt_mask
    MPI_xyz_ref = torch.cat((rbg_MPI_ref, sigma_MPI_ref, XYZ_tgt), dim=2)   # [B, ndepth, 3+1+3, H_mpi, W_mpi]
    tgt_MPI_XYZ, tgt_mask = homoWarpSample(MPI_xyz_ref, depth_sample_ref, T_tgt_ref, K_ref, K_tgt, height_render, width_render) # [B, ndepth, 3+1+3, H_render, W_render], [B, ndepth, H_render, W_render]
    tgt_MPI_rgb = tgt_MPI_XYZ[:, :, 0:3, :, :]   # [B, ndepth, 3, H_render, W_render]
    tgt_MPI_sigma = tgt_MPI_XYZ[:, :, 3:4, :, :]    # [B, ndepth, 1, H_render, W_render]
    tgt_XYZ_warped = tgt_MPI_XYZ[:, :, 4:, :, :]    # [B, ndepth, 3, H_render, W_render]
    tgt_mask = torch.where(tgt_mask,
                           torch.ones((batch_size, ndepth, height_render, width_render), dtype=torch.float32, device=device),
                           torch.zeros((batch_size, ndepth, height_render, width_render), dtype=torch.float32, device=device))    # [B, ndepth, H, W]
    tgt_warped_Z = tgt_XYZ_warped[:, :, -1:]    # [B, ndepth, 1, H_render, W_render]
    tgt_MPI_sigma = torch.where(tgt_warped_Z >= 0,
                                tgt_MPI_sigma,
                                torch.zeros_like(tgt_MPI_sigma, device=device)) # [B, ndepth, 1, H_render, W_render]
    tgt_rgb_syn, tgt_depth_syn, tgt_transparency_acc, tgt_weights = planeVolumeRendering(tgt_MPI_rgb, tgt_MPI_sigma, tgt_XYZ_warped)
    tgt_mask = torch.sum(tgt_mask, dim=1, keepdim=True) # [B, 1, H_render, W_render], when all plane is not visible, mask value equal to zero

    # binary thresh mask
    tgt_mask = torch.where(tgt_mask > 0,
                           torch.ones((batch_size, 1, height_render, width_render), dtype=torch.float32, device=device),
                           torch.zeros((batch_size, 1, height_render, width_render), dtype=torch.float32, device=device))

    return tgt_rgb_syn, tgt_depth_syn, tgt_mask



if __name__ == '__main__':
    from utils import *
    import cv2
    import torch.nn.functional as F

    def visualizeRenderRGB(tgt_rgb_syn, window_name="rgb_render"):
        """
        Visualize tgt-view rendered rgb image
        Args:
            tgt_rgb_syn: tgt-view rgb image rendered from renderNovelView, type:torch.Tensor, shape:[B, 3, H, W], only support B=1
            window_name: window name, type:str

        Returns: None

        """
        tgt_rgb_syn = tgt_rgb_syn.squeeze().permute(1, 2, 0).to("cpu").numpy()
        tgt_rgb_syn = (tgt_rgb_syn * 255.).astype(np.uint8)
        tgt_rgb_syn = cv2.cvtColor(tgt_rgb_syn, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, tgt_rgb_syn)

    def visualizeRenderMask(tgt_mask, window_name="mask_render"):
        """
        Visualize tgt-view rendered mask image
        Args:
            tgt_mask: tgt-view mask rendered from renderNovelView, type:torch.Tensor, shape:[B, 1, H, W], only support B=1
            window_name: window name, type:str

        Returns: None

        """
        tgt_mask = tgt_mask.squeeze().to("cpu").numpy()
        tgt_mask = tgt_mask.astype(np.uint8) * 255
        cv2.imshow(window_name, tgt_mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neighbor_view_num = 4
    dataset_dirpath = "../testdata/whu_debug"
    depth_sample_num = 32
    mpi_ratio = 1.0 # MPI representation height/width with respect to init image height/width
    render_ratio = 1.0  # rendered image height/width with respect to init image height/width

    sigma_mpi_ref = torch.tensor(np.load("../testdata/sigma_mpi_ref_009_53_002002.npy")).unsqueeze(0).to(device)
    rgb_mpi_ref = torch.tensor(np.load("../testdata/rgb_mpi_ref_009_53_002002.npy")).unsqueeze(0).to(device)

    height, width = 384, 768
    height_mpi, width_mpi = int(height * mpi_ratio), int(width * mpi_ratio)
    assert height_mpi == sigma_mpi_ref.shape[3] and width_mpi == sigma_mpi_ref.shape[4]
    height_render, width_render = int(height * render_ratio), int(width * render_ratio)

    scene_id, sample_id = "009_53", "002002"
    scene_data = SceneData(dataset_dirpath, scene_id, sample_id, neighbor_view_num=neighbor_view_num).loadSceneData()
    if device == torch.device("cuda"):
        scene_data = dict2cuda(scene_data)
    image_ref, depth_min_ref, depth_max_ref, K_ref, depth_ref = scene_data["image_ref"], scene_data["depth_min_ref"], scene_data[
        "depth_max_ref"], \
                                                                scene_data["K_ref"], scene_data["depth_ref"]
    images_tgt, Ks_tgt, Ts_tgt_ref = scene_data["images_tgt"], scene_data["Ks_tgt"], scene_data["Ts_tgt_ref"]
    # rescale intrinsics and resize images_tgt and depth_ref with respect to ratio
    with torch.no_grad():
        images_tgt = torch.stack(
            [F.interpolate(images_tgt[:, i, :, :, :], size=None, scale_factor=mpi_ratio, mode="bilinear", align_corners=False) for i in
             range(neighbor_view_num)],
            dim=1)
        depth_ref = F.interpolate(depth_ref.unsqueeze(1), size=None, scale_factor=mpi_ratio, mode="nearest")
        K_ref_render = K_ref.clone()
        K_ref_render[:, :2, :] = K_ref_render[:, :2, :] * render_ratio
        Ks_tgt[:, :, :2, :] = Ks_tgt[:, :, :2, :] * render_ratio
        K_ref[:, :2, :] = K_ref[:, :2, :] * mpi_ratio

    # render tgt-view image
    novel_view_idx = 0
    T_tgt_ref, K_tgt = Ts_tgt_ref[:, novel_view_idx, :, :], Ks_tgt[:, novel_view_idx, :, :]
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
    visualizeRenderRGB(tgt_rgb_syn[:1], "tgt_rgb_render")
    visualizeRenderMask(tgt_mask[:1], "tgt_mask_render")

    # render ref_view
    T_ref_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(1, 1, 1)
    tgt_rgb_syn, tgt_depth_syn, tgt_mask = renderNovelView(
        rbg_MPI_ref=rgb_mpi_ref,
        sigma_MPI_ref=sigma_mpi_ref,
        depth_min_ref=depth_min_ref,
        depth_max_ref=depth_max_ref,
        depth_hypothesis_num=depth_sample_num,
        T_tgt_ref=T_ref_ref,
        K_ref=K_ref,
        K_tgt=K_ref_render,
        height_render=height_render,
        width_render=width_render,
    )
    visualizeRenderRGB(tgt_rgb_syn, "ref_rgb_render")
    visualizeRenderMask(tgt_mask, "ref_mask_render")

    cv2.waitKey(0)
