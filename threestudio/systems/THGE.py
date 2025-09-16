from dataclasses import dataclass, field

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
import threestudio
import os, tempfile
from threestudio.systems.base import BaseLift3DSystem

from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor

import matplotlib.pyplot as plt
import clip

import smplx
import trimesh, pickle
import open3d as o3d

from pytorch3d.io import load_obj
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes, Pointclouds  # Meshes 仍用于射线法
# from pytorch3d.ops import _PointFaceDistance
from pytorch3d.loss.point_mesh_distance import point_face_distance 
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import math
from gaussiansplatting.gaussian_renderer import (
    GaussianRasterizer, GaussianRasterizationSettings
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d
from collections import defaultdict
from pathlib import Path
from collections import Counter
import ImageReward as RM
class MatrixCam(CamerasBase):
    """
    Wrap 3D‑Gaussian camera matrices for PyTorch3D:

        * world_view_transform …… 4×4  (world → cam)
        * full_proj_transform  …… 4×4  (cam   → clip)

    No extra intrinsics or FoV needed.
    """
    def __init__(self, w2c: torch.Tensor, proj: torch.Tensor, H: int, W: int):
        super().__init__(device=w2c.device)
        self.register_buffer("w2c",  w2c.float()[None])   # (1,4,4)
        self.register_buffer("proj", proj.float()[None])  # (1,4,4)
        self._H, self._W = H, W
        self._N = 1

    # --- PyTorch3D 接口 ---
    def get_world_to_view_transform(self, **kw):      # world → camera
        return Transform3d(matrix=self.w2c, device=self.device)

    def get_full_projection_transform(self, **kw):    # camera → clip
        return Transform3d(matrix=self.proj, device=self.device)

    get_projection_transform = get_full_projection_transform
    def in_ndc(self):     return True                 # 已经是 NDC
    def is_perspective(self): return True
    @property
    def R(self): return self.w2c[:, :3, :3]
    @property
    def T(self): return self.w2c[:, :3, 3]
    @property
    def image_size(self): return [(self._H, self._W)]

    get_full_projection_transform = get_projection_transform

    def in_ndc(self) -> bool:
        # 告诉 P3D：当前 projection 仍在 Clip 空间，需要自动做 Clip→NDC
        return False

    def is_perspective(self) -> bool:
        return True

    # PyTorch3D 用于屏幕‑到‑NDC 的 xyscale 计算
    @property
    def image_size(self):
        return [(self._H, self._W)]

    # 与 CamerasBase 属性一致
    @property
    def R(self):
        return self.w2c[..., :3, :3]

    @property
    def T(self):
        return self.w2c[..., :3, 3]
    
def check_consistency(pix2shell_extra, res_list=[64,32,16,8]):
    stats = {}
    for R in res_list:
        vids = torch.stack(pix2shell_extra[R], dim=0)   # (V, R, R)
        V, H, W = vids.shape
        uniq_map = torch.zeros((H, W), dtype=torch.int16)

        for i in range(H):
            for j in range(W):
                ids = vids[:, i, j].view(-1).cpu().numpy()
                ids = ids[ids >= 0]      # 去掉 -1
                uniq_map[i, j] = len(np.unique(ids))

        # 统计
        total   = H * W
        perfect = (uniq_map == 1).sum().item()
        stats[R] = {
            "perfect_ratio": perfect / total,
            "uniq_hist": Counter(uniq_map.view(-1).cpu().tolist()),
            "uniq_map": uniq_map,        # 留给可视化
        }
        print(f"{R}×{R}:  完全一致 {perfect}/{total} = {perfect/total:.2%}")
    return stats

def build_smpl_sdf(obj_path: str, cache_path: str, pts: torch.Tensor):
    """
    Return signed distance (meters) from each pts(N,3) to SMPL mesh surface.
    Cache to disk after first build.
    """
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=pts.device)

    # 加载网格
    verts, faces_idx, _ = load_obj(obj_path, load_textures=False)
    faces = faces_idx.verts_idx  # (F,3)

    # 移动到相同设备并确保 long 类型
    verts = verts.to(pts.device)
    faces = faces.to(pts.device).long()


    # 构造三角形顶点 (F, 3, 3)
    tris = verts[faces]  # :contentReference[oaicite:1]{index=1}

    # PyTorch3D 要求的 batch 索引
    # 这里我们只有一个“batch”（整个 pts 作为一个样本）
    points_first_idx = torch.tensor([0], dtype=torch.int64, device=pts.device)
    tris_first_idx  = torch.tensor([0], dtype=torch.int64, device=pts.device)
    max_points      = pts.shape[0]

    # 2. 点到面的平方距离（返回 (P,)）
    dists_sq = point_face_distance(
        pts,                # (P,3)
        points_first_idx,   # (1,)
        tris,               # (F,3,3)
        tris_first_idx,     # (1,)
        max_points          # scalar
    )
    # 取平方根得到真实距离 (P,)
    dist = torch.sqrt(dists_sq)

    # 后续内部外部判断沿用原来三角射线法
    mesh_tri = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
        process=False
    )
    ray_orig = pts.cpu().numpy()
    ray_dir  = np.tile([0, 0, 1.0], (len(ray_orig), 1))
    inside   = mesh_tri.ray.intersects_any(ray_orig, ray_dir)
    signed   = torch.from_numpy(inside).to(pts.device)
    sdf      = dist * (signed * -2 + 1)  # inside -> negative

    if cache_path:
        torch.save(sdf, cache_path)
    return sdf


@threestudio.register("THGE-system")
class THGE(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None
        obj_source : str = ""
        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000

        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.6
        mask_thres_2: float = 0.65
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""
        seg_prompt2: str = ""

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""


        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        
        training_args: dict = field(default_factory=dict)

        use_masked_image: bool = True 
        local_edit: bool = False

        # guidance 
        camera_update_per_step: int = 500
        added_noise_schedule: List[int] = field(default_factory=[999, 200, 200, 21])    

        seg_prompt_editor: str =""
        seg_text_computer: str =""
        human_prompt: str = "human"
        seg_prompt_hands : str ="finger"
        edi_view: int = 200
        top_view_num: int = 10
        max_clothe: bool = False
        sdf_thin: float = 0
        sdf_thick: float = 0.02
        offset : float = 0.02
    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        bg_color =  [0, 0, 0]
        # bg_color = [1, 1, 1]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())

        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=get_device())
        # self.clip_model.eval()
        self.clip_text_feature_cache = {}

        self.cache_mask = []

        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))

        # === Method‑1: build / load SMPL‑SDF once ===========================

        self.T_IN, self.T_OUT = 0.02, 0.03
        # ================================================================

        if not hasattr(self, "image_reward_model"):
            
            self.image_reward_model = RM.load("/root/autodl-tmp/ImageReward/ImageReward.pt")

    def build_pix2vert_maps(self, overwrite=False):

        save_path  = os.path.join(self.cache_dir, "pix2vert.pt")
        multi_path = os.path.join(self.cache_dir, "pix2shell_multi.pt")
        extra_res  = [64, 32, 16, 8]


        if os.path.exists(save_path) and os.path.exists(multi_path) and not overwrite:
            self.pix2vert_all    = torch.load(save_path,  map_location="cpu")
            self.pix2shell_extra = torch.load(multi_path, map_location="cpu")
            print(f"[Pix2Vert] ✓ loaded cached maps")
            return


        device   = get_device()
        H = self.trainer.datamodule.train_dataset.height     # 512
        W = self.trainer.datamodule.train_dataset.width
        cams   = self.trainer.datamodule.train_dataset.scene.cameras
        obj_path = Path(self.cfg.obj_source)/"rigid.obj"


        verts, _, _ = load_obj(obj_path, load_textures=False)
        verts = verts.to(device)                           # (V,3)
        V     = verts.shape[0]
        means3D = verts
        means2D = torch.zeros_like(means3D)
        scales  = torch.full((V, 3), 1e-3, device=device)
        rot     = torch.zeros((V, 4), device=device); rot[:, 0] = 1
        opa     = torch.ones(V, device=device)

        ids    = torch.arange(V, device=device)
        colors = torch.zeros(V, 3, device=device)
        colors[:, 0] = torch.floor(ids / 256) / 255.0
        colors[:, 1] = (ids % 256) / 255.0

        gauss_kwargs = dict(
            means3D       = means3D.float(),
            means2D       = means2D.float(),
            opacities     = opa.float(),
            scales        = scales.float(),
            rotations     = rot.float(),
            cov3D_precomp = None,
            shs           = None,
            colors_precomp= colors.float(),
        )


        pix2vert_all    = []                              # 512×512
        pix2shell_extra = {R: [] for R in extra_res}      # R×R  int16
        seeds_dict      = {R: set() for R in extra_res}   

        print(f"[Pix2Vert] Rasterizing {len(cams)} views via 3DGS …")
        for cam_idx, cam in enumerate(tqdm(cams)):
            tanfovx = math.tan(cam.FoVx * 0.5)
            tanfovy = math.tan(cam.FoVy * 0.5)


            Rm = cam.world_view_transform[:3, :3].to(device)   # (3,3)
            t  = cam.world_view_transform[:3, 3].to(device)
            z_cam = (verts @ Rm.T + t)[:, 2].abs()
            pixel_size = 2.0 * z_cam * tanfovy / H
            gauss_kwargs["scales"] = (pixel_size * 0.6).unsqueeze(1).repeat(1,3).float()

            rast_set = GaussianRasterizationSettings(
                image_height   = H,
                image_width    = W,
                tanfovx        = tanfovx,
                tanfovy        = tanfovy,
                bg             = torch.tensor([1.0, 1.0, 1.0], device=device),
                scale_modifier = 2.0,
                viewmatrix     = cam.world_view_transform.to(device),
                projmatrix     = cam.full_proj_transform.to(device),
                sh_degree      = 0,
                campos         = cam.camera_center.to(device),
                prefiltered    = False,
                debug          = False,
            )
            rasterizer = GaussianRasterizer(raster_settings=rast_set)
            img, _, _  = rasterizer(**gauss_kwargs)

            rgb = (img.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            vid512 = rgb[:, :, 0].astype(np.int32) * 256 + rgb[:, :, 1].astype(np.int32)
            vid512[vid512 >= V] = -1
            pix2vert_all.append(torch.from_numpy(vid512.astype(np.int16)))


            verts_cam = (verts @ Rm.T + t)                  # (V,3)
            xy_norm   = verts_cam[:, :2] / verts_cam[:, 2:3]
            u_raw = ((xy_norm[:,0] / tanfovx) + 1) * W/2   
            v_raw = ((xy_norm[:,1] / tanfovy) + 1) * H/2
 
            mask_in_fov = (u_raw >= 0) & (u_raw < W) & (v_raw >= 0) & (v_raw < H)  
            u = u_raw.clamp(0, W-1)    
            v = v_raw.clamp(0, H-1)


            for R in extra_res:
                k = H // R                                  # 512→64 => k=8

                pixel_size_R = 2.0 * z_cam * tanfovy / R
                gauss_kwargs["scales"] = (pixel_size_R * 0.6).unsqueeze(1).repeat(1,3).float()
                rast_set_R = GaussianRasterizationSettings(
                    image_height   = R,  image_width = R,
                    tanfovx        = tanfovx, tanfovy = tanfovy,
                    bg             = torch.tensor([1.0, 1.0, 1.0], device=device),
                    scale_modifier = 0.1,
                    viewmatrix     = cam.world_view_transform.to(device),
                    projmatrix     = cam.full_proj_transform.to(device),
                    sh_degree      = 0,
                    campos         = cam.camera_center.to(device),
                    prefiltered    = False,
                    debug          = False,
                )
                rasterizer_R = GaussianRasterizer(raster_settings=rast_set_R)
                imgR, _, _   = rasterizer_R(**gauss_kwargs)
                rgbR = (imgR.permute(1,2,0) * 255).round().byte().cpu().numpy()
                vidR = rgbR[:,:,0].astype(np.int32)*256 + rgbR[:,:,1].astype(np.int32)
                vidR[vidR >= V] = -1
                vid_base = torch.from_numpy(vidR.astype(np.int16))    

                vid_cur  = torch.full((R, R), -1, dtype=torch.int16,  device="cpu")

                S = seeds_dict[R]
                if len(S):
                    mask_seed_all  = torch.isin(ids, torch.tensor(list(S), device=device))
                    mask_seed = mask_seed_all & mask_in_fov
                    if mask_seed.any():
                        us = u[mask_seed]; vs = v[mask_seed]; zs = verts_cam[mask_seed, 2]
                        ii = (vs // k).long().clamp(0, R-1)
                        jj = (us // k).long().clamp(0, R-1)
                        ids_seed = ids[mask_seed].to(torch.int16).cpu()
                        for idx in range(ids_seed.shape[0]):
                            ii_ = ii[idx].item(); jj_ = jj[idx].item()

                            if ii_ < 0 or ii_ >= R or jj_ < 0 or jj_ >= R:
                                continue
                            if vid_base[ii_, jj_] >= 0:          
                                vid_cur[ii_, jj_] = ids_seed[idx].item()

                mask_fill = (vid_cur == -1)              
                vid_cur[mask_fill] = vid_base[mask_fill] 

                # d) 写入列表；更新种子集合
                pix2shell_extra[R].append(vid_cur.clone())
                seeds_dict[R].update(vid_cur[vid_cur >= 0].tolist())   

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(pix2vert_all, save_path)
        torch.save(pix2shell_extra, multi_path)

        self.pix2vert_all    = pix2vert_all
        self.pix2shell_extra = pix2shell_extra
        print(f"[Pix2Vert] ✓ saved 512 + unified low-res maps")
   

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        self.masks_org={}

        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            human_weights = torch.zeros_like(self.gaussian._opacity)
            human_weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )

                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

                mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                    0
                ].to(get_device())
                if self.cfg.seg_prompt2:
                    mask2 = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt2)[
                        0
                    ].to(get_device())
                    mask = mask.int() | mask2.int()

                human_mask = self.text_segmentor(self.origin_frames[id], self.cfg.human_prompt)[
                    0
                ].to(get_device())

                self.masks_org[id] = mask
                mask_to_save = (
                        mask[0]
                        .cpu()
                        .detach()[..., None]
                        .repeat(1, 1, 3)
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = self.origin_frames[id].detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                        masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)
                self.gaussian.apply_weights(cur_cam, human_weights, human_weights_cnt, human_mask)

            weights /= weights_cnt + 1e-7
            selected_mask = weights > self.cfg.mask_thres
            self.org_selected_mask = selected_mask[:, 0]
            selected_mask = self.org_selected_mask
            human_weights /= human_weights_cnt + 1e-7
            self.human_selected_mask = human_weights > self.cfg.mask_thres
            self.human_selected_mask = self.human_selected_mask[:, 0]

        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)
            
        self.gaussian.remove_grad_mask()
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)


    def on_validation_epoch_end(self):
        pass

    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        self.gaussian.localize = local
        for id, cam in enumerate(batch["camera"]):

            render_pkg = render(cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                renderbackground,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        self.gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(range(self.trainer.datamodule.train_dataset.total_view_num)):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + self.viewspace_point_list[idx].grad
                    )
                # Keep track of max radii in image-space for pruning
                # vis_filter = self.visibility_filter
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )

                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        (self.true_global_step >= self.cfg.densify_from_iter
                        and self.true_global_step == 0)
                        or (self.true_global_step % 20 == 0 and self.true_global_step != self.trainer.max_steps-1 )
                        # or self.true_global_step == self.trainer.max_steps-1
                ):  # 500 100
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        self.cfg.max_clothe,
                        5,
                        obj_path=Path(self.cfg.obj_source)/"rigid.obj",
                    )
                if self.true_global_step == self.trainer.max_steps-1 or (self.true_global_step+1) % self.cfg.densification_interval == 0 or self.true_global_step == 0:
                    weights = torch.zeros_like(self.gaussian._opacity)
                    weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
                    for index, id in enumerate(self.view_list):
                        # if renderbackground is None:
                        base_mask = self.all_view_mask[id]
                        renderbackground = self.background_tensor


                        semantic_map = render(
                            self.trainer.datamodule.train_dataset.scene.cameras[id],
                            self.gaussian,
                            self.pipe,
                            renderbackground,
                            override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
                        )["render"]


                        base_zero_mask = torch.norm(base_mask, dim=0) ==0

                        semantic_nonzero_mask = torch.norm(semantic_map, dim=0) > 0
                        prune_mask = (base_zero_mask & semantic_nonzero_mask).float().unsqueeze(0)



                        self.gaussian.apply_weights_all(self.trainer.datamodule.train_dataset.scene.cameras[id], weights, weights_cnt, prune_mask)

                    weights /= weights_cnt + 1e-7
                    selected_mask_2 = weights > 0.0


                    selected_mask_2 = selected_mask_2[:, 0]

                    count = int(selected_mask_2.sum().item())
                    self.gaussian.prune_out_mask(selected_mask_2)
                    

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-val/{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-val/{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=5,
            name="test",
            step=self.true_global_step,
        )
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

        save_path = self.get_save_path(f"last.ply")
        print("save_path", save_path)
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        self.view_num = len(self.view_list)
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        self.gaussian.load_ply(self.cfg.gs_source)

        # === Method‑1: build / load SMPL‑SDF once ===========================
        smpl_obj = Path(self.cfg.obj_source)/"rigid.obj"           # 
        sdf_cache = Path(self.cfg.obj_source)/"sdf_cache.pkl"
        with torch.no_grad():
            self.smpl_sdf= build_smpl_sdf(
                smpl_obj, sdf_cache,
                self.gaussian.get_xyz.detach()
            )
        self.gaussian.set_SDF(self.smpl_sdf )


        verts, faces_idx, _ = load_obj(smpl_obj, load_textures=False)
        self.smpl_verts   = verts
        self.smpl_normals = Meshes(verts=[verts], faces=[faces_idx.verts_idx]).verts_normals_list()[0]
        # faces_idx.verts_idx: LongTensor of shape (F,3)
        self.smpl_faces = faces_idx.verts_idx.long()     # (F,3)
        

        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
    
    
    
    def edit_all_view(self, original_render_name, cache_name, update_camera=False, global_step=0, edit_2=False):
        # if self.true_global_step >= self.cfg.camera_update_per_step * 2:
        #     self.guidance.use_normal_unet()
        self.all_view_mask = {}
        self.mask_edi = {}
        self.edited_cams = []
        edi_mask_cache = {}
        m_count = 0
        self.cache_mask= self.gaussian.mask.clone()
        # if update_camera:
        if edit_2 and not self.mask_attn:
            self.cfg.top_view_num = 20
            self.trainer.datamodule.train_dataset.update_cameras(random_seed = 1)
            self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
            sorted_train_view_list = sorted(self.view_list)
            selected_views = torch.linspace(
                0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
            )
            self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]
        if len(self.cfg.seg_prompt) > 0 and edit_2==True:
            self.update_mask()
        self.edit_frames = {}
        similarity_scores = {}

        delta_feats = {}

        cache_dir = os.path.join(self.cache_dir, cache_name)
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)
        mask_shrink = {}
        cameras = []
        images = []
        original_frames = []
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.true_global_step//self.cfg.camera_update_per_step)]
        with torch.no_grad():
                for id in self.view_list:
                    cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
                sorted_cam_idx = self.sort_the_cameras_idx(cameras)
                view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]
                cams_sorted = [cameras[idx] for idx in sorted_cam_idx]  

                ############################
                from pathlib import Path
                save_dir = Path("debug_masks")
                save_dir.mkdir(exist_ok=True)   
                    
                for id in view_sorted:
                    cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                    original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out_pkg = self(cur_batch)
                    out = out_pkg["comp_rgb"]
                    # if self.cfg.use_masked_image:
                    #     out = out * out_pkg["masks"].unsqueeze(-1)
                    images.append(out)
                    assert os.path.exists(original_image_path)
                    cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                    self.origin_frames[id] = torch.tensor(
                        cached_image / 255, device="cuda", dtype=torch.float32
                    )[None]
                    original_frames.append(self.origin_frames[id])
                    #----------------------------------#
                  
                images = torch.cat(images, dim=0)
                original_frames = torch.cat(original_frames, dim=0)

                # stats = check_consistency(self.pix2shell_extra)

                if edit_2 == True :
                    if self.mask_attn:
                       
                        cloth_mask = [ self.cloth_mask[v] for v in view_sorted ]
                        edited_images = self.guidance(
                        original_frames,
                        images,                                                                                                                                     
                        self.prompt_processor(),
                        cams = cams_sorted, 
                        cloth_mask=cloth_mask                     
                    )
                    else:
                        if global_step == 605:
                            cloth_mask = None
                            edited_images = self.guidance(
                                images,
                                original_frames,
                                self.prompt_processor(),
                                cams = cams_sorted,
                                cloth_mask=cloth_mask
                            )
                        else:
                            cloth_mask = None
                            edited_images = self.guidance(
                                images,
                                images,                                
                                self.prompt_processor(),
                                cams = cams_sorted,
                                cloth_mask=cloth_mask
                            )

                else:
                    edited_images = self.guidance(
                        images,
                        original_frames,
                        self.prompt_processor(),
                        cams = cams_sorted,                      
                    )

                # 

                for view_index_tmp, id in enumerate(self.view_list):
                    self.edit_frames[view_sorted[view_index_tmp]] = edited_images['edit_images'][view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C


                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[view_sorted[view_index_tmp]]
                    
                    mask_editor_human = self.text_segmentor(self.edit_frames[view_sorted[view_index_tmp]], self.cfg.human_prompt)[
                        0
                    ].to(get_device())

                   
                    image_editor = self.edit_frames[view_sorted[view_index_tmp]].to(get_device())  # (1, H, W, 3)
                    mask_bin = (mask_editor_human > 0).float().unsqueeze(-1)
                    image_editor = image_editor * mask_bin 
                    mask_editor = self.text_segmentor(image_editor, self.cfg.seg_prompt_editor)[
                        0
                    ].to(get_device())

                    mask = self.masks_org[view_sorted[view_index_tmp]].int() | mask_editor.int()
                    
                    mask = mask.float()

                    self.mask_edi[view_sorted[view_index_tmp]]=mask
                    edi_mask_cache[view_sorted[view_index_tmp]]=mask_editor

                    mask_shrink_tmp = ((mask_editor_human == 0) & (mask == 1)).float()

                    mask_shrink[view_sorted[view_index_tmp]] = mask_shrink_tmp

                if global_step ==  0 or edit_2==True:
                    view_ids    = list(self.mask_edi.keys())
                    text_scores = []
                    tmp_dir = tempfile.mkdtemp(prefix="imgreward_") 
                    for v in view_ids:
                        # self.edit_frames[v] : torch.Size([1, 512, 512, 3])，
                        arr = self.edit_frames[v][0].detach().cpu().numpy()        # (H,W,3)
                        if arr.shape[-1] != 3:
                            arr = np.transpose(arr, (1, 2, 0))
                        arr = np.clip(arr, 0.0, 1.0)

                        pil_img = Image.fromarray((arr * 255).round().astype(np.uint8))
                        img_path = os.path.join(tmp_dir, f"view_{v}.png")
                        pil_img.save(img_path)

                        score_v = float(self.image_reward_model.score(self.cfg.seg_text_computer, img_path))
                        text_scores.append(score_v)

                    text_scores = torch.tensor(text_scores, device=torch.device("cpu"))        # (N,)

                    topk        = min(self.cfg.top_view_num, len(view_ids))       
                    _, idx_sort = torch.topk(text_scores, k=topk, largest=True, sorted=True)
                    selected_idx   = idx_sort.tolist()                              

                    self.top_k_view_ids = [view_ids[i] for i in selected_idx[:topk]]       

                    self.mask_edi    = {k: v for k, v in self.mask_edi.items()    if k in self.top_k_view_ids}
                    self.edit_frames = {k: v for k, v in self.edit_frames.items() if k in self.top_k_view_ids}


                if self.cfg.max_clothe == True and global_step == 0:
                    self.gaussian.seed_virtual_gaussians_from_smpl(
                        self.smpl_verts.to(self.device),
                        self.smpl_normals.to(self.device),
                        self.human_selected_mask,
                        obj_path=Path(self.cfg.obj_source)/"rigid.obj",  
                        offset=self.cfg.offset, sdf_thin=self.cfg.sdf_thin, sdf_thick=self.cfg.sdf_thick
                    )
                    # self.build_pix2shell_maps(overwrite=self.cfg.cache_overwrite, offset=0.07)
                    
                weights = torch.zeros_like(self.gaussian._opacity)
                weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
                for view_id in self.mask_edi:


                    mask_top = self.mask_edi[view_id] 
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[view_id]
                    self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask_top)

                weights /= weights_cnt + 1e-7

                selected_mask = weights > self.cfg.mask_thres
                selected_mask = selected_mask[:, 0]
                if self.cfg.max_clothe == True and global_step == 0:
                    N_old = self.org_selected_mask.shape[0]
                    N_new = selected_mask.shape[0]
                    pad = torch.zeros(
                        N_new - N_old,
                        dtype=torch.bool,
                        device=self.org_selected_mask.device
                    )
                    org_padded = torch.cat([self.org_selected_mask, pad], dim=0) 
                    all_selected_mask = org_padded | selected_mask
                    num_selected = int(all_selected_mask.sum().item())

                    new_gauss = torch.zeros(
                        N_new, dtype=torch.bool, device=self.gaussian._xyz.device
                    )
                    new_gauss[N_old:N_new] = True
                    drop_mask = new_gauss & (~all_selected_mask)
                    if drop_mask.any():
                        self.gaussian.prune_points(drop_mask)
                        all_selected_mask  = all_selected_mask[~drop_mask]
                        num_selected = int(all_selected_mask.sum().item())
                else:
                    all_selected_mask  = selected_mask
        self.gaussian.remove_grad_mask()
        self.gaussian.set_mask(all_selected_mask)
        self.gaussian.apply_grad_mask(all_selected_mask)


        self.trainer.datamodule.train_dataset.update_cameras_1(self.top_k_view_ids)
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        sorted_train_view_list = sorted(self.view_list)
        selected_views = torch.linspace(
            0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
        )
        self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]

        for index, id in enumerate(self.top_k_view_ids):
                # if renderbackground is None:
            renderbackground = self.background_tensor
            semantic_map_rel = render(
                self.trainer.datamodule.train_dataset.scene.cameras[id],
                self.gaussian,
                self.pipe,
                renderbackground,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            self.all_view_mask[id]=semantic_map_rel           
            mask_rel = (self.all_view_mask[id].sum(dim=0) == 0)

        finalselect = int(all_selected_mask.sum().item())
        self.cloth_mask=self.all_view_mask

        if global_step > 0 :
            N = self.cache_mask.shape[0]

            all_selected_mask[:N] |= self.cache_mask
            self.gaussian.remove_grad_mask()
            self.gaussian.set_mask(all_selected_mask)
            self.gaussian.apply_grad_mask(all_selected_mask)

        if self.cfg.max_clothe == False:    
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            for index, id in enumerate(self.top_k_view_ids):
                
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask_shrink[id])

            weights /= weights_cnt + 1e-7
            if  global_step == 0:
                selected_mask = weights > 0.3
            else:
                selected_mask = weights > 0.3
            selected_mask = selected_mask[:, 0]
            drop_shrink_point = selected_mask & all_selected_mask
            if drop_shrink_point.any():
                self.gaussian.prune_points(drop_shrink_point)
                all_selected_mask  = all_selected_mask[~drop_shrink_point]
                self.gaussian.remove_grad_mask()
                self.gaussian.set_mask(all_selected_mask)
                self.gaussian.apply_grad_mask(all_selected_mask)

    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), 0, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) for cam in cams]
        
        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        self.build_pix2vert_maps(overwrite=False)

        if len(self.cfg.seg_prompt) > 0:
            self.update_mask()


        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0 or self.cfg.loss.use_sds:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            

    def training_step(self, batch, batch_idx):
        self.jump_view=False
        edit_2 = False
        self.mask_attn = False
        if  self.cfg.guidance_type == 'THGE-guidance'  and (self.true_global_step % (self.cfg.camera_update_per_step + 1) == 0
                                                           or self.true_global_step  == 300
                                                           or self.true_global_step  == 600
                                                           or self.true_global_step==self.cfg.camera_update_per_step +self.cfg.edi_view+ 1):
            if self.true_global_step == self.cfg.camera_update_per_step + self.cfg.edi_view + 1 or self.true_global_step  == 600 or self.true_global_step  == 300:
                edit_2 = True

            self.edit_all_view(original_render_name='origin_render', cache_name="edited_views", update_camera=self.true_global_step >= self.cfg.camera_update_per_step, global_step=self.true_global_step, edit_2 = edit_2) 
            # self.edit_all_view(original_render_name='origin_render', cache_name="edited_views", update_camera=True, global_step=self.true_global_step)

        self.gaussian.update_learning_rate(self.true_global_step)
        batch_index = batch["index"]



        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if self.cfg.guidance_type == 'THGE-guidance': 
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames:
                    batch_index[img_index] = self.view_list[img_index]

        out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]
        mask = out["masks"].unsqueeze(-1)
        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                # if cur_index not in self.edit_frames:
                #     # cur_index = self.view_list[0]
                if (cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                )) and 'THGE' not in str(self.cfg.guidance_type) and not self.cfg.loss.use_sds:
                    print(self.cfg.guidance_type)
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )
                
                    self.edit_frames[cur_index] = result["edit_images"].detach().clone()

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)
            # no_mask = 1 - self.all_view_mask[batch_index[0]]
            mask_b = (self.all_view_mask[batch_index[0]] ).bool()          # → [3,512,512]
            mask_b = mask_b.permute(1, 2, 0).unsqueeze(0)  
            if self.cfg.use_masked_image:
                # print("use masked image")
                guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images[ mask_b], gt_images[mask_b]),
                # "loss_l1": torch.nn.functional.l1_loss(images * self.all_view_mask[batch_index[0]].unsqueeze(-1), gt_images * self.all_view_mask[batch_index[0]].unsqueeze(-1)),
                "loss_on_mask": torch.nn.functional.l1_loss(images[~mask_b], self.origin_frames[cur_index][~mask_b]),
                "loss_p": self.perceptual_loss(
                    (images*mask_b).permute(0, 3, 1, 2).contiguous(),
                    (gt_images*mask_b).permute(0, 3, 1, 2).contiguous(),
                ).sum(),
                }
            else:
                guidance_out = {
                    "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                    "loss_p": self.perceptual_loss(
                        images.permute(0, 3, 1, 2).contiguous(),
                        gt_images.permute(0, 3, 1, 2).contiguous(),
                    ).sum(),
                }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )
        # sds loss
        if self.cfg.loss.use_sds:
            prompt_utils = self.prompt_processor()
            self.guidance.cfg.use_sds = True
            guidance_out = self.guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                prompt_utils)  
            loss += guidance_out["loss_sds"] * self.cfg.loss.lambda_sds 

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
    
        return {"loss": loss}