from typing import Type
import torch
import os

from pathlib import Path
from PIL import Image
import torch
import yaml
import math

from gaussiansplatting.utils.graphics_utils import get_fundamental_matrix_with_H
import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from torchvision.io import write_video
from kornia.geometry.transform import remap

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False
def register_cloth_mask(unet, cloth_list):

    for _, m in unet.named_modules():
        if isinstance_str(m, "BasicTransformerBlock"):
            m.attn1.cloth_mask = cloth_list
# --------------------------------------------------


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def resize_bool_tensor(bool_tensor, size):
    """
    Resizes a boolean tensor to a new size using nearest neighbor interpolation.
    """
    # Convert boolean tensor to float
    H_new, W_new = size
    tensor_float = bool_tensor.float()

    # Resize using nearest interpolation
    resized_float = torch.nn.functional.interpolate(tensor_float, size=(H_new, W_new), mode='nearest')

    # Convert back to boolean
    resized_bool = resized_float > 0.5
    return resized_bool

def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def register_vertex_constraints(diffusion_model, vertex_constraints):
    """
    vertex_constraints: dict from sequence_length to tensor (N, L, L) of bool masks
    """
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "vertex_constraints", vertex_constraints)

def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64):
    n_frames = 1
    sequence_length = current_W * current_H
    fundamental_matrix_1 = []
    
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)

    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    # epipolar_line: n_frames X seq_len,  3
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1.cuda()).view(-1, 3)
    
    distance1 = point_to_line_dist(heto_cam1, line1)

    
    idx1_epipolar = distance1 > 1 # sequence_length x sequence_lengths

    return idx1_epipolar

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_epipolar_constrains(diffusion_model, epipolar_constrains):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "epipolar_constrains", epipolar_constrains)

def register_cams(diffusion_model, cams, pivot_this_batch, key_cams):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "cams", cams)
            setattr(module, "pivot_this_batch", pivot_this_batch)
            setattr(module, "key_cams", key_cams)

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
            if hasattr(module, "attn1"):
                setattr(module.attn1, "pivotal_pass", is_pivotal)
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_t(diffusion_model, t):

    for _, module in diffusion_model.named_modules():
    # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "t", t)


def register_normal_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            # assert encoder_hidden_states is None 
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.head_to_batch_dim(q)
            key = self.head_to_batch_dim(k)
            value = self.head_to_batch_dim(v)

            attention_probs = self.get_attention_scores(query, key)
            hidden_states = torch.bmm(attention_probs, value)
            out = self.batch_to_head_dim(hidden_states)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.normal_attn = sa_forward(module.attn1)
            module.use_normal_attn = True

def register_normal_attn_flag(diffusion_model, use_normal_attn):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "use_normal_attn", use_normal_attn)

def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        pix2shell_path = "/root/edit_cache/-root-autodl-tmp-dataset-person-point_cloud-iteration_7000-point_cloud.ply/pix2vert.pt"
        self.pix2shell_all = torch.load(pix2shell_path, map_location="cpu")

        pix2shell_extra_path = "/root/edit_cache/-root-autodl-tmp-dataset-person-point_cloud-iteration_7000-point_cloud.ply/pix2shell_multi.pt"
        self.pix2shell_extra = torch.load(pix2shell_extra_path, map_location="cpu")

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            assert encoder_hidden_states is None  
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            device = x.device

            feat_H = int(math.sqrt(sequence_length))
            feat_W = sequence_length // feat_H
            assert feat_H * feat_W == sequence_length

            frame_ids = getattr(self, "current_frames", None) or getattr(self, "current_pivots", None)
            if frame_ids is None:
                raise RuntimeError("error")

            if feat_H in self.pix2shell_extra:                 # 64 / 32 / 16 / 8
                maps_ds_long = torch.stack(
                    [self.pix2shell_extra[feat_H][fid] for fid in frame_ids],
                    dim=0
                ).unsqueeze(1).long().to(device)               # (N,1,H,W)
            else:                                            
                maps_512 = [self.pix2shell_all[fid] for fid in frame_ids]   # list (512,512)
                maps_t   = torch.stack(maps_512, dim=0).unsqueeze(1).float()# (N,1,512,512)
                maps_ds  = torch.nn.functional.interpolate(
                    maps_t, size=(feat_H, feat_W), mode="nearest"
                )
                maps_ds_long = maps_ds.long().to(device)      
            feat_maps = maps_ds_long.squeeze(1)                              # (N,feat_H,feat_W)

            smplx_map_cur_all = feat_maps.view(-1).long()                    # (N*L,)

            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            q_text, q_image, q_uncond = (
                q[:n_frames],
                q[n_frames:2 * n_frames],
                q[2 * n_frames:3 * n_frames],
            )
            k_text, k_image, k_uncond = (
                k[:n_frames],
                k[n_frames:2 * n_frames],
                k[2 * n_frames:3 * n_frames],
            )
            v_text, v_image, v_uncond = (
                v[:n_frames],
                v[n_frames:2 * n_frames],
                v[2 * n_frames:3 * n_frames],
            )

            def THGE_global(qb, kb, vb, mask_cols=None):
                kb_all = kb.reshape(1, -1, dim).repeat(n_frames, 1, 1)  # (n_frames, n_frames*L, D)
                vb_all = vb.reshape(1, -1, dim).repeat(n_frames, 1, 1)
                qh = self.head_to_batch_dim(qb)      # (n_frames*h, L, D/h)
                kh = self.head_to_batch_dim(kb_all)  # (n_frames*h, n_frames*L, D/h)
                vh = self.head_to_batch_dim(vb_all)  # (n_frames*h, n_frames*L, D/h)
                out_list = []
                qj = qh.view(n_frames, h, sequence_length, dim // h)             # (n_frames, L, D/h)
                kj = kh.view(n_frames, h, n_frames * sequence_length, dim // h)  # (n_frames, n_frames*L, D/h)
                vj = vh.view(n_frames, h, n_frames * sequence_length, dim // h)  # (n_frames, n_frames*L, D/h)
                neg_inf = torch.tensor(-1e4, dtype=qj.dtype, device=device)
                for j in range(h):
                    sim = torch.bmm(qj[:, j], kj[:, j].transpose(-1, -2)) * self.scale  # (n_frames, L, n_frames*L)
                    if mask_cols is not None:
                        sim = sim.masked_fill(mask_cols, neg_inf)
                    out_list.append(torch.bmm(sim.softmax(dim=-1), vj[:, j]))           # (n_frames, L, D/h)
                out = torch.cat(out_list, dim=0)  # (h*n_frames, L, D/h)
                out = out.view(h, n_frames, sequence_length, dim // h) \
                         .permute(1, 0, 2, 3) \
                         .reshape(h * n_frames, sequence_length, -1)
                out = self.batch_to_head_dim(out)
                return out

            msk=None
            out_text_global   = THGE_global(q_text,   k_text,   v_text, msk)     # (n_frames, L, D)
            out_image_global  = THGE_global(q_image,  k_image,  v_image, msk)    # (n_frames, L, D)
            out_uncond_global = THGE_global(q_uncond, k_uncond, v_uncond, msk)   # (n_frames, L, D)

            def branch_attn(qb, kb, vb):

                tokens_q_orig = qb.reshape(-1, dim)        # (N*L, D)
                tokens_k_glb = kb.reshape(-1, dim)         # (N*L, D)
                tokens_v_glb = vb.reshape(-1, dim)         # (N*L, D)

                tokens_out = torch.zeros_like(tokens_q_orig)  # (N*L, D)

                vertex2pix_orig = {}
                flat_list = smplx_map_cur_all.cpu().tolist()  # length = N*L
                for idx_token, v_id in enumerate(flat_list):
                    if v_id >= 0:
                        vertex2pix_orig.setdefault(v_id, []).append(idx_token)

                feat_flat = feat_maps.view(n_frames, -1)  # (n_frames, L)
                from collections import Counter
                counter = Counter()
                for i in range(n_frames):
                    ids_i = feat_flat[i]
                    valid_ids = torch.unique(ids_i[ids_i >= 0]).tolist()
                    counter.update(valid_ids)

                K = max(1, n_frames - 1)
                common_vertices = {vid for vid, cnt in counter.items() if cnt >= K}

                if sequence_length == 64:
                    r = 1  # 窗口半径
                elif sequence_length == 256:
                    r = 2
                elif sequence_length == 1024:
                    r = 3
                elif sequence_length == 4096:
                    r = 4
                expanded_neighbors = {}
                for v_id in common_vertices:
                    neighbors = set()
                    for i in range(n_frames):
                        coords = (feat_maps[i] == v_id).nonzero(as_tuple=False)  # (n_pts, 2)
                        if coords.numel() == 0:
                            continue
                        center = coords.float().mean(dim=0).round().long()
                        uc, vc = int(center[0].item()), int(center[1].item())
                        u0, u1 = max(0, uc - r), min(feat_H - 1, uc + r)
                        v0, v1 = max(0, vc - r), min(feat_W - 1, vc + r)
                        window = feat_maps[i, u0:u1+1, v0:v1+1].reshape(-1)
                        cand = window[(window >= 0) & (window != v_id)].unique().tolist()
                        neighbors.update(cand)
                    expanded_neighbors[v_id] = neighbors

                vertex2pix = {}
                for v_id in common_vertices:
                    pix_indices = []
                    pix_indices.extend(vertex2pix_orig.get(v_id, []))
                    for nbr in expanded_neighbors[v_id]:
                        pix_indices.extend(vertex2pix_orig.get(nbr, []))
                    pix_indices = list(set(pix_indices))
                    vertex2pix[v_id] = pix_indices

                local_idx = sorted({idx for pix_list in vertex2pix.values() for idx in pix_list})

                if len(local_idx) > 0:
                    Vn = len(vertex2pix)                       
                    expanded_lists = list(vertex2pix.values()) 
                    d  = dim
                    hN = h * Vn                               

                    Q_groups = [tokens_q_orig[torch.tensor(lst, device=device)]
                                for lst in expanded_lists]                # list[(L_i,D)]
                    K_groups = [tokens_k_glb[torch.tensor(lst, device=device)]
                                for lst in expanded_lists]
                    V_groups = [tokens_v_glb[torch.tensor(lst, device=device)]
                                for lst in expanded_lists]

                    out_groups = []                                       
                    for i_vk in range(Vn):
                        Qi = Q_groups[i_vk]                               # (L_i,D)
                        Ki = K_groups[i_vk]
                        Vi = V_groups[i_vk]
                        if Qi.numel() == 0:                              
                            out_groups.append(Qi)                        
                            continue


                        Qi_h = Qi.view(-1, h, d // h).transpose(0, 1)     # (h,L_i,d/h)
                        Ki_h = Ki.view(-1, h, d // h).transpose(0, 1)
                        Vi_h = Vi.view(-1, h, d // h).transpose(0, 1)

                        heads_out = []
                        for j in range(h):
                            qj = Qi_h[j]                                  # (L_i,d/h)
                            kj = Ki_h[j]
                            vj = Vi_h[j]

                            sim = torch.matmul(qj, kj.T) * self.scale     # (L_i,L_i)
                            att = torch.softmax(sim, dim=-1)              
                            heads_out.append(torch.matmul(att, vj))       # (L_i,d/h)


                        out_groups.append(torch.cat(heads_out, dim=-1))


                    for pix_list, feats in zip(expanded_lists, out_groups):
                        if feats.numel() > 0:
                            tokens_out[pix_list] = feats

                all_indices = set(range(n_frames * sequence_length))
                rest_idx = torch.tensor(
                    sorted(all_indices.difference(local_idx)),
                    dtype=torch.long,
                    device=device,
                )  

                if rest_idx.numel() > 0:

                    masked_k = tokens_k_glb.clone()
                    masked_v = tokens_v_glb.clone()
                    masked_k[local_idx] = 0.0
                    masked_v[local_idx] = 0.0

                    col_mask = torch.zeros(
                        (n_frames, 1, n_frames * sequence_length),
                        dtype=torch.bool, device=device
                    )
                    col_mask[:, :, local_idx] = True       

                    out_rest_global = THGE_global(
                        qb.view(n_frames, sequence_length, dim),
                        masked_k.view(n_frames, sequence_length, dim),
                        masked_v.view(n_frames, sequence_length, dim),
                        mask_cols = col_mask               
                    )

                    out_rest_flat = out_rest_global.view(-1, dim)  # (n_frames*L, D)
                    tokens_out[rest_idx] = out_rest_flat[rest_idx]

                updated = tokens_out.view(n_frames, sequence_length, dim)
                updated_qh = self.head_to_batch_dim(updated)      # (h*n_frames, L, d/h)
                updated_final = self.batch_to_head_dim(updated_qh)  # (n_frames, L, D)
                return updated_final
                
            
            phase = getattr(self, "phase", 2)
            
            

            threshold = 0  
            if sequence_length > threshold:
                out_text_final   = branch_attn(q_text,   k_text,   v_text)
                out_image_final  = branch_attn(q_image,  k_image,  v_image)
                out_uncond_final = branch_attn(q_uncond, k_uncond, v_uncond)
                out_all = torch.cat([out_text_final, out_image_final, out_uncond_final], dim=0)  # (3*n_frames, L, D)
            else:
                out_all = torch.cat([out_text_global, out_image_global, out_uncond_global], dim=0)

            return to_out(out_all)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            module.attn1.current_pivots = None



def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers).cuda()
    key_cam_centers = torch.stack(key_cam_centers).cuda()
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance 
def make_THGE_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class THGEBlock(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # phase = getattr(self, "phase", 2)
            if not hasattr(self, "pix2shell_extra"):
                pix2shell_extra_path = (
                    "/root/edit_cache/"
                    "-root-autodl-tmp-dataset-person-point_cloud-iteration_7000-point_cloud.ply/"
                    "pix2shell_multi.pt"
                )
                self.pix2shell_extra = torch.load(pix2shell_extra_path, map_location="cpu")
            # ------------------------------------------
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
        
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            if not self.use_normal_attn:
                if self.pivotal_pass:
                    self.pivot_hidden_states = norm_hidden_states
                else:
                    batch_idxs = [0, 1, 2, 3]      
                    Kp = len(batch_idxs)            


                    idx1, idx2, idx3, idx4 = [], [], [], []         
                    vc_all = self.vertex_constraints[sequence_length]           # (N,4,L,L)  bool
                    overlap_per_pair = vc_all.any(dim=3).sum(dim=2)             # (N,4) int 
                    
                    ##### PATCH-1  (mark same-view) #####
                    device = hidden_states.device
                    cam_ids_batch = torch.tensor([c.colmap_id for c in self.cams], device=device)   # (n_frames,)
                    cam_ids_key   = torch.tensor([kc.colmap_id for kc in self.key_cams], device=device)  # (4,)

                    key_id2idx = {int(id.item()): i for i, id in enumerate(cam_ids_key)}

                    pairs_same = [
                        (j, key_id2idx[int(cid.item())])
                        for j, cid in enumerate(cam_ids_batch)
                        if int(cid.item()) in key_id2idx
                    ]
                    ##### END PATCH-1 #####

                    overlap_val, closest_cam = torch.topk(                      # (N,K′), (N,K′)
                        overlap_per_pair, k=Kp, dim=1, largest=True, sorted=True
                    )

                    overlap_1 = overlap_val[:, 0].float()                 # 一定有
                    overlap_2 = overlap_val[:, 1].float()
                    overlap_3 = overlap_val[:, 2].float()             ##### >>> 新增
                    overlap_4 = overlap_val[:, 3].float()             ##### >>> 新增

                    closest_pivot_h  = self.pivot_hidden_states[1][closest_cam]            # (N,K′,L,D)
                    sim = torch.einsum(
                            'bld,bcsd->bcls',
                            norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True),
                            closest_pivot_h       / closest_pivot_h.norm(dim=-1, keepdim=True)
                        )                                                                # (N,K′,L,L)


                    vc_all      = self.vertex_constraints[sequence_length]                 # (N,4,L,L)

                    gather_idx  = closest_cam.unsqueeze(-1).unsqueeze(-1)                  # (N,K′,1,1)
                    gather_idx  = gather_idx.expand(-1, -1, sequence_length, sequence_length)
                    mask_gather = vc_all.gather(dim=1, index=gather_idx)                   # (N,K′,L,L)


                    if len(batch_idxs) == 4:                                        
                        sim1, sim2, sim3, sim4 = sim.chunk(4, dim=1)            # (N,1,L,L)
                        sim1, sim2 = sim1.view(-1, sequence_length), sim2.view(-1, sequence_length)
                        sim3, sim4 = sim3.view(-1, sequence_length), sim4.view(-1, sequence_length)
                        sim1_raw, sim2_raw = sim1.clone(), sim2.clone()
                        sim3_raw, sim4_raw = sim3.clone(), sim4.clone()

                        vc1, vc2, vc3, vc4 = mask_gather[:, 0], mask_gather[:, 1], mask_gather[:, 2], mask_gather[:, 3]                    # (N,L,L)
                        vc1 = vc1.reshape(-1, sequence_length)                             # (N*L, L)
                        vc2 = vc2.reshape(-1, sequence_length)
                        vc3 = vc3.reshape(-1, sequence_length)
                        vc4 = vc4.reshape(-1, sequence_length)

                        sim1 = sim1.masked_fill(~vc1, -1e4)
                        sim2 = sim2.masked_fill(~vc2, -1e4)
                        sim3 = sim3.masked_fill(~vc3, -1e4)
                        sim4 = sim4.masked_fill(~vc4, -1e4)


                        empty1 = vc1.sum(dim=-1) == 0
                        empty2 = vc2.sum(dim=-1) == 0
                        empty3 = vc3.sum(dim=-1) == 0
                        empty4 = vc4.sum(dim=-1) == 0
                        if empty1.any():
                            sim1[empty1] = sim1_raw[empty1]
                        if empty2.any():
                            sim2[empty2] = sim2_raw[empty2]
                        if empty3.any(): sim3[empty3] = sim3_raw[empty3]
                        if empty4.any(): sim4[empty4] = sim4_raw[empty4]

                        sim1_max, sim2_max = sim1.max(dim=-1), sim2.max(dim=-1)
                        sim3_max, sim4_max = sim3.max(dim=-1), sim4.max(dim=-1)
                        idx1.append(sim1_max[1])
                        idx2.append(sim2_max[1])
                        idx3.append(sim3_max[1])      
                        idx4.append(sim4_max[1])     

                        same_mask1 = (~empty1).view(n_frames, -1)        # (N,L)
                        same_mask2 = (~empty2).view(n_frames, -1)        # (N,L)
                        same_mask3 = (~empty3).view(n_frames, -1)          
                        same_mask4 = (~empty4).view(n_frames, -1)          

                    else:                                                                  
                        sim        = sim.squeeze(1).view(-1, sequence_length)              # (N*L, L)
                        sim_raw    = sim.clone()
                        vc_single  = mask_gather.squeeze(1).reshape(-1, sequence_length)   # (N*L, L)

                        sim = sim.masked_fill(~vc_single, -1e4)

                        # —— fallback —— #
                        empty = vc_single.sum(dim=-1) == 0
                        if empty.any():
                            sim[empty] = sim_raw[empty]

                        sim_max = sim.max(dim=-1)
                        idx1.append(sim_max[1])
                        same_mask_single = (~empty).view(n_frames, -1)   # (N,L)

                    idx1 = torch.stack(idx1 * 3, dim=0).squeeze(1)          # (3, N*L)
                    if len(batch_idxs) == 4:
                        idx2 = torch.stack(idx2 * 3, dim=0).squeeze(1)      # (3, N*L)
                        idx3 = torch.stack(idx3 * 3, dim=0).squeeze(1)    
                        idx4 = torch.stack(idx4 * 3, dim=0).squeeze(1)   

                            
            
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.use_normal_attn:
                # print("use normal attn")
                self.attn_output = self.attn1.normal_attn(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )         
            else:
                # print("use extend attn")
                if self.pivotal_pass:
                    # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                    self.attn_output = self.attn1(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            **cross_attention_kwargs,
                        )
                    # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                    self.kf_attn_output = self.attn_output

                else:
                    batch_kf_size, _, _ = self.kf_attn_output.shape
                    self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,
                                    closest_cam]

            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.use_normal_attn:
                if not self.pivotal_pass:

                    W = int(math.sqrt(sequence_length))          
                    neigh_w = 0.25

                    if len(batch_idxs) == 4:
                        attn_1, attn_2, attn_3, attn_4 = (
                            self.attn_output[:, :, 0],
                            self.attn_output[:, :, 1],
                            self.attn_output[:, :, 2],
                            self.attn_output[:, :, 3],
                        )
                        idx1 = idx1.view(3, n_frames, sequence_length)
                        idx2 = idx2.view(3, n_frames, sequence_length)
                        idx3 = idx3.view(3, n_frames, sequence_length)
                        idx4 = idx4.view(3, n_frames, sequence_length)

                        def fuse_with_neighbors(attn_core, idx_core, closest_core, same_mask):

                            coord_map = torch.arange(sequence_length, device=attn_core.device)
                            row_map   = (coord_map // W).view(1, -1)
                            col_map   = (coord_map %  W).view(1, -1)

                            N = idx_core.shape[1]      
                            row_map_exp = row_map.expand(N, -1)
                            col_map_exp = col_map.expand(N, -1)

                            row = row_map_exp.gather(1, idx_core[0])
                            col = col_map_exp.gather(1, idx_core[0])

                            idx_l = row * W + (col - 1)                
                            idx_r = row * W + (col + 1)               

                            valid_l = col > 0
                            valid_r = col < (W - 1)

                            vid_maps = []
                            for n in range(n_frames):
                                k_idx = int(closest_core[n])
                                fid   = self.key_cams[k_idx].colmap_id - 1
                                vid_maps.append(
                                    self.pix2shell_extra[W][fid].to(attn_core.device).view(-1)
                                )
                            vids = torch.stack(vid_maps, dim=0)       # (N,L)

                            vid_c = vids.gather(1, idx_core[0])
                            vid_l = vids.gather(1, torch.where(valid_l, idx_l, idx_core[0]))
                            vid_r = vids.gather(1, torch.where(valid_r, idx_r, idx_core[0]))

                            valid_l &= (vid_l >= 0) & (vid_l != vid_c) & same_mask
                            valid_r &= (vid_r >= 0) & (vid_r != vid_c) & same_mask

                            idx_l = torch.where(valid_l, idx_l, idx_core[0])
                            idx_r = torch.where(valid_r, idx_r, idx_core[0])

                            center = attn_core.gather(
                                2, idx_core.unsqueeze(-1).repeat(1, 1, 1, dim)
                            )
                            left = attn_core.gather(
                                2, idx_l.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, dim)
                            )
                            right = attn_core.gather(
                                2, idx_r.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, dim)
                            )

                            w_center_base = 1.0 - 2.0 * neigh_w       
                            valid_cnt     = valid_l.float() + valid_r.float()   


                            share = torch.where(
                                valid_cnt > 0,
                                (1.0 - w_center_base) / valid_cnt,      # =0.8 / (#valid)
                                torch.zeros_like(valid_cnt)
                            )

                            w_left  = share * valid_l.float()          
                            w_right = share * valid_r.float()

                            w_center = torch.where(
                                valid_cnt > 0,
                                torch.full_like(valid_cnt, w_center_base),
                                torch.ones_like(valid_cnt)
                            )
                            # ---- END PATCH ----

                            fused = (
                                center * w_center.unsqueeze(0).unsqueeze(-1)
                                + left  * w_left.unsqueeze(0).unsqueeze(-1)
                                + right * w_right.unsqueeze(0).unsqueeze(-1)
                            )
                            return fused

                        fused_1 = fuse_with_neighbors(attn_1, idx1, closest_cam[:, 0], same_mask1)
                        fused_2 = fuse_with_neighbors(attn_2, idx2, closest_cam[:, 1], same_mask2)
                        fused_3 = fuse_with_neighbors(attn_3, idx3, closest_cam[:, 2], same_mask3)
                        fused_4 = fuse_with_neighbors(attn_4, idx4, closest_cam[:, 3], same_mask4)

                        total_overlap = overlap_1 + overlap_2 + overlap_3 + overlap_4 + 1e-8
                        w1 = (overlap_1 / total_overlap).clamp(0, 1)
                        w2 = (overlap_2 / total_overlap).clamp(0, 1)
                        w3 = (overlap_3 / total_overlap).clamp(0, 1)
                        w4 = (overlap_4 / total_overlap).clamp(0, 1)
                        w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                        w2 = w2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                        w3 = w3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                        w4 = w4.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                        # attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                        # attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                        attn_output = (
                            w1 * fused_1
                            + w2 * fused_2
                            + w3 * fused_3
                            + w4 * fused_4
                        ).reshape(batch_size, sequence_length, dim).half()
                        # attn_output =  1.3 * attn_output
                    else:
                        idx1_v  = idx1.view(3, n_frames, sequence_length)
                        attn_pivot = self.attn_output[:, :, 0]                  # (3,N,L,D)
                         
                        coord_map = torch.arange(sequence_length, device=attn_pivot.device)          # 0..L-1
                        row_map   = (coord_map // W).view(1, -1)   # (1,L)
                        col_map   = (coord_map %  W).view(1, -1)   # (1,L)

                        N = idx1_v.shape[1] if idx1_v.dim() == 3 else idx1_v.shape[1]  
                        row_map_exp = row_map.expand(N, -1)   
                        col_map_exp = col_map.expand(N, -1)  

                        row  = row_map_exp.gather(1, idx1_v[0])      
                        col  = col_map_exp.gather(1, idx1_v[0])     

                        idx_l = row * W + (col - 1)                
                        idx_r = row * W + (col + 1)               

                        valid_l = col > 0
                        valid_r = col < (W - 1)

                        vid_maps = []
                        for n in range(n_frames):
                            k_idx = int(closest_cam[n])
                            fid = self.key_cams[k_idx].colmap_id - 1
                            vid_maps.append(
                                self.pix2shell_extra[W][fid].to(attn_pivot.device).view(-1)
                            )
                        vids = torch.stack(vid_maps, dim=0)             # (N,L)

                        vid_c = vids.gather(1, idx1_v[0])
                        vid_l = vids.gather(1, torch.where(valid_l, idx_l, idx1_v[0]))
                        vid_r = vids.gather(1, torch.where(valid_r, idx_r, idx1_v[0]))

                        valid_l &= (vid_l >= 0) & (vid_l != vid_c) & same_mask_single
                        valid_r &= (vid_r >= 0) & (vid_r != vid_c) & same_mask_single

                        idx_l = torch.where(valid_l, idx_l, idx1_v[0])
                        idx_r = torch.where(valid_r, idx_r, idx1_v[0])

                        center = attn_pivot.gather(
                            2, idx1_v.unsqueeze(-1).repeat(1, 1, 1, dim)
                        )
                        left = attn_pivot.gather(
                            2, idx_l.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, dim)
                        )
                        right = attn_pivot.gather(
                            2, idx_r.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, dim)
                        )

                        w_left  = neigh_w * valid_l.float()
                        w_right = neigh_w * valid_r.float()
                        w_center = 1.0 - w_left - w_right

                        attn_output = (
                            center * w_center.unsqueeze(0).unsqueeze(-1)
                            + left  * w_left.unsqueeze(0).unsqueeze(-1)
                            + right * w_right.unsqueeze(0).unsqueeze(-1)
                        ).reshape(batch_size, sequence_length, dim).half()

                    ##### PATCH-2  (overwrite same-view frames) #####
                    if pairs_same:     
                        attn_out_4d = attn_output.view(3, n_frames, sequence_length, dim)
                        kf_full     = self.kf_attn_output.view(3, 4, sequence_length, dim)   # (3, 4, L, D)

                        for b_idx, k_idx in pairs_same:          # b_idx ∈ [0, n_frames-1]; k_idx ∈ [0,3]
                            attn_out_4d[:, b_idx] = kf_full[:, k_idx]

                        attn_output = attn_out_4d.reshape(batch_size, sequence_length, dim).half()
                   
                else:
                    attn_output = self.attn_output
            else:
                attn_output = self.attn_output
            
            
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            if not self.pivotal_pass: 
                if self.phase == 1:
                    hidden_states = 1.0 * attn_output + hidden_states
                elif self.phase == 2:
                    hidden_states = 1.0 * attn_output + hidden_states
                else:
                    hidden_states = attn_output + hidden_states
            else:
                hidden_states = attn_output + hidden_states
            hidden_states = hidden_states.to(self.norm2.weight.dtype)
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

    return THGEBlock