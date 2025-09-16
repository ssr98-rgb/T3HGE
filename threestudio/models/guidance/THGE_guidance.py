from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import math
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
from threestudio.utils.A import register_pivotal, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attention, register_extended_attention, make_THGE_block, isinstance_str, compute_epipolar_constrains, register_normal_attn_flag, register_vertex_constraints, register_cloth_mask


@threestudio.register("THGE-guidance")
class THGEGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "/root/autodl-tmp/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
        ip2p_name_or_path: str = "/root/autodl-tmp/models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        diffusion_steps: int = 20
        use_sds: bool = False
        camera_batch_size: int = 5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading InstructPix2Pix ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InstructPix2Pix!")
        self.step=0
        self.cloth_mask={}
        self.phase = 0


    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet_1(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        out = self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)
        selected = None
        if hasattr(self.unet, "global_key_view_indices"):
            selected = self.unet.global_key_view_indices  # Tensor[n]

        return selected
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
    self,
    latents: Float[Tensor, "..."],
    t: Float[Tensor, "..."],
    encoder_hidden_states: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def use_normal_unet(self):
        # print("use normal unet")
        register_normal_attention(self)
        register_normal_attn_flag(self.unet, True)
    def disable_normal_unet(self):
        # print("disable normal unet")
        register_normal_attn_flag(self.unet, False)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        #  # ① 保存 RNG 状态
        # _cpu_state = torch.random.get_rng_state()
        # _cuda_state = torch.cuda.get_rng_state(latents.device)

        # ② 使用局部 Generator
        # g = torch.Generator(device=latents.device).manual_seed(1)  # 固定 seed，确保一致性
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        B = latents.shape[0]  # 20

        with torch.no_grad():
            first_timestep = self.scheduler.timesteps[0]
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t) 
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)

            if self.step == 0:
                self.step=self.step+1
                for i in range(B):
                    latent = latents[i].unsqueeze(0)
                    
                    
                    cond_latent_pos = image_cond_latents[i].unsqueeze(0)              
                    cond_latent_neg = image_cond_latents[i + B].unsqueeze(0)         
                    cond_latent_uncond = image_cond_latents[i + 2 * B].unsqueeze(0)   
                    image_cond_latent = torch.cat([cond_latent_pos, cond_latent_neg, cond_latent_uncond], dim=0)


                    embed_pos = text_embeddings[i].unsqueeze(0)              # shape [1, 77, 768]
                    embed_neg = text_embeddings[i + B].unsqueeze(0)
                    embed_uncond = text_embeddings[i + 2 * B].unsqueeze(0)
                    embed_all = torch.cat([embed_pos, embed_neg, embed_uncond], dim=0)

                    
                    for t in self.scheduler.timesteps:

                        latent_model_input = torch.cat([latent] * 3)
                        latent_model_input = torch.cat([latent_model_input, image_cond_latent], dim=1)


                        noise_pred = self.forward_unet(
                            latent_model_input, t, encoder_hidden_states=embed_all
                        )

                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                            + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                        )

                        # get previous sample, continue loop
                        latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                        
                    latents[i] = latent.squeeze(0)
            else:
                self.phase = self.phase+1
                pix2shell_path = "/root/edit_cache/-root-autodl-tmp-dataset-person-point_cloud-iteration_7000-point_cloud.ply/pix2vert.pt"
                self.pix2shell_all = torch.load(pix2shell_path, map_location="cpu")
                pix2shell_extra_path = "/root/edit_cache/-root-autodl-tmp-dataset-person-point_cloud-iteration_7000-point_cloud.ply/pix2shell_multi.pt"
                self.pix2shell_extra = torch.load(pix2shell_extra_path, map_location="cpu")
                for _, module in self.unet.named_modules():
                    if isinstance_str(module, "BasicTransformerBlock"):
                        make_block_fn = make_THGE_block 
                        module.__class__ = make_block_fn(module.__class__)
                        # Something needed for older versions of diffusers
                        if not hasattr(module, "use_ada_layer_norm_zero"):
                            module.use_ada_layer_norm = False
                            module.use_ada_layer_norm_zero = False
                        module.phase = self.phase
                register_extended_attention(self)
                # register_normal_attention(self)
                # --------------------------------------------------
                def register_frames(unet, frame_ids: list[int]):
                    for _, m in unet.named_modules():
                        if isinstance_str(m, "BasicTransformerBlock"):
                            m.attn1.current_frames = frame_ids  
                # --------------------------------------------------
                # --------------------------------------------------
                def register_phase(unet, phase:int):
                    for _, m in unet.named_modules():
                        if isinstance_str(m, "BasicTransformerBlock"):
                            m.attn1.phase = phase  # 1 或 2
                            m.phase = phase
                # --------------------------------------------------
                # pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                # pivotal_idx = np.array([11,12,13,14])
                total_T = len(self.scheduler.timesteps)
                for step_idx, t in enumerate(self.scheduler.timesteps):                   
                    register_phase(self.unet, self.phase)
                    if t < 50:
                        self.use_normal_unet()
                    else:
                        register_normal_attn_flag(self.unet, False)
                    with torch.no_grad():           

                        noise_pred_text = []
                        noise_pred_image = []
                        noise_pred_uncond = []
                        # pivotal_idx = np.array([10,11,12,13])
                        pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                        register_pivotal(self.unet, True)                       
                        key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                        if self.cloth_mask is not None:
                            pivot_cloth = [self.cloth_mask[i] for i in pivotal_idx.tolist()]  
                            register_cloth_mask(self.unet, pivot_cloth)

                        true_pivots = []
                        for cam_local_idx in pivotal_idx.tolist():
                            cam_obj = cams[cam_local_idx]          # e.g.  cams[3]
                            colmap_id = cam_obj.colmap_id          
                            global_idx = colmap_id - 1             
                            true_pivots.append(global_idx)         

                        register_frames(self.unet, true_pivots)     
                        for _, module in self.unet.named_modules():
                            if isinstance_str(module, "BasicTransformerBlock"):
                                module.attn1.current_pivots = true_pivots

                        latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                        pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                        pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                        latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)
                        self.unet.current_pivots = pivotal_idx.tolist()
                        self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                        register_pivotal(self.unet, False)

                        for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                            register_batch_idx(self.unet, i)
                            register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                            true_frames = [
                                cam.colmap_id - 1                
                                for cam in cams[b : b + camera_batch_size]
                            ]
                            register_frames(self.unet, true_frames)   

                            maps_512_batch = [self.pix2shell_all[fid] for fid in true_frames]
                            maps_t_batch  = torch.stack(maps_512_batch, dim=0).unsqueeze(1).float()  

                            TOPK = 2                                         
                            vertex_constraints = {}

                            for R in [64, 32, 16, 8]:                      ### <<<
                                maps_R_batch = [                           ### <<<
                                    self.pix2shell_extra[R][fid]           ### <<<  
                                    for fid in true_frames
                                ]                                          ### <<<
                                H = W = R                                  ### <<<
                                L = R * R
                                patch = 1          

                                ids_norm = torch.full((camera_batch_size, L, TOPK), -1,
                                                    dtype=torch.int32, device=latents.device)
                                cnt_norm = torch.zeros_like(ids_norm)

                                for n in range(camera_batch_size):
                                    flat = maps_R_batch[n].to(latents.device).view(-1)  # (L,)
                                    ids_norm[n, :, 0] = flat                            
                                    cnt_norm[n, :, 0] = 1                               


                                ids_key = torch.full((4, L, TOPK), -1,
                                                    dtype=torch.int32, device=latents.device)
                                cnt_key = torch.zeros_like(ids_key)
                                for k_idx, kc in enumerate(key_cams):
                                    flat = self.pix2shell_extra[R][kc.colmap_id - 1] \
                                            .to(latents.device).view(-1)
                                    ids_key[k_idx, :, 0] = flat
                                    cnt_key[k_idx, :, 0] = 1

                                vc_batch = []                                    # -> (N,4,L,L)
                                for n in range(camera_batch_size):              
                                    per_key = []
                                    ids_n = ids_norm[n]                          # (L,K)
                                    cnt_n = cnt_norm[n]

                                    ids_n_u = ids_n.unsqueeze(1).unsqueeze(3)    # (L,1,K,1)
                                    cnt_n_u = cnt_n.unsqueeze(1).unsqueeze(3)    # (L,1,K,1)
                                    valid_n = ids_n_u >= 0

                                    for k_idx in range(4):
                                        ids_k = ids_key[k_idx]                   # (L,K)
                                        cnt_k = cnt_key[k_idx]

                                        ids_k_u = ids_k.unsqueeze(0).unsqueeze(2)    # (1,L,1,K)
                                        cnt_k_u = cnt_k.unsqueeze(0).unsqueeze(2)    # (1,L,1,K)
                                        valid_k = ids_k_u >= 0

                                        same = (ids_n_u == ids_k_u) & valid_n & valid_k

                                        inter = same.to(torch.int32)         # (L,L,K,K)
                                        W     = inter.sum(dim=(2,3))        

                                        best_j   = W.argmax(dim=1)          # (L,)
                                        best_val = W.max(dim=1).values      # (L,)

                                        mask = torch.zeros((L, L), dtype=torch.bool, device=latents.device)
                                        rows = torch.arange(L, device=latents.device)
                                        mask[rows, best_j] = best_val > 0

                                        if cams[b + n] is key_cams[k_idx]:
                                            mask[:] = False

                                        per_key.append(mask)
                                    vc_batch.append(torch.stack(per_key, dim=0))  # (4,L,L)

                                vertex_constraints[L] = torch.stack(vc_batch, dim=0)  # (N,4,L,L)

                            register_vertex_constraints(self.unet, vertex_constraints)                            

                            batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                            batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                            batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                            batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                            batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                            batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                            noise_pred_text.append(batch_noise_pred_text)
                            noise_pred_image.append(batch_noise_pred_image)
                            noise_pred_uncond.append(batch_noise_pred_uncond)

                        noise_pred_text = torch.cat(noise_pred_text, dim=0)
                        noise_pred_image = torch.cat(noise_pred_image, dim=0)
                        noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                        cond_min   = 0.4
                        cond_max = self.cfg.condition_scale        
                        beta      = 1                            
                        frac      = (step_idx + 1) / total_T      
                        cond_scale_t = 1.7 - (1.7-cond_min) * (frac ** beta)   
                        # --------------------------------------------
                        # perform classifier-free guidance
                        noise_pred = (
                            noise_pred_uncond
                            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                            + cond_scale_t * (noise_pred_image - noise_pred_uncond)
                        )

                        # get previous sample, continue loop
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ):
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t) 
        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]
        camera_batch_size = self.cfg.camera_batch_size
        
        with torch.no_grad():
            noise_pred_text = []
            noise_pred_image = []
            noise_pred_uncond = []
            pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0,len(latents),camera_batch_size) 
            print(pivotal_idx)
            register_pivotal(self.unet, True)

            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
            pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
            pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)
            
            key_cams = cams[pivotal_idx]
            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
            register_pivotal(self.unet, False)


            for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                register_batch_idx(self.unet, i)
                register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                
                epipolar_constrains = {}
                for down_sample_factor in [1, 2, 4, 8]:
                    H = current_H // down_sample_factor
                    W = current_W // down_sample_factor
                    epipolar_constrains[H * W] = []
                    for cam in cams[b:b + camera_batch_size]:
                        cam_epipolar_constrains = []
                        for key_cam in key_cams:
                            cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                        epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                    epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                register_epipolar_constrains(self.unet, epipolar_constrains)

                batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                noise_pred_text.append(batch_noise_pred_text)
                noise_pred_image.append(batch_noise_pred_image)
                noise_pred_uncond.append(batch_noise_pred_uncond)

            noise_pred_text = torch.cat(noise_pred_text, dim=0)
            noise_pred_image = torch.cat(noise_pred_image, dim=0)
            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

            # perform classifier-free guidance
            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad
    



    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        gaussians = None,
        cams= None,
        render=None,
        pipe=None,
        background=None,
        cloth_mask=None,
        **kwargs,
    ):
        assert cams is not None, "cams is required for THGE guidance"
        self.cloth_mask=cloth_mask
        batch_size, H, W, _ = rgb.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        RH, RW = height, width

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)
        
        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


