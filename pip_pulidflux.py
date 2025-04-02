# Import all necessary modules from the original implementation
from .encoders_flux import IDFormer, PerceiverAttentionCA
import torch
from torch import nn, Tensor
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import torch.nn.functional as F

# Import everything from the original pulidflux module for consistent functionality
from .pulidflux import PulidFluxModelLoader, PulidFluxInsightFaceLoader, PulidFluxEvaClipLoader, ApplyPulidFlux
from .pulidflux import tensor_to_image, image_to_tensor, resize_with_pad, to_gray
from .pulidflux import forward_orig, online_train


class PipApplyPulidFlux(ApplyPulidFlux):
    """Enhanced version of ApplyPulidFlux with simplified interface and orthogonal projection
    
    This class extends the original ApplyPulidFlux with a more user-friendly interface
    that includes a "method" parameter similar to the original PuLID implementation,
    and implements orthogonal projection for better results.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_flux": ("PULIDFLUX", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "method": (["fidelity", "adaptive", "residual"],),  # 
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
                "prior_image": ("IMAGE",),  # For train_weight scenario
                "fusion": (["auto", "mean", "concat", "max", "norm_id", "max_token", "auto_weight", "train_weight"],),  # Made optional
                "fusion_weight_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "fusion_weight_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "train_step": ("INT", {"default": 1000, "min": 0, "max": 20000, "step": 1 }),
                "use_gray": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pip_pulid_flux"
    CATEGORY = "pulid"

    def apply_pip_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, method, weight, start_at, end_at, 
                           prior_image=None, fusion=None, fusion_weight_max=1.0, fusion_weight_min=0.0, 
                           train_step=1000, use_gray=True, attn_mask=None, unique_id=None):
        
        # 
        original_fusion = fusion
        print(f"\n[PIP-PuLID-DEBUG] : = {method}, = {original_fusion}")
        
        # 
        ortho = False
        ortho_v2 = False
        adaptive = False   # 
        residual = False   # 
        
        if method == "fidelity":
            ortho_v2 = True  # 
            fusion = "norm_id" if fusion is None or fusion == "auto" else fusion
            print(f"[PIP-PuLID-DEBUG] : = {method}, fusion = {fusion}")
        elif method == "adaptive":
            adaptive = True   # 
            fusion = "concat" if fusion is None or fusion == "auto" else fusion
            print(f"[PIP-PuLID-DEBUG] : = {method}, fusion = {fusion}")
        elif method == "residual":
            residual = True   # 
            fusion = "max" if fusion is None or fusion == "auto" else fusion
            print(f"[PIP-PuLID-DEBUG] : = {method}, fusion = {fusion}")
        else:  # neutral
            # 
            fusion = "concat" if fusion is None or fusion == "auto" else fusion
            print(f"[PIP-PuLID-DEBUG] : = {method}, fusion = {fusion}")
        
        print(f"[PIP-PuLID-DEBUG] : = {method}, fusion = {fusion}, = {ortho}, = {ortho_v2}, = {adaptive}, = {residual}")
        
        # Call the parent class method with the appropriate parameters
        device = comfy.model_management.get_torch_device()
        dtype = model.model.diffusion_model.dtype
        # For 8bit use bfloat16 (because ufunc_add_CUDA is not implemented)
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            dtype = torch.bfloat16

        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)

        # 
        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if prior_image is not None:
            prior_image = resize_with_pad(prior_image.to(image.device, dtype=image.dtype), target_size=(image.shape[1], image.shape[2]))
            image=torch.cat((prior_image,image),dim=0)
        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        # 
        # 
        for i in range(image.shape[0]):
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            # 
            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                continue

            # 
            align_face = face_helper.cropped_faces[0]
            align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = face_helper.face_parse(functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)
            if use_gray:
                _align_face = to_gray(align_face)
            else:
                _align_face = align_face
            face_features_image = torch.where(bg, white_image, _align_face)

            # 
            face_features_image = functional.resize(face_features_image, eva_clip.image_size, 
                                                 transforms.InterpolationMode.BICUBIC if 'cuda' in device.type else transforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
            face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            # 
            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # 
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

            # 
            cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))

        if not cond:
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            return (model,)

        # 
        if fusion == "mean":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)
        elif fusion == "concat":
            cond = torch.cat(cond, dim=1).to(device, dtype=dtype)
        elif fusion == "max":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.max(cond, dim=0, keepdim=True)[0]
        elif fusion == "norm_id":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=(1,2))
                norm=norm/torch.sum(norm)
                cond=torch.einsum("wij,w->ij",cond,norm).unsqueeze(0)
        elif fusion == "max_token":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                _,idx=torch.max(norm,dim=0)
                cond=torch.stack([cond[j,i] for i,j in enumerate(idx)]).unsqueeze(0)
        elif fusion == "auto_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                order=torch.argsort(norm,descending=False,dim=0)
                regular_weight=torch.linspace(fusion_weight_min,fusion_weight_max,norm.shape[0]).to(device, dtype=dtype)

                _cond=[]
                for i in range(cond.shape[1]):
                    o=order[:,i]
                    _cond.append(torch.einsum('ij,i->j',cond[:,i,:],regular_weight[o]))
                cond=torch.stack(_cond,dim=0).unsqueeze(0)
        elif fusion == "train_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                if train_step > 0:
                    with torch.inference_mode(False):
                        cond = online_train(cond, device=cond.device, step=train_step)
                else:
                    cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # 
        flux_model = model.model.diffusion_model
        
        # 
        if not hasattr(flux_model, "pulid_ca"):
            # 
            def forward_orig_with_ortho(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None, control=None,
                                 transformer_options={}, attn_mask=None, **kwargs):
                patches_replace = transformer_options.get("patches_replace", {})

                if img.ndim != 3 or txt.ndim != 3:
                    raise ValueError("Input img and txt tensors must have 3 dimensions.")

                # 
                img = self.img_in(img)
                vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
                if self.params.guidance_embed:
                    if guidance is None:
                        raise ValueError("Didn't get guidance strength for guidance distilled model.")
                    vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

                vec = vec + self.vector_in(y)
                txt = self.txt_in(txt)

                ids = torch.cat((txt_ids, img_ids), dim=1)
                pe = self.pe_embedder(ids)

                ca_idx = 0
                blocks_replace = patches_replace.get("dit", {})
                
                # 
                for i, block in enumerate(self.double_blocks):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args):
                            out = {}
                            out["img"], out["txt"] = block(img=args["img"],
                                                           txt=args["txt"],
                                                           vec=args["vec"],
                                                           pe=args["pe"],
                                                           attn_mask=args.get("attn_mask"))
                            return out

                        out = blocks_replace[("double_block", i)]({"img": img,
                                                                   "txt": txt,
                                                                   "vec": vec,
                                                                   "pe": pe,
                                                                   "attn_mask": attn_mask},
                                                                  {"original_block": block_wrap})
                        txt = out["txt"]
                        img = out["img"]
                    else:
                        img, txt = block(img=img,
                                         txt=txt,
                                         vec=vec,
                                         pe=pe,
                                         attn_mask=attn_mask)

                    if control is not None: # Controlnet
                        control_i = control.get("input")
                        if i < len(control_i):
                            add = control_i[i]
                            if add is not None:
                                img += add

                    # 
                    if hasattr(self, "pulid_data") and self.pulid_data:
                        if i % self.pulid_double_interval == 0:
                            # 
                            for uid, node_data in self.pulid_data.items():
                                condition_start = node_data['sigma_start'] >= timesteps
                                condition_end = timesteps >= node_data['sigma_end']
                                condition = torch.logical_and(condition_start, condition_end).all()
                                
                                if condition:
                                    # 
                                    orig_attn_result = self.pulid_ca[ca_idx](node_data['embedding'], img)
                                    
                                    # 
                                    if node_data.get('ortho_v2', False):
                                        # fidelity
                                        img_float = img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        # 
                                        projection = (torch.sum((img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((img_float * img_float), dim=-2, keepdim=True) * img_float)
                                        # 
                                        orthogonal = attn_float - projection
                                        img = img + node_data['weight'] * orthogonal.to(img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] fidelity = {method}, = {uid}")
                                    elif node_data.get('ortho', False):
                                        # pose_free
                                        img_float = img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        projection = (torch.sum((img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((img_float * img_float), dim=-2, keepdim=True) * img_float)
                                        
                                        # 
                                        num_channels = attn_float.shape[-1]
                                        pose_weight = torch.ones((1, 1, num_channels), device=attn_float.device, dtype=attn_float.dtype)
                                        
                                        # 
                                        pose_channels = num_channels // 3  
                                        pose_weight[..., :pose_channels] = 3.5  
                                        
                                        # 
                                        enhanced_projection = projection * pose_weight
                                        orthogonal = attn_float - enhanced_projection
                                        
                                        img = img + node_data['weight'] * orthogonal.to(img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] pose_free = {method}, = {uid}")
                                    elif 'adaptive' in node_data and node_data['adaptive']:
                                        # adaptive
                                        img_float = img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        projection = (torch.sum((img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((img_float * img_float), dim=-2, keepdim=True) * img_float)
                                        
                                        # 
                                        img_features = img_float.mean(dim=1, keepdim=True)  # 
                                        attn_features = attn_float.mean(dim=1, keepdim=True)
                                        
                                        # 
                                        corr = torch.abs(img_features * attn_features).mean(dim=1, keepdim=True)  # 
                                        
                                        # 
                                        norm_corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)  # 
                                        
                                        # 
                                        inv_corr = 1.0 - norm_corr
                                        
                                        # 
                                        dynamic_weights = 1.0 + 3.0 * inv_corr
                                        
                                        # 
                                        enhanced_projection = projection * dynamic_weights
                                        orthogonal = attn_float - enhanced_projection
                                        
                                        img = img + node_data['weight'] * orthogonal.to(img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] adaptive = {method}, = {uid}")
                                    elif 'residual' in node_data and node_data['residual']:
                                        # residual - 
                                        img_float = img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        img_features = img_float.mean(dim=1, keepdim=True)  # 
                                        attn_features = attn_float.mean(dim=1, keepdim=True)
                                        
                                        # 
                                        corr = torch.abs(img_features * attn_features).mean(dim=1, keepdim=True) 
                                        norm_corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)
                                        
                                        # 
                                        identity_mask = (norm_corr > 0.4).float()  # 
                                        
                                        # 
                                        identity_features = attn_float * (identity_mask * 1.5 + 0.2)  # 
                                        
                                        # 
                                        img = img + node_data['weight'] * 1.2 * identity_features.to(img.dtype)  # 
                                        print(f"[PIP-PuLID-DEBUG] - residual {method}, = {uid}")
                                    else:
                                        # neutral
                                        img = img + node_data['weight'] * orig_attn_result
                                        print(f"[PIP-PuLID-DEBUG] neutral = {method}, = {uid}")
                            ca_idx += 1

                # 
                img = torch.cat((txt, img), 1)
                
                # 
                for i, block in enumerate(self.single_blocks):
                    if ("single_block", i) in blocks_replace:
                        def block_wrap(args):
                            out = {}
                            out["img"] = block(args["img"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                            return out

                        out = blocks_replace[("single_block", i)]({"img": img,
                                                                   "vec": vec,
                                                                   "pe": pe,
                                                                   "attn_mask": attn_mask}, 
                                                                  {"original_block": block_wrap})
                        img = out["img"]
                    else:
                        img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                    if control is not None: # Controlnet
                        control_o = control.get("output")
                        if i < len(control_o):
                            add = control_o[i]
                            if add is not None:
                                img[:, txt.shape[1] :, ...] += add

                    # 
                    if hasattr(self, "pulid_data") and self.pulid_data:
                        real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                        if i % self.pulid_single_interval == 0:
                            # 
                            for uid, node_data in self.pulid_data.items():
                                condition_start = node_data['sigma_start'] >= timesteps
                                condition_end = timesteps >= node_data['sigma_end']
                                condition = torch.logical_and(condition_start, condition_end).all()

                                if condition:
                                    # 
                                    orig_attn_result = self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                                    
                                    # 
                                    if node_data.get('ortho_v2', False):
                                        # fidelity
                                        real_img_float = real_img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        # 
                                        projection = (torch.sum((real_img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((real_img_float * real_img_float), dim=-2, keepdim=True) * real_img_float)
                                        # 
                                        orthogonal = attn_float - projection
                                        real_img = real_img + node_data['weight'] * orthogonal.to(real_img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] fidelity = {method}, = {uid}")
                                    elif node_data.get('ortho', False):
                                        # pose_free
                                        real_img_float = real_img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        projection = (torch.sum((real_img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((real_img_float * real_img_float), dim=-2, keepdim=True) * real_img_float)
                                        
                                        # 
                                        num_channels = attn_float.shape[-1]
                                        pose_weight = torch.ones((1, 1, num_channels), device=attn_float.device, dtype=attn_float.dtype)
                                        
                                        # 
                                        pose_channels = num_channels // 3  
                                        pose_weight[..., :pose_channels] = 3.5  
                                        
                                        # 
                                        enhanced_projection = projection * pose_weight
                                        orthogonal = attn_float - enhanced_projection
                                        
                                        real_img = real_img + node_data['weight'] * orthogonal.to(real_img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] pose_free = {method}, = {uid}")
                                    elif 'adaptive' in node_data and node_data['adaptive']:
                                        # adaptive
                                        real_img_float = real_img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        projection = (torch.sum((real_img_float * attn_float), dim=-2, keepdim=True) / 
                                                      torch.sum((real_img_float * real_img_float), dim=-2, keepdim=True) * real_img_float)
                                        
                                        # 
                                        img_features = real_img_float.mean(dim=1, keepdim=True)  # 
                                        attn_features = attn_float.mean(dim=1, keepdim=True)
                                        
                                        # 
                                        corr = torch.abs(img_features * attn_features).mean(dim=1, keepdim=True)  # 
                                        
                                        # 
                                        norm_corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)  # 
                                        
                                        # 
                                        inv_corr = 1.0 - norm_corr
                                        
                                        # 
                                        dynamic_weights = 1.0 + 3.0 * inv_corr
                                        
                                        # 
                                        enhanced_projection = projection * dynamic_weights
                                        orthogonal = attn_float - enhanced_projection
                                        
                                        real_img = real_img + node_data['weight'] * orthogonal.to(real_img.dtype)
                                        print(f"[PIP-PuLID-DEBUG] adaptive = {method}, = {uid}")
                                    elif 'residual' in node_data and node_data['residual']:
                                        # residual - 
                                        real_img_float = real_img.to(dtype=torch.float32)
                                        attn_float = orig_attn_result.to(dtype=torch.float32)
                                        
                                        # 
                                        img_features = real_img_float.mean(dim=1, keepdim=True)  # 
                                        attn_features = attn_float.mean(dim=1, keepdim=True)
                                        
                                        # 
                                        corr = torch.abs(img_features * attn_features).mean(dim=1, keepdim=True) 
                                        norm_corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)
                                        
                                        # 
                                        identity_mask = (norm_corr > 0.4).float()  # 
                                        
                                        # 
                                        identity_features = attn_float * (identity_mask * 1.5 + 0.2)  # 
                                        
                                        # 
                                        real_img = real_img + node_data['weight'] * 1.2 * identity_features.to(real_img.dtype)  # 
                                        print(f"[PIP-PuLID-DEBUG] - residual {method}, = {uid}")
                                    else:
                                        # neutral
                                        real_img = real_img + node_data['weight'] * orig_attn_result
                                        print(f"[PIP-PuLID-DEBUG] neutral = {method}, = {uid}")
                            ca_idx += 1
                        img = torch.cat((txt, real_img), 1)

                # 从生成的图像中移除文本部分
                img = img[:, txt.shape[1] :, ...]
                img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
                return img

            # 
            flux_model.pulid_ca = pulid_flux.pulid_ca
            flux_model.pulid_double_interval = pulid_flux.double_interval
            flux_model.pulid_single_interval = pulid_flux.single_interval
            flux_model.pulid_data = {}
            # 
            new_method = forward_orig_with_ortho.__get__(flux_model, flux_model.__class__)
            setattr(flux_model, 'forward_orig', new_method)
            print(f"[PIP-PuLID-DEBUG] 应用方法 = {method}, 唯一ID = {unique_id}")

        # 
        flux_model.pulid_data[unique_id] = {
            'weight': weight,
            'embedding': cond,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end,
            'ortho': ortho,          # 
            'ortho_v2': ortho_v2,    # 
            'adaptive': adaptive,     # 
            'residual': residual,     # 
        }

        # 
        self.pulid_data_dict = {'data': flux_model.pulid_data, 'unique_id': unique_id}
        
        print(f"[PIP-PuLID-DEBUG] : = {method}, = {fusion}, = {ortho or ortho_v2}\n")
        return (model,)
    
    def __del__(self):
        # 
        if hasattr(self, 'pulid_data_dict') and self.pulid_data_dict:
            del self.pulid_data_dict['data'][self.pulid_data_dict['unique_id']]
            del self.pulid_data_dict


# Update NODE_CLASS_MAPPINGS to include the new node
NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,  # Include original nodes
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,  # Original node
    "PipApplyPulidFlux": PipApplyPulidFlux,  # New enhanced node
}

# Update display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
    "PipApplyPulidFlux": "Apply PIP PuLID Flux",  # Display name for new node
}
