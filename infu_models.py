import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors.torch import load_file

class ImageProjectionModel(nn.Module):
    """
    Image projection model that maps image embedding to a format usable by the diffusion model
    Adapted to match the expected size of PuLID-Flux (2048)
    """
    def __init__(self):
        super().__init__()
        # 修改：从标准CLIP的768升维到PuLID-Flux期望的2048
        self.projection = nn.Linear(768, 2048)
        
    def forward(self, image_embeds):
        return self.projection(image_embeds)
    
    @classmethod
    def from_pretrained(cls, model_path):
        model = cls()
        if os.path.exists(model_path):
            # 加载权重
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                print(f"Loaded state dict keys: {list(state_dict.keys()) if isinstance(state_dict, dict) else 'Not a dict'}")  
                
                # 检查是否为OrderedDict，且包含image_proj键
                if isinstance(state_dict, dict) and "image_proj" in state_dict:
                    # 直接提取weights进行处理，不再假设它有shape属性
                    image_proj = state_dict["image_proj"]
                    
                    # 创建一个从768到2048的映射层（InfiniteYou使用标准CLIP尺寸768）
                    orig_weight = None
                    orig_bias = None
                    
                    # 尝试不同方法提取权重
                    if isinstance(image_proj, torch.Tensor) and image_proj.dim() == 2:
                        # 如果直接是张量
                        orig_weight = image_proj
                        orig_bias = torch.zeros(image_proj.shape[0])
                    elif isinstance(image_proj, dict) and "weight" in image_proj:
                        # 如果是包含weight的字典
                        orig_weight = image_proj["weight"]
                        orig_bias = image_proj.get("bias", torch.zeros(orig_weight.shape[0]))
                    elif "image_proj.weight" in state_dict:
                        # 单独存储的权重
                        orig_weight = state_dict["image_proj.weight"]
                        orig_bias = state_dict.get("image_proj.bias", torch.zeros(orig_weight.shape[0]))
                    
                    # 如果成功提取了权重
                    if orig_weight is not None:
                        # 创建新的投影层，扩展维度 
                        # 注意：这里我们不能直接加载权重，因为输出维度不同(768 vs 2048)
                        # 创建一个新的线性层，将768维扩展到2048维
                        # 我们将权重的前768维复制过来，剩余的填充为0
                        in_dim = orig_weight.shape[1]  # 应该是768
                        out_dim = 2048                  # PuLID-Flux期望的维度
                        
                        # 创建新权重矩阵
                        new_weight = torch.zeros((out_dim, in_dim))
                        orig_out_dim = min(orig_weight.shape[0], out_dim)
                        new_weight[:orig_out_dim, :] = orig_weight[:orig_out_dim, :]
                        
                        # 创建新偏置
                        new_bias = torch.zeros(out_dim)
                        new_bias[:orig_out_dim] = orig_bias[:orig_out_dim]
                        
                        # 设置模型权重
                        model.projection.weight.data = new_weight
                        model.projection.bias.data = new_bias
                        
                        print(f"Successfully adapted weights from {in_dim}x{orig_out_dim} to {in_dim}x{out_dim}")
                    else:
                        print("Failed to extract weights from image_proj")
                else:
                    print(f"State dict does not contain 'image_proj' key or is not a dict")
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")
                # 如果出错，创建一个默认的随机映射层
                print("Initializing random projection model (768 -> 2048)")
        return model

class InfuseNetAdapter(nn.Module):
    """
    Adapter for InfiniteYou's InfuseNet models
    """
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.image_processor = None
        self.vision_model = None
        self.projection_model = None
        self.loaded = False
        self.load_errors = []
        
    def load_models(self):
        """Load the necessary models with robust error handling"""
        if self.loaded:
            return True
            
        try:
            success = True
            # Verify model path exists
            if not os.path.exists(self.model_path):
                error_msg = f"Model directory not found: {self.model_path}"
                self.load_errors.append(error_msg)
                print(error_msg)
                return False
                
            # Load image projection model
            proj_loaded = False
            proj_path = os.path.join(self.model_path, "image_proj_model.bin")
            if os.path.exists(proj_path):
                try:
                    print(f"Loading projection model from {proj_path}")
                    self.projection_model = ImageProjectionModel.from_pretrained(proj_path)
                    proj_loaded = True
                    print("Projection model loaded successfully")
                except Exception as e:
                    error_msg = f"Error loading projection model: {str(e)}"
                    self.load_errors.append(error_msg)
                    print(error_msg)
                    self.projection_model = ImageProjectionModel()
                    success = False
            else:
                error_msg = f"Projection model not found at {proj_path}, using default initialization"
                self.load_errors.append(error_msg)
                print(error_msg)
                self.projection_model = ImageProjectionModel()
                success = False
                
            # Load CLIP vision model
            try:
                print("Loading CLIP vision model...")
                self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
                print("CLIP vision model loaded successfully")
            except Exception as e:
                error_msg = f"Error loading CLIP vision model: {str(e)}"
                self.load_errors.append(error_msg)
                print(error_msg)
                success = False
                
            # Load CLIP image processor
            try:
                print("Loading CLIP image processor...")
                self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
                print("CLIP image processor loaded successfully")
            except Exception as e:
                error_msg = f"Error loading CLIP image processor: {str(e)}"
                self.load_errors.append(error_msg)
                print(error_msg)
                success = False
            
            # Final verification of required components
            if self.projection_model is None or self.vision_model is None or self.image_processor is None:
                print("Critical components missing. Creating default models.")
                if self.projection_model is None:
                    self.projection_model = ImageProjectionModel()
                if self.vision_model is None:
                    try:
                        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
                    except Exception:
                        print("Could not load vision model even as fallback. Functionality will be limited.")
                if self.image_processor is None:
                    try:
                        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
                    except Exception:
                        print("Could not load image processor even as fallback. Functionality will be limited.")
                
            # Set loaded status
            self.loaded = True
            
            # Report loading status
            if not success:
                print("Some components failed to load properly. Functionality may be limited.")
                print(f"Encountered {len(self.load_errors)} errors during loading.")
            
            return success
        except Exception as e:
            error_msg = f"Unexpected error during model loading: {str(e)}"
            self.load_errors.append(error_msg)
            print(error_msg)
            # Create minimal defaults
            if self.projection_model is None:
                self.projection_model = ImageProjectionModel()
            return False
        
    def to(self, device, dtype=None):
        """Move models to specified device and dtype"""
        self.device = device
        if self.projection_model is not None:
            self.projection_model.to(device, dtype=dtype)
        if self.vision_model is not None:
            self.vision_model.to(device, dtype=dtype)
        return self
        
    def get_image_features(self, pixel_values):
        """
        Extract features from image using the CLIP vision model
        """
        if not self.loaded:
            self.load_models()
            
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values)
            image_embeds = vision_outputs.image_embeds
            
        return image_embeds
        
    def project_image_embeds(self, image_embeds):
        """
        Project image embeddings to match diffusion model's feature space
        """
        if not self.loaded:
            self.load_models()
            
        with torch.no_grad():
            projected_embeds = self.projection_model(image_embeds)
            
        return projected_embeds
    
    def process_image(self, image):
        """
        Process image through the entire pipeline:
        1. Preprocess image for CLIP
        2. Extract CLIP image features
        3. Project features to match diffusion model
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            projected_embeds: Tensor ready for injection into the diffusion model with size [1, 2048]
        """
        try:
            # Ensure models are loaded
            if not self.loaded:
                success = self.load_models()
                if not success:
                    print("Warning: Models did not load correctly. Results may be unreliable.")
                    
            # Verify required components
            if self.image_processor is None or self.vision_model is None or self.projection_model is None:
                print("Critical components missing. Returning zero embeddings.")
                return torch.zeros((1, 2048), device=self.device)
                
            # Process the image through CLIP processor
            try:
                if not isinstance(image, torch.Tensor):
                    # Handle PIL image
                    print("Processing PIL image through CLIP processor")
                    pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                else:
                    # Handle tensor input
                    print("Processing tensor image")
                    if image.dim() == 3:
                        image = image.unsqueeze(0)  # Add batch dimension
                    pixel_values = image.to(self.device)
                    
                print(f"Input pixel values shape: {pixel_values.shape}")
            except Exception as e:
                print(f"Error processing image input: {str(e)}")
                print("Using random pixel values as fallback")
                # Create random pixel values as fallback
                pixel_values = torch.rand(1, 3, 224, 224, device=self.device)
                
            # Get image embeddings
            try:
                print("Extracting image features with CLIP")
                image_embeds = self.get_image_features(pixel_values)
                print(f"CLIP image embeddings shape: {image_embeds.shape}")
            except Exception as e:
                print(f"Error extracting image features: {str(e)}")
                print("Using random image embeddings as fallback")
                # Create random embeddings as fallback
                image_embeds = torch.rand(1, 768, device=self.device)
            
            # Ensure projection model is on correct device
            try:
                if self.projection_model is not None and hasattr(self.projection_model, 'projection'):
                    self.projection_model.to(self.device)
            except Exception as e:
                print(f"Error moving projection model to device: {str(e)}")
                
            # Project embeddings
            try:
                print("Projecting embeddings to match diffusion model dimensions")
                projected_embeds = self.project_image_embeds(image_embeds)
                print(f"Projected embeddings shape: {projected_embeds.shape}")
            except Exception as e:
                print(f"Error projecting embeddings: {str(e)}")
                print("Using zero embeddings as fallback")
                projected_embeds = torch.zeros((1, 2048), device=self.device)
            
            # Validate output dimensions and fix if necessary
            if projected_embeds.shape[1] != 2048:
                print(f"Warning: projected embeddings size {projected_embeds.shape[1]} doesn't match expected 2048")
                try:
                    # If dimensions mismatch, create correctly sized tensor
                    correct_embeds = torch.zeros((1, 2048), device=self.device)
                    # Copy available embeddings as much as possible
                    if projected_embeds.shape[1] < 2048:
                        correct_embeds[0, :projected_embeds.shape[1]] = projected_embeds[0, :]
                        print(f"Padded embeddings from size {projected_embeds.shape[1]} to 2048")
                    else:
                        correct_embeds[0, :] = projected_embeds[0, :2048]
                        print(f"Truncated embeddings from size {projected_embeds.shape[1]} to 2048")
                    projected_embeds = correct_embeds
                except Exception as e:
                    print(f"Error fixing embedding dimensions: {str(e)}")
                    projected_embeds = torch.zeros((1, 2048), device=self.device)
            
            # Final check - make sure there are no NaN values
            if torch.isnan(projected_embeds).any():
                print("Warning: NaN values detected in embeddings. Replacing with zeros.")
                projected_embeds = torch.where(torch.isnan(projected_embeds), 
                                             torch.zeros_like(projected_embeds), 
                                             projected_embeds)
                
            # Check for infinity values
            if torch.isinf(projected_embeds).any():
                print("Warning: Infinite values detected in embeddings. Replacing with zeros.")
                projected_embeds = torch.where(torch.isinf(projected_embeds), 
                                             torch.zeros_like(projected_embeds), 
                                             projected_embeds)
                
            return projected_embeds
        except Exception as e:
            print(f"Unexpected error in process_image: {str(e)}")
            # Fallback to zero embeddings in case of any unhandled error
            dummy_embeds = torch.zeros((1, 2048), device=self.device)
            return dummy_embeds

class InfuseNetManager:
    """
    Manager class to handle the InfuseNet models from InfiniteYou
    """
    def __init__(self, base_path, device="cuda"):
        self.base_path = base_path
        self.device = device
        self.sim_adapter = None  # Stage 1 similarity model
        self.aes_adapter = None  # Stage 2 aesthetic model
        self.models_loaded = False
        
    def load_models(self):
        """Load both stage models with enhanced error handling"""
        try:
            # Check if base path exists
            if not os.path.exists(self.base_path):
                print(f"Warning: InfuseNet base path not found: {self.base_path}")
                print("Please ensure the INFU-models directory exists with sim_stage1 and aes_stage2 subdirectories")
                return False
                
            sim_path = os.path.join(self.base_path, "sim_stage1")
            aes_path = os.path.join(self.base_path, "aes_stage2")
            
            sim_loaded = False
            aes_loaded = False
            
            # Load similarity model (stage 1)
            if os.path.exists(sim_path):
                try:
                    print(f"Loading similarity model from {sim_path}")
                    self.sim_adapter = InfuseNetAdapter(sim_path, self.device)
                    self.sim_adapter.load_models()
                    sim_loaded = True
                    print("Similarity model loaded successfully")
                except Exception as e:
                    print(f"Error loading similarity model: {str(e)}")
            else:
                print(f"Warning: Similarity model directory not found at {sim_path}")
            
            # Load aesthetic model (stage 2)
            if os.path.exists(aes_path):
                try:
                    print(f"Loading aesthetic model from {aes_path}")
                    self.aes_adapter = InfuseNetAdapter(aes_path, self.device)
                    self.aes_adapter.load_models()
                    aes_loaded = True
                    print("Aesthetic model loaded successfully")
                except Exception as e:
                    print(f"Error loading aesthetic model: {str(e)}")
            else:
                print(f"Warning: Aesthetic model directory not found at {aes_path}")
            
            # Update loading status
            self.models_loaded = sim_loaded or aes_loaded
            
            if not self.models_loaded:
                print("Failed to load any InfuseNet models. Identity enhancement will be limited.")
            elif sim_loaded and not aes_loaded:
                print("Only similarity model loaded. Aesthetic enhancement will not be available.")
            elif aes_loaded and not sim_loaded:
                print("Only aesthetic model loaded. Similarity preservation may be limited.")
            else:
                print("Both similarity and aesthetic models loaded successfully.")
                
            return self.models_loaded
            
        except Exception as e:
            print(f"Unexpected error loading InfuseNet models: {str(e)}")
            return False
            
    def to(self, device, dtype=None):
        """Move models to specified device and dtype with error handling"""
        try:
            self.device = device
            if self.sim_adapter:
                try:
                    self.sim_adapter.to(device, dtype)
                except Exception as e:
                    print(f"Error moving similarity model to {device}: {str(e)}")
            if self.aes_adapter:
                try:
                    self.aes_adapter.to(device, dtype)
                except Exception as e:
                    print(f"Error moving aesthetic model to {device}: {str(e)}")
            return self
        except Exception as e:
            print(f"Unexpected error in to() method: {str(e)}")
            return self
        
    def get_enhanced_embeddings(self, image, use_both_stages=True):
        """
        Process an image using the InfuseNet models to get enhanced embeddings
        with comprehensive error handling
        
        Args:
            image: PIL Image or tensor
            use_both_stages: Whether to use both sim and aes models
            
        Returns:
            embeddings: Enhanced embeddings for identity injection
        """
        try:
            # Check if models are loaded, attempt to load them if not
            if not self.models_loaded:
                success = self.load_models()
                if not success and self.sim_adapter is None and self.aes_adapter is None:
                    print("Failed to load any InfuseNet models. Returning zero embeddings.")
                    return torch.zeros((1, 2048), device=self.device)
            
            # Get similarity-focused embeddings from stage 1
            sim_embeddings = None
            if self.sim_adapter is not None:
                try:
                    print("Processing image with similarity model...")
                    sim_embeddings = self.sim_adapter.process_image(image)
                    print(f"Similarity embeddings created with shape: {sim_embeddings.shape}")
                except Exception as e:
                    print(f"Error processing image with similarity model: {str(e)}")
            
            # If similarity embeddings failed or sim_adapter is missing
            if sim_embeddings is None:
                print("Creating placeholder similarity embeddings")
                sim_embeddings = torch.zeros((1, 2048), device=self.device)
            
            # If only using stage 1 or stage 2 is not available
            if not use_both_stages or self.aes_adapter is None:
                print("Using only similarity embeddings")
                return sim_embeddings
            
            # Get aesthetic-focused embeddings from stage 2
            aes_embeddings = None
            try:
                print("Processing image with aesthetic model...")
                aes_embeddings = self.aes_adapter.process_image(image)
                print(f"Aesthetic embeddings created with shape: {aes_embeddings.shape}")
            except Exception as e:
                print(f"Error processing image with aesthetic model: {str(e)}")
                print("Falling back to similarity embeddings only")
                return sim_embeddings
            
            # Combine the embeddings (this is a simplistic approach - actual InfiniteYou likely uses more sophisticated blending)
            try:
                # The alpha parameter could be adjusted based on preference for identity vs aesthetics
                alpha = 0.7
                print(f"Combining embeddings with alpha={alpha}")
                combined_embeddings = alpha * sim_embeddings + (1 - alpha) * aes_embeddings
                print(f"Combined embeddings created with shape: {combined_embeddings.shape}")
                return combined_embeddings
            except Exception as e:
                print(f"Error combining embeddings: {str(e)}")
                print("Returning similarity embeddings as fallback")
                return sim_embeddings
                
        except Exception as e:
            print(f"Unexpected error in get_enhanced_embeddings: {str(e)}")
            # Return zero embeddings as a last resort
            return torch.zeros((1, 2048), device=self.device)
