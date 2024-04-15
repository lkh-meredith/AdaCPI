import os.path as osp
from collections import OrderedDict
import time
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import datetime
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import (MetricMeter, AverageMeter, mkdir_if_missing)
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from tqdm import tqdm
import collections

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ADA_CPI',
                      "visual_ctx_length": cfg.TRAINER.ADA_CPI.N_CTX,
                      "text_ctx_length": cfg.TRAINER.ADA_CPI.N_CTX,
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details, pro=True)

    return model

class Cross_attn_mapping_layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1): #visual->text
        super().__init__()
        self.mapping = nn.Linear(in_dim, out_dim) # for dim 768->512
        self.cross_attention = nn.MultiheadAttention(out_dim, num_heads=num_heads)

    def forward(self, k_v_context, q_context, nctx): #q(text): 8 100 512 k_v(visual): 8 4 768,  k_v -> q  , attention_mask
        q_context_pooling = q_context.mean(dim=0) #4 768 , 16 768
        k_v_context_pooling = k_v_context.mean(dim=0) #100 512, 100 512
        k_v_context_proj = self.mapping(k_v_context_pooling) #100 768, 100 512
        # attention_mask = attention_mask.to(dtype=q_context.dtype, device=q_context.device) #4 100
        fusion_context = self.cross_attention(query=q_context_pooling, key=k_v_context_proj, value=k_v_context_proj, need_weights=False)[0] #4 768  ,attn_mask=attention_mask
        fusion_context = fusion_context.expand(nctx,-1,-1) # 8 4 768
        return fusion_context

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, num=None, print_info=False):
        super().__init__()
        n_cls = len(classnames) #100
        nctx = cfg.TRAINER.ADA_CPI.N_CTX
        ctx_init = cfg.TRAINER.ADA_CPI.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # initialize text context
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        prefix_text = ctx_init
        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        hard_prompts = [prefix_text + " " + name + "." for name in classnames]

        tokenized_hard_prompts = torch.cat([clip.tokenize(p) for p in hard_prompts])  # (n_cls, n_tkn) #100 77
        with torch.no_grad():
            self.text_hard_prompt_embedding = clip_model.token_embedding(tokenized_hard_prompts).type(dtype)  # 100 77 512

        # initialize visual context
        ctx_vectors_visual = torch.empty(nctx, 768, dtype=dtype)  # 4 768
        nn.init.normal_(ctx_vectors_visual, std=0.02)
        self.prompts_visual = nn.Parameter(ctx_vectors_visual)

        # random initialization for text prompt
        ctx_vectors_text = torch.empty(nctx, ctx_dim, dtype=dtype) # 4 512
        nn.init.normal_(ctx_vectors_text, std=0.02)
        random_prefix_prompt = " ".join(["X"] * nctx) #第一层随机初始化 X作为占位符
        self.prompts_opt_text = nn.Parameter(ctx_vectors_text)
        if num != None:
            print(f"{num}th MultiPromptLearner has constructed!")
            if print_info:
                print(f'Initial textual context: "{ctx_init}"')
                print(f"random textual context: {random_prefix_prompt}")
                print(f"Number of ADA_CPI text context words (tokens): {nctx}")
                print(f"Number of ADA_CPI visual context words (tokens): {nctx}")

        else:
            print('ADA_CPI design: Multi-modal Prompt Learning')
            print(f'Initial textual context: "{ctx_init}"')
            print(f"random textual context: {random_prefix_prompt}")
            print(f"Number of ADA_CPI text context words (tokens): {nctx}")
            print(f"Number of ADA_CPI visual context words (tokens): {nctx}")

        # classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [random_prefix_prompt + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn) #100 77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 100 77 512
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS #100 1 512
        self.register_buffer("token_suffix", embedding[:, 1 + nctx:, :])  # CLS, EOS 100 64 512

        self.n_cls = n_cls
        # self.visual_nctx = visual_nctx
        # self.text_nctx = text_nctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_hard_prompts = tokenized_hard_prompts
        # self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(  # 100 77 512
            [
                prefix,  # (dim0, 1, dim) cls
                ctx,  # (dim0, n_ctx, dim) prompt
                suffix,  # (dim0, *, dim) label+eos
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        t_ctx = self.prompts_opt_text  # 4 512

        if t_ctx.dim() == 2:
            t_ctx = t_ctx.unsqueeze(0).expand(self.n_cls, -1, -1) #100 4 512

        prefix = self.token_prefix  # 100 1 512
        suffix = self.token_suffix  # 100 64 512
        prompts_text = self.construct_prompts(t_ctx, prefix, suffix)  # 100 77 512

        prompts_visual = self.prompts_visual  # 12 768
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts_text, prompts_visual #, self.fusion_text_prompt_list, self.fusion_visual_prompt_list

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder_PRO(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        #visual
        self.conv1_visual = clip_model.conv1_visual
        self.class_embedding_visual = clip_model.class_embedding_visual
        self.positional_embedding_visual = clip_model.positional_embedding_visual
        self.ln_pre_visual = clip_model.ln_pre_visual
        self.ln_post_visual = clip_model.ln_post_visual
        self.proj_visual = clip_model.proj_visual

        # visual and language attention
        self.resblocks = clip_model.resblocks

        # language
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x_visual, visual_ctx, prompts_text, tokenized_prompts, fusion_depth=None,
                t2v_mapping_layer=None, v2t_mapping_layer=None):

        # visual pre-pro
        x_visual = self.conv1_visual(x_visual)  # shape = [*, width, grid, grid]
        x_visual = x_visual.reshape(x_visual.shape[0], x_visual.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_visual = x_visual.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_visual = torch.cat(
            [self.class_embedding_visual.to(x_visual.dtype) + torch.zeros(x_visual.shape[0], 1, x_visual.shape[-1],
                                                                          dtype=x_visual.dtype, device=x_visual.device),
             x_visual], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_visual = x_visual + self.positional_embedding_visual.to(x_visual.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        visual_ctx = visual_ctx.expand(x_visual.shape[0], -1, -1)
        x_visual = torch.cat([x_visual, visual_ctx], dim=1)  # 4 209 768

        # Normal code as before
        x_visual = self.ln_pre_visual(x_visual)

        x_visual = x_visual.permute(1, 0, 2)  # NLD -> LND # 209 4 768

        # text pre pro
        x_text = prompts_text + self.positional_embedding.type(self.dtype)  # 100 77 512
        x_text = x_text.permute(1, 0, 2)  # NLD -> LND # 77 100 512
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass

        combined = [x_visual, x_text, fusion_depth, 0, t2v_mapping_layer, v2t_mapping_layer]  # third argument is the counter which denotes depth of prompt

        outputs = self.resblocks(combined)

        # visual post_pro
        x_visual = outputs[0]
        x_visual = x_visual.permute(1, 0, 2)  # LND -> NLD  4 213 768

        x_visual = self.ln_post_visual(x_visual[:, 0, :])

        if self.proj_visual is not None:
            x_visual = x_visual @ self.proj_visual.half()  # 4 512

        # text post_pro
        x_text = outputs[1]  # extract the x back from here
        x_text = x_text.permute(1, 0, 2)  # LND -> NLD
        x_text = self.ln_final(x_text).type(self.dtype) #100 512

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_text = x_text[torch.arange(x_text.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.half()  # 100 512

        return x_visual, x_text

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class VisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # visual
        self.conv1_visual = clip_model.conv1_visual
        self.class_embedding_visual = clip_model.class_embedding_visual
        self.positional_embedding_visual = clip_model.positional_embedding_visual
        self.ln_pre_visual = clip_model.ln_pre_visual
        self.ln_post_visual = clip_model.ln_post_visual
        self.proj_visual = clip_model.proj_visual
        self.transformer = clip_model.visual_transformer

    def forward(self, x):
        # visual pre-pro
        x = self.conv1_visual(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
         [self.class_embedding_visual.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                       dtype=x.dtype,device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding_visual.to(x.dtype)
        x = self.ln_pre_visual(x)
        x = x.permute(1, 0, 2)  # NLD -> LND # 209 4 768

        x = self.transformer(x)

        # visual post_pro
        x = x.permute(1, 0, 2)  # LND -> NLD  4 213 768
        x = self.ln_post_visual(x[:, 0, :])

        if self.proj_visual is not None:
            x = x @ self.proj_visual.half()  # 4 512

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, text_encoder, vision_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

        self.cross_attention = nn.MultiheadAttention(512, num_heads=4).half()
        self.classifier = nn.Linear(512, 6).half()
        # nn.init.normal_(self.classifier.weight, std=0.02)
        # self.initialize_parameters()


    def forward(self, image, prompts_text, tokenized_prompts):
        with torch.no_grad():
            text_feature_zs = self.text_encoder(prompts_text, tokenized_prompts) #100 512
            vision_feature_zs = self.vision_encoder(image) #4 512

        # #采用image和text之间的关系特征进行分类，决定融合的层数
        feature = self.cross_attention(query=text_feature_zs,key=vision_feature_zs,value=vision_feature_zs, need_weights=False)[0]
        feature = torch.mean(feature, dim=0).unsqueeze(0)
        logits = self.classifier(feature)

        # return logits
        if self.training:
            with torch.no_grad():
                text_feature_zs = text_feature_zs / text_feature_zs.norm(dim=-1, keepdim=True)  # 100 512
            return logits, text_feature_zs

        else:
            return logits

class CustomCLIP_ADAPTIVE(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.fusion_depth = cfg.TRAINER.ADA_CPI.FUSION_DEPTH  # 0-12， 0表示无融合层，1-12分别表示后i层融合
        self.adaptive_fusion = cfg.TRAINER.ADA_CPI.ADAPTIVE_FUSION

        # Initialize module
        if self.adaptive_fusion:
            #所有情况都共享参数
            # self.t2v_mapping_layer = Cross_attn_mapping_layer(clip_model.embed_dim, clip_model.vision_width,
            #                                                   num_heads=4)
            # self.v2t_mapping_layer = Cross_attn_mapping_layer(clip_model.vision_width, clip_model.embed_dim,
            #                                                   num_heads=4)
            self.t2v_mapping_layer_list = nn.ModuleList(
                [Cross_attn_mapping_layer(clip_model.embed_dim, clip_model.vision_width, num_heads=4)
                 for _ in range(6)])
            self.v2t_mapping_layer_list = nn.ModuleList(
                [Cross_attn_mapping_layer(clip_model.vision_width, clip_model.embed_dim, num_heads=4)
                 for _ in range(6)])#每一种情况都共享参数

            self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)

            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.tokenized_hard_prompts = self.prompt_learner.tokenized_hard_prompts
            self.text_hard_prompt_embedding = self.prompt_learner.text_hard_prompt_embedding

            # self.prompt_learner_list = nn.ModuleList(
            #     [MultiModalPromptLearner(cfg, classnames, clip_model, num=i, print_info=(i==5)) for i in range(6)])
            #
            # self.tokenized_prompts = self.prompt_learner_list[0].tokenized_prompts
            # self.tokenized_hard_prompts = self.prompt_learner_list[0].tokenized_hard_prompts
            # self.text_hard_prompt_embedding = self.prompt_learner_list[0].text_hard_prompt_embedding

        elif self.fusion_depth != 0:
            self.t2v_mapping_layer = Cross_attn_mapping_layer(clip_model.embed_dim, clip_model.vision_width, num_heads=4)
            self.v2t_mapping_layer = Cross_attn_mapping_layer(clip_model.vision_width, clip_model.embed_dim, num_heads=4)

            self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)

            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.tokenized_hard_prompts = self.prompt_learner.tokenized_hard_prompts
            self.text_hard_prompt_embedding = self.prompt_learner.text_hard_prompt_embedding

        else: # VL independent
            self.t2v_mapping_layer = None
            self.v2t_mapping_layer = None
            self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)

            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.tokenized_hard_prompts = self.prompt_learner.tokenized_hard_prompts
            self.text_hard_prompt_embedding = self.prompt_learner.text_hard_prompt_embedding

        self.encoder = Encoder_PRO(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.visual_eval_batch_size = cfg.DATALOADER.TEST.BATCH_SIZE
        self.n_cls = len(classnames)


    def forward(self, image, one_hot=None, fusion_depth=None):
        # last_batch = False
        # attention_mask = torch.ones((self.n_cls, image.size(0)))#102 4
        # if not self.training: #training时会自动dropout
        #     if image.size(0) != self.visual_eval_batch_size:
        #         last_batch = True
        #         num_len = image.size(0)
        #         num_pad = abs(num_len - self.visual_eval_batch_size)
        #         padding = torch.zeros((num_pad, image.size(1),image.size(2),image.size(3)),dtype=image.dtype, device=image.device)
        #         image = torch.cat([image, padding], dim=0)
        #         attention_mask = torch.cat((torch.ones((1,num_len)),torch.zeros((1, num_pad))),dim=1).repeat(self.n_cls,1) #102 16

        logit_scale = self.logit_scale.exp()

        prompts_text, prompts_visual = self.prompt_learner()

        if self.adaptive_fusion:
            if self.training and fusion_depth == None:
                logits_l = 0
                text_features_l = 0

                for index in range(6):
                    # prompt_learner = self.prompt_learner_list[index]

                    image_features, text_features = self.encoder(image.type(self.dtype), prompts_visual,
                                                                 prompts_text, self.tokenized_prompts, index*2+2,
                                                                 t2v_mapping_layer=self.t2v_mapping_layer_list[index].half(), v2t_mapping_layer=self.v2t_mapping_layer_list[index].half(),
                                                                 )

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits = logit_scale * image_features @ text_features.t()

                    logits_l += logits * one_hot[index].view(-1, 1) #one hot shape:[6]
                    text_features_l += text_features * one_hot[index].view(-1, 1)

                logits = logits_l
                text_features = text_features_l

            elif fusion_depth != None:
                # prompt_learner = self.prompt_learner_list[fusion_depth]
                # prompts_text, prompts_visual = prompt_learner()

                t2v_mapping_layer = self.t2v_mapping_layer_list[fusion_depth]
                v2t_mapping_layer = self.v2t_mapping_layer_list[fusion_depth]

                t2v_mapping_layer = t2v_mapping_layer.half()
                v2t_mapping_layer = v2t_mapping_layer.half()

                image_features, text_features = self.encoder(image.type(self.dtype), prompts_visual,
                                                             prompts_text, self.tokenized_prompts, fusion_depth*2+2,
                                                             t2v_mapping_layer=t2v_mapping_layer,
                                                             v2t_mapping_layer=v2t_mapping_layer,
                                                             )

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()  # 1 4 102

        else:
            if self.t2v_mapping_layer != None:
                self.t2v_mapping_layer.half()
            if self.v2t_mapping_layer != None:
                self.v2t_mapping_layer.half()

            prompts_text, prompts_visual = self.prompt_learner()

            image_features, text_features = self.encoder(image.type(self.dtype), prompts_visual,
                                                         prompts_text, self.tokenized_prompts, self.fusion_depth,#表示融合深度
                                                         t2v_mapping_layer=self.t2v_mapping_layer,
                                                         v2t_mapping_layer=self.v2t_mapping_layer,
                                                         )

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()  # 1 4 102

        # return logits
        if self.training:
            return logits, text_features

        else:
            # if last_batch:
            #     logits = logits[0:num_len, :]

            return logits


@TRAINER_REGISTRY.register()
class ADA_CPI(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADA_CPI.PREC in ["fp16", "fp32", "amp"] # CLIP's default precision is fp16

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        # initialize
        self.adaptive_fusion = cfg.TRAINER.ADA_CPI.ADAPTIVE_FUSION
        self.fusion_depth = cfg.TRAINER.ADA_CPI.FUSION_DEPTH
        self.lambda_w = cfg.TRAINER.ADA_CPI.W
        self.warmup_epoch = cfg.OPTIM.WARMUP_EPOCH
        # self.temperature = cfg.TRAINER.ADA_CPI.TAU
        # self.train_batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if self.adaptive_fusion:
            self.temperature = cfg.TRAINER.ADA_CPI.TEMPERATURE
            self.temperature_decay_list = np.linspace(5.0, 0.5, num=self.max_epoch-self.warmup_epoch) #线性衰减

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADA_CPI.PREC == "fp32" or cfg.TRAINER.ADA_CPI.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP_ADAPTIVE(cfg, classnames, clip_model)

        # policy network
        self.policy_network = PolicyNetwork(TextEncoder(clip_model), VisionEncoder(clip_model))

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                if "t2v_mapping_layer" in name:
                    param.requires_grad_(True)
                elif "v2t_mapping_layer" in name:
                    param.requires_grad_(True)
                # elif "t2v_mapping_layer_list" in name:
                #     param.requires_grad_(True)
                # elif "v2t_mapping_layer_list" in name:
                #     param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        if self.adaptive_fusion:
            for name, param in self.policy_network.named_parameters():
                if "classifier" in name:
                    param.requires_grad_(True)
                elif "cross_attention" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:
            for name, param in self.policy_network.named_parameters():
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        for name, param in self.policy_network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pytorch_total_params += sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)

        print(f"Updated parameters count: {pytorch_total_params}")
        # self.device = torch.device("cuda:1")
        self.model.to(self.device)
        self.policy_network.to(self.device)

        self.optim1 = build_optimizer(self.model, cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.register_model("model", self.model, self.optim1, self.sched1)

        if self.adaptive_fusion: #只有自适应时才需要优化
            self.optim2 = build_optimizer(self.policy_network, cfg.OPTIM1)
            self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM1)
            self.register_model("policy_network",self.policy_network,self.optim2,self.sched2)

        self.scaler = GradScaler() if cfg.TRAINER.ADA_CPI.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()


    def train(self):
        self.before_train()  # prepare
        for self.epoch in range(self.start_epoch, self.max_epoch):
            if self.adaptive_fusion:
                if self.epoch >= self.warmup_epoch:
                    self.temperature = self.temperature_decay_list[self.epoch-self.warmup_epoch]

                print(f"temperature on epoch {self.epoch} is: {self.temperature}")
                self.write_scalar("train/temperature", self.temperature, self.epoch)

            self.run_epoch()
            self.after_epoch() #保存模型

        if self.adaptive_fusion:
            print(f"current temperature is {self.temperature}")
        self.after_train()


    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")

            print("=== Evaluation in test data ===")
            self.test(split="test")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()


    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()

        # fusion_num_dict = {"7": 0, "8": 0, "9": 0, "10": 0, "11": 0, "12": 0}
        fusion_num_dict = {"2": 0, "4": 0, "6": 0, "8": 0, "10": 0, "12": 0}

        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.set_model_mode("train")
            data_time.update(time.time() - end)

            loss_summary,one_hot = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # fusion_depth = torch.argmax(one_hot) + 6 #
            if self.adaptive_fusion:
                fusion_depth = torch.argmax(one_hot) * 2 + 2
                # fusion_num_dict[str(int(fution_depth))] += 1
                fusion_num_dict[str(int(fusion_depth))] += 1

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"model_lr {self.get_current_lr('model'):.4e}"]

                if self.adaptive_fusion:
                    info += [f"policy_lr {self.get_current_lr('policy_network'):.4e}"]

                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/model_lr", self.get_current_lr('model'), n_iter)

            if self.adaptive_fusion:
                self.write_scalar("train/policy_lr", self.get_current_lr('policy_network'), n_iter)

            end = time.time()

        if self.adaptive_fusion:
            for key, value in fusion_num_dict.items():
                fusion_num_dict[key] = value / self.num_batches
                self.write_scalar(f"train/ratio/{key}", fusion_num_dict[key], self.epoch)

            print(f"Training on {self.epoch} epoch: {fusion_num_dict}")

        # self.test("val")

        return

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        policy_network = self.policy_network

        prec = self.cfg.TRAINER.ADA_CPI.PREC

        if prec == "fp16":
            if self.adaptive_fusion:
                logits, text_features_zs = policy_network(image.type(model.dtype),
                                                          model.text_hard_prompt_embedding.to(image.device),
                                                          model.tokenized_hard_prompts)

                # gumble softmax
                dist_logits = F.log_softmax(logits, dim=1)
                one_hot = F.gumbel_softmax(dist_logits, tau=self.temperature, hard=True).squeeze(0)  # 12
                fusion_depth = None

            else:
                fusion_depth = None
                one_hot = None
                with torch.no_grad():
                    text_features_zs = policy_network.text_encoder(model.text_hard_prompt_embedding.to(image.device),
                                                                   model.tokenized_hard_prompts)

                    text_features_zs = text_features_zs / text_features_zs.norm(dim=-1, keepdim=True)  # 100 512


            logits,text_features = model(image, one_hot=one_hot, fusion_depth=fusion_depth) #text_features

            loss_ce = F.cross_entropy(logits, label) #loss_ce

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = cos(text_features, text_features_zs)
            score = 1.0 - torch.mean(score)
            # loss_scl = F.l1_loss(text_features, text_features_zs.cuda(), reduction='mean')
            loss = self.lambda_w * score + loss_ce

            self.model_zero_grad()
            self.model_backward(loss)
            self.model_update()

            loss_summary = {"loss": loss.item()}

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            return loss_summary, one_hot

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        loss_l = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # fusion_num_dict = {"6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0} #实际指的是融合后7-12层
        fusion_num_dict = {"2": 0, "4": 0, "6": 0, "8": 0, "10": 0, "12": 0} #指融合后2、4、6、8、10、12层
        # fusion_num_dict = {"7": 0, "8": 0, "9": 0, "10": 0, "11": 0, "12": 0}
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            if self.adaptive_fusion:
                logits = self.policy_network(image.type(self.model.dtype),
                                              self.model.text_hard_prompt_embedding.to(image.device),
                                              self.model.tokenized_hard_prompts)  # 获得分布

                dist_logits = F.log_softmax(logits, dim=1)

                probs = F.gumbel_softmax(dist_logits,tau=self.temperature, dim=1)
                # fusion_depth = torch.argmax(probs) + 6  # 获得最大概率的layer
                fusion_num = torch.argmax(probs) #0-5
                # 统计概率
                # fusion_num_dict[str(int(fusion_depth))] += 1
                fusion_num_dict[str(int(fusion_num * 2 + 2))] += 1
            else:
                fusion_num = None

            logits = self.model(image.type(self.model.dtype), fusion_depth=fusion_num)
            loss = F.cross_entropy(logits, label)
            loss_l.append(loss.cpu().item())
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()
        loss_mean = np.mean(loss_l)

        if self.adaptive_fusion:
            for key, value in fusion_num_dict.items():
                fusion_num_dict[key] = value/len(data_loader)
                self.write_scalar(f"{split}/ratio/{key}",fusion_num_dict[key], self.epoch)

            print(f"ratio of every fusion depth: {fusion_num_dict}")

        self.write_scalar(f"{split}/loss", loss_mean, self.epoch)
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)