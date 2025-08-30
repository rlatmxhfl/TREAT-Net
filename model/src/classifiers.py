import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


def get_echo_prime_encoder(device, disable_pe=False):
    checkpoint = torch.load(
        "/mnt/rcl-server/workspace/diane/echo_prime_encoder.pt", map_location='cpu', weights_only=True)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    # set output dim of mvit_v2_s to 512 since EchoPrime uses 512
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    echo_encoder.eval()

    # for param in echo_encoder.head.parameters():
    #     param.requires_grad = True

    if disable_pe:
        echo_encoder.use_abs_pos = False
        print("Positional encoding is disabled.")

    echo_encoder.to(device)

    return echo_encoder


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x: [B, V, D] where V = # videos
        """
        attn_scores = self.attn(x)  # [B, V, 1]
        weights = torch.softmax(attn_scores, dim=1)  # [B, V, 1]
        fused = torch.sum(weights * x, dim=1)  # [B, D]
        return fused, weights.squeeze(-1)

def get_sinusoid_encoding(n_position, d_model):
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                         -(torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(n_position, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [n_position, d_model]


class StudyClassifier(nn.Module):
    def __init__(self, num_classes=2, emb_dim=512, num_views=3, num_layers=2, nhead=8,
                 tab_emb_dim=192):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.task_classifier = nn.Linear(emb_dim * 2, 1)

        self.tabular_proj = nn.Linear(tab_emb_dim, emb_dim)
        self.view_classifier = nn.Linear(emb_dim, num_views)

    def forward(self, x, tabular_feat=None, view=None, training=False, head="task"):
        cls_token = self.cls_token.expand(len(x), 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        pe = get_sinusoid_encoding(x.size(1), x.size(2)).to(x.device)  # [seq+1, D]
        x = x.transpose(1, 0)  # [seq+1, B, D]
        x = x + pe.unsqueeze(1)  # [seq+1, B, D]

        x = self.transformer(x)
        cls_repr = x[0]  # [B, D]

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat)  # [B, D]
            cls_repr = ((cls_repr - cls_repr.mean(axis=1, keepdims=True)) /
                        cls_repr.std(axis=1, keepdims=True))
            tabular_emb = ((tabular_emb - tabular_emb.mean(axis=1, keepdims=True)) /
                           tabular_emb.std(axis=1, keepdims=True))
            cls_repr = torch.cat([cls_repr, tabular_emb], dim=1)  # [B, 2D]

        if head == "view":
            return self.view_classifier(cls_repr[:, :self.cls_token.shape[-1]])  # ignore tab features

        return self.task_classifier(cls_repr)

    def extract_cls_embedding(self, x, tabular_feat=None, view_mask=None):
        # x: [B, L, D]
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)  # [B, L+1, D]
        x = x.transpose(1, 0)  # [L+1, B, D]

        if view_mask is not None:
            src_padding_mask = (view_mask == -1)  # [B, L]
            cls_pad = torch.zeros((view_mask.size(0), 1), dtype=torch.bool, device=x.device)
            src_padding_mask = torch.cat([cls_pad, src_padding_mask], dim=1)  # [B, L+1]
        else:
            src_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_padding_mask)  # [L+1, B, D]
        cls_repr = x[0]  # [B, D]

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat)  # [B, D]
            cls_repr = torch.cat([cls_repr, tabular_emb], dim=1)  # [B, 2D]

        return cls_repr  # [B, D] or [B, 2D]

    def freeze_pretrain_head(self):
        for p in self.view_classifier.parameters():
            p.requires_grad = False

    def unfreeze_pretrain_head(self):
        for p in self.view_classifier.parameters():
            p.requires_grad = True

    def freeze_task_head(self):
        for p in self.task_classifier.parameters():
            p.requires_grad = False

    def unfreeze_task_head(self):
        for p in self.task_classifier.parameters():
            p.requires_grad = True

    def freeze_encoder(self):
        # Optionally freeze layers
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        # Optionally freeze layers
        for param in self.transformer.parameters():
            param.requires_grad = True

    @staticmethod
    def augment_embeddings(x, training=False):
        if not training:
            return x
        x = x + torch.randn_like(x) * 0.05
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > 0.3).float()
        scale = torch.empty(x.size(0), x.size(1), 1, device=x.device).uniform_(0.9, 1.1)
        x = x * mask * scale
        return x


class StudyClassifierVideoOnly(StudyClassifier):
    def __init__(self, num_classes=3, emb_dim=512, num_views=3, num_layers=2, nhead=8, tab_emb_dim=192):
        super().__init__(num_classes, emb_dim, num_views, num_layers, nhead, tab_emb_dim)

        # Replace unused components with identity
        self.tabular_proj = nn.Identity()

        # Adjust task classifier to take just video cls embedding (not concat with tabular_emb)
        self.task_classifier = nn.Linear(emb_dim, 1)  # or num_classes for multiclass

    def forward(self, x, tabular_feat=None, view=None, training=False, head="task"):
        cls_token = self.cls_token.expand(len(x), 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        pe = get_sinusoid_encoding(x.size(1), x.size(2)).to(x.device)
        x = x.transpose(1, 0)
        x = x + pe.unsqueeze(1)

        x = self.transformer(x)
        cls_repr = x[0]  # [B, D]

        if head == "view":
            return self.view_classifier(cls_repr)

        return self.task_classifier(cls_repr)  # [B, 1]


class StudyClassifierV1(StudyClassifier):
    def __init__(self, num_classes=1, emb_dim=512, num_views=3, num_layers=2,
                 nhead=8, tab_emb_dim=192):
        super().__init__()
        self.transformer = nn.Identity()

        # Class token (optional — here not used in forward)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Tabular projection: map tabular features into embedding space
        self.tabular_proj = nn.Linear(tab_emb_dim, emb_dim)

        # Cross-attention: tabular attends to video
        self.cross_attn = MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, batch_first=True)

        # FFN after cross-attn
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(emb_dim, emb_dim),
        )

        # Task classifier head (binary classification)
        self.task_classifier = nn.Linear(emb_dim, 1)

        # Optional auxiliary view classifier (if used)
        self.view_classifier = nn.Linear(emb_dim, num_views)

    def forward(self, x, tabular_feat=None, view=None, training=False, head="task"):
        """
        x: [B, L, D] — video tokens
        tabular_feat: [B, F] — raw tabular input
        """
        B, L, D = x.shape

        if tabular_feat is not None:
            # Project tabular features to embedding space
            tabular_emb = self.tabular_proj(tabular_feat).unsqueeze(1)  # [B, 1, D]

            # Cross-attention: tabular → video
            attn_out, _ = self.cross_attn(query=tabular_emb, key=x, value=x)  # [B, 1, D]
            fused = attn_out + tabular_emb
            fused = self.cross_ffn(fused).squeeze(1)  # [B, D]
            cls_repr = fused
        else:
            # If no tabular input, just average video tokens or use cls token
            cls_repr = x.mean(dim=1)  # [B, D] — fallback

        if head == "view":
            return self.view_classifier(cls_repr)
        return self.task_classifier(cls_repr)


class StudyClassifierV1LateFusion(StudyClassifierV1):
    def __init__(self, num_classes=1, emb_dim=512, num_views=3, num_layers=2,
                 nhead=8, tab_emb_dim=192):
        super().__init__(num_classes=num_classes, emb_dim=emb_dim, num_views=num_views,
                         num_layers=num_layers, nhead=nhead, tab_emb_dim=tab_emb_dim)

        self.tab_classifier = nn.Sequential(
            nn.Linear(tab_emb_dim, 192, bias=False),
            nn.ReLU(),
            nn.Linear(192, 1, bias=False))

        self.layer_norm = nn.LayerNorm(2)  # normalize fused [task_logit, tab_logit]

    def forward(self, x, tabular_feat=None, view=None, training=False, head="task"):
        B, L, D = x.shape

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat).unsqueeze(1)  # [B, 1, D]
            attn_out, _ = self.cross_attn(query=tabular_emb, key=x, value=x)  # [B, 1, D]
            fused = attn_out + tabular_emb
            fused = self.cross_ffn(fused).squeeze(1)  # [B, D]
            cls_repr = fused
        else:
            cls_repr = x.mean(dim=1)  # fallback if no tabular input

        if head == "view":
            return self.view_classifier(cls_repr)

        # --- Late fusion of logits ---
        task_logit = self.task_classifier(cls_repr)  # [B, 1]
        tab_logit = self.tab_classifier(tabular_feat)  # [B, 1]

        fused = torch.cat([task_logit, tab_logit], dim=1)  # [B, 2]
        fused = self.layer_norm(fused)  # normalize
        return fused.mean(dim=1)


class StudyClassifierMultiTask(StudyClassifier):
    def __init__(self, num_classes=5, emb_dim=512, tab_emb_dim=192, nhead=8, *args, **kwargs):
        super().__init__(
            num_classes=num_classes,
            emb_dim=emb_dim,
            num_views=0,  # not used
            num_layers=0,  # not used
            nhead=nhead,
            tab_emb_dim=tab_emb_dim,
        )

        # Replace task head with multitask heads
        self.cad_head = nn.Linear(emb_dim, num_classes)  # e.g., 3-class softmax
        self.treatment_head = nn.Linear(emb_dim, 1)  # binary (logits)

        # Drop view classifier
        # del self.view_classifier
        # del self.task_classifier
        self.view_classifier = nn.Identity()
        self.task_classifier = nn.Identity()

    def forward(self, x, tabular_feat=None, view=None, head="all", *args, **kwargs):
        """
        x: [B, L, D] — video tokens
        tabular_feat: [B, F] — raw tabular input
        head: "all", "cad", "treatment"
        """
        B, L, D = x.shape

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat).unsqueeze(1)  # [B, 1, D]
            attn_out, _ = self.cross_attn(query=tabular_emb, key=x, value=x)  # [B, 1, D]
            fused = attn_out + tabular_emb
            fused = self.cross_ffn(fused).squeeze(1)  # [B, D]
            cls_repr = fused
        else:
            cls_repr = x.mean(dim=1)  # fallback

        if head == "all":
            return self.treatment_head(cls_repr), self.cad_head(cls_repr).squeeze(-1)
        elif head == "cad":
            return self.cad_head(cls_repr).squeeze(-1)
        elif head == "treatment":
            return self.treatment_head(cls_repr)
        else:
            raise ValueError(f"Invalid head: {head}")


class StudyClassifierFromTokens(StudyClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        emb_dim = kwargs['emb_dim']
        num_classes = kwargs['num_classes']
        self.task_classifier = nn.Linear(emb_dim, num_classes)
        self.freeze_unused()

    def forward(self, x, *args, **kwargs):
        if 'training' in kwargs:
            x = self.augment_embeddings(x, training=kwargs['training'])
        return self.task_classifier(x)

    def freeze_unused(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only task_classifier
        for param in self.task_classifier.parameters():
            param.requires_grad = True

    @staticmethod
    def augment_embeddings(x, training=False):
        if not training:
            return x
        x = x + torch.randn_like(x) * 0.05
        mask = (torch.rand(x.size(0), 1, device=x.device) > 0.3).float()
        scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.9, 1.1)
        x = x * mask * scale
        return x


class StudyClassifierFromTokensInterView(StudyClassifier):
    def __init__(self, num_classes=3, num_views=3, embed_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views

        # Transformer encoder for 3 views
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.task_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: [B, 1536] (concatenated 3 views of [512])
            training: whether to apply augmentation
        Returns:
            logits: [B, num_classes]
        """
        training = False
        if 'training' in kwargs:
            training = kwargs['training']

        B = x.size(0)

        # Reshape to [B, 3, 512]
        x = x.view(B, self.num_views, self.embed_dim)

        # Optional token-level augmentation
        # x = self.augment_embeddings(x, training=training)

        # Transformer expects [L, B, D]
        x = x.transpose(0, 1)  # [3, B, 512]

        out = self.encoder(x)  # [3, B, 512]
        cls_token = out.mean(dim=0)  # [B, 512]

        return self.task_classifier(cls_token)

    @staticmethod
    def augment_embeddings(x, training=False, *args, **kwargs):
        """
        Token-level augmentation: add noise + random masking + scaling.
        x: [B, 3, 512]
        """
        if not training:
            return x
        x = x + torch.randn_like(x) * 0.05
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > 0.3).float()  # [B, 3, 1]
        scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)  # [B, 1, 1]
        return x * mask * scale


class StudyClassifierAttention(StudyClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_classifier = nn.Linear(kwargs['emb_dim'], kwargs['num_classes'])

    def forward(self, x, tabular_feat=None, view=None, training=False, head="task"):
        cls_token = self.cls_token.expand(len(x), -1, -1)  # [B, 1, D]

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat)  # [B, 1, D]
            x = ((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True))
            tabular_emb = ((tabular_emb - tabular_emb.mean(axis=1, keepdims=True)) /
                           tabular_emb.std(axis=1, keepdims=True)).unsqueeze(1)
            x = torch.cat([cls_token, tabular_emb, x], dim=1)  # [B, 1 + 1 + L, D]
        else:
            x = torch.cat([cls_token, x], dim=1)  # [B, 1 + L, D]

        x = x.transpose(1, 0)  # [seq_len, B, D]
        x = self.transformer(x)
        cls_repr = x[0]  # [B, D]

        if head == "view":
            return self.view_classifier(cls_repr)
        return self.task_classifier(cls_repr)


# class StudyClassifierMultiTaskSharedClsToken(StudyClassifier):
class StudyClassifierMultiTaskV0(StudyClassifier):
    def __init__(self, emb_dim=512, num_views=3, num_layers=2, nhead=8, tab_emb_dim=192, num_cad_classes=5):
        # Call parent constructor to set up transformer, tabular_proj, cls_token, etc.
        super().__init__(num_classes=1, emb_dim=emb_dim, num_views=num_views,
                         num_layers=num_layers, nhead=nhead, tab_emb_dim=tab_emb_dim)

        self.task_classifier = nn.Identity()
        # Replace the single-task head with multi-task heads
        self.tp_head = nn.Linear(emb_dim * 2, 1)  # Binary → BCEWithLogitsLoss
        # self.cad_head = nn.Linear(emb_dim * 2, 1)  # Binary → BCEWithLogitsLoss
        self.cad_head = nn.Linear(emb_dim * 2, num_cad_classes)  # K-class → CrossEntropyLoss

        # self.tp_head = nn.Sequential(
        #     nn.Linear(emb_dim * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )
        #
        # self.cad_head = nn.Sequential(
        #     nn.Linear(emb_dim * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )

    def forward(self, x, tabular_feat=None, view=None, training=False, *args, **kwargs):
        cls_token = self.cls_token.expand(len(x), 1, -1)
        x = torch.cat([cls_token, x], dim=1).transpose(1, 0)
        x = self.transformer(x)
        cls_repr = x[0]

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat)
            cls_repr = ((cls_repr - cls_repr.mean(dim=1, keepdim=True)) / cls_repr.std(dim=1, keepdim=True))
            tabular_emb = ((tabular_emb - tabular_emb.mean(dim=1, keepdim=True)) / tabular_emb.std(dim=1, keepdim=True))
            cls_repr = torch.cat([cls_repr, tabular_emb], dim=1)

        return self.tp_head(cls_repr), self.cad_head(cls_repr)


class StudyClassifierMultiTaskV1(StudyClassifier):
    def __init__(self, emb_dim=512, num_views=3, num_layers=2, nhead=8, tab_emb_dim=192):
        super().__init__(num_classes=1, emb_dim=emb_dim, num_views=num_views,
                         num_layers=num_layers, nhead=nhead, tab_emb_dim=tab_emb_dim)

        # Disable the original task classifier
        self.task_classifier = nn.Identity()
        self.cls_token.requires_grad = False

        # Separate task-specific classification heads
        self.tp_head = nn.Linear(emb_dim * 2, 1)  # For treatment prediction (BCE)
        self.cad_head = nn.Linear(emb_dim * 2, 5)  # For CAD prediction (CE)

        # Task-specific cls_tokens
        self.cls_token_tp = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.cls_token_cad = nn.Parameter(torch.zeros(1, 1, emb_dim))

    def forward(self, x, tabular_feat=None, view=None, training=False, *args, **kwargs):
        # Task-specific CLS tokens
        cls_tp = self.cls_token_tp.expand(x.size(0), 1, -1)
        cls_cad = self.cls_token_cad.expand(x.size(0), 1, -1)

        # Concatenate both tokens for shared transformer pass
        x_with_tp = torch.cat([cls_tp, x], dim=1)
        x_with_cad = torch.cat([cls_cad, x], dim=1)

        # Shared transformer encoder
        x_tp = self.transformer(x_with_tp.transpose(1, 0))  # [seq_len+1, B, D]
        x_cad = self.transformer(x_with_cad.transpose(1, 0))  # [seq_len+1, B, D]

        cls_repr_tp = x_tp[0]  # [B, D]
        cls_repr_cad = x_cad[0]  # [B, D]

        if tabular_feat is not None:
            tabular_emb = self.tabular_proj(tabular_feat)

            # Normalize each
            cls_repr_tp = (cls_repr_tp - cls_repr_tp.mean(dim=1, keepdim=True)) / cls_repr_tp.std(dim=1, keepdim=True)
            cls_repr_cad = (cls_repr_cad - cls_repr_cad.mean(dim=1, keepdim=True)) / cls_repr_cad.std(dim=1,
                                                                                                      keepdim=True)
            tabular_emb = (tabular_emb - tabular_emb.mean(dim=1, keepdim=True)) / tabular_emb.std(dim=1, keepdim=True)

            cls_repr_tp = torch.cat([cls_repr_tp, tabular_emb], dim=1)
            cls_repr_cad = torch.cat([cls_repr_cad, tabular_emb], dim=1)

        return self.tp_head(cls_repr_tp), self.cad_head(cls_repr_cad)