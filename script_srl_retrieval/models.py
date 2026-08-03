import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv

class SRLAttentionEncoder(nn.Module):
    def __init__(self, hidden_dim, device='cpu'):
        super().__init__()

        # 固定 role 顺序
        self.roles = [
            "agent", "negation", "relation",
            "patient", "recipient", "time", "location"
        ]
        self.role2id = {r: i for i, r in enumerate(self.roles)}
        self.device = device

        self.hidden_dim = hidden_dim
        self.num_roles = len(self.roles)

        # role embedding
        self.role_emb = nn.Embedding(self.num_roles, hidden_dim, device=self.device)

        # null embedding（处理缺失 argument）
        self.null_emb = nn.Embedding(self.num_roles, hidden_dim, device=self.device)

        # attention scoring
        self.attn = nn.Linear(hidden_dim, 1, device=self.device)

    def forward(self, srl_dict):
        """
        srl_dict: dict
            e.g.
            {
                'relation': Tensor(d),
                'patient': Tensor(d),
                ...
            }

        return:
            srl_emb: Tensor(d)
        """

        role_embs = []
        mask = []

        for role in self.roles:
            role_id = self.role2id[role]

            if role in srl_dict and srl_dict[role] is not None:
                # print(role, "exist")
                emb = srl_dict[role]  # [d]
                mask.append(1)
            else:
                # print(role, "not exist")
                emb = self.null_emb(torch.tensor(role_id, device=self.device))
                mask.append(0)

            # 加 role embedding
            emb = emb + self.role_emb(torch.tensor(role_id, device=self.device))
            role_embs.append(emb)

        role_embs = torch.stack(role_embs)  # [num_roles, d]
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)  # [num_roles]

        # attention score
        scores = self.attn(role_embs).squeeze(-1)  # [num_roles]

        # mask掉不存在的argument（-inf trick）
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=0)  # [num_roles]

        # 加权求和
        srl_emb = torch.sum(attn_weights.unsqueeze(-1) * role_embs, dim=0)

        return srl_emb
    

class DocAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, docs):
        doc_embeddings = []
        
        for sentences in docs:
            if len(sentences) == 0:
                doc_embeddings.append(torch.zeros(self.score.in_features))
                continue
            
            x = torch.stack(sentences)  # [num_sent, dim]
            
            attn = self.score(x)  # [num_sent, 1]
            attn = F.softmax(attn, dim=0)
            
            doc_emb = (attn * x).sum(dim=0)
            doc_embeddings.append(doc_emb)
        
        return torch.stack(doc_embeddings)
    


class HGTRankingModel(nn.Module):
    def __init__(
        self,
        metadata,              # (node_types, edge_types)
        in_dim_dict,           # dict: {node_type: input_dim}
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        device='cpu'):
        super().__init__()

        self.device = device
        self.node_types, self.edge_types = metadata

        # 🔹 不同类型节点先映射到统一空间
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim_dict[node_type], hidden_dim)
            for node_type in self.node_types
        }).to(self.device)

        # 🔹 HGT layers
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads
            )
            for _ in range(num_layers)
        ]).to(self.device)

        # 🔹 Ranking head（sentence + SRL 共用 or 分开都可以）
        self.sent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.srl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

    def forward(self, x_dict, edge_index_dict):
        # 🔹 Step 1: input projection
        x_dict = {
            k: F.relu(self.input_proj[k](v))
            for k, v in x_dict.items()
        }

        # 🔹 Step 2: HGT message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # 🔹 Step 3: scoring
        sent_scores = self.sent_head(x_dict["sentence"]).squeeze(-1)
        srl_scores = self.srl_head(x_dict["srl"]).squeeze(-1)

        return sent_scores, srl_scores
    


def bpr_loss_with_sampling(scores, labels, num_neg=5):
    """
    scores: [N]
    labels: [N]
    """

    N = scores.size(0)

    # 🔹 排序（重要：确定正负关系）
    sorted_idx = torch.argsort(labels, descending=True)

    loss = 0.0
    count = 0

    for rank, i in enumerate(sorted_idx):
        # 当前是正样本
        pos_score = scores[i]

        # 🔹 从比它差的样本里采样负样本
        neg_candidates = sorted_idx[rank+1:]

        if len(neg_candidates) == 0:
            continue

        # 随机采样
        neg_idx = neg_candidates[
            torch.randint(0, len(neg_candidates), (num_neg,))
        ]

        neg_score = scores[neg_idx]

        # 🔹 weighted BPR loss
        weight = (labels[i] - labels[neg_idx]).detach()
        loss += (-torch.log(torch.sigmoid(pos_score - neg_score)) * weight).mean()
        count += 1

    return loss / (count + 1e-8)

def sent_srl_consistency(sent_scores, srl_scores, sent2srl):
    """
    sent2srl: dict {sent_id: [srl_ids]}
    """
    loss = 0
    count = 0

    for s, srl_list in sent2srl.items():
        if len(srl_list) == 0:
            continue

        srl_mean = srl_scores[srl_list].mean()
        loss += (sent_scores[s] - srl_mean) ** 2
        count += 1

    return loss / (count + 1e-8)

