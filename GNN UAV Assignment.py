"""
================================================================================
GNN-BASED UAV TRAJECTORY DESIGN: Student Assignment
================================================================================

Course: Introduction to Graph Neural Networks for Robotics
Level: Undergraduate (Junior/Senior)
Estimated Time: 2-3 weeks

ASSIGNMENT OVERVIEW:
--------------------
You will implement a Graph Neural Network (GNN) to solve a multi-UAV target 
assignment problem. The infrastructure (environment, baseline, training loop,
evaluation) is provided. Your task is to implement the GNN architecture.

YOUR TASK:
----------
Complete the TODO sections in:
  1. AttentionLayer class (Section 3A)
  2. GNNEncoder class (Section 3B)  
  3. AssignmentNetwork class (Section 3C)

LEARNING OBJECTIVES:
--------------------
After completing this assignment, you will be able to:
  1. Implement attention mechanisms for graph-structured data
  2. Design encoder networks that process heterogeneous nodes (UAVs vs targets)
  3. Create scoring functions for combinatorial assignment problems
  4. Train neural networks via imitation learning

GRADING RUBRIC:
---------------
  - AttentionLayer implementation: 25 points
  - GNNEncoder implementation: 35 points
  - AssignmentNetwork implementation: 25 points
  - Code runs and produces reasonable results: 15 points

HINTS:
------
  - Start by understanding the input/output shapes at each step
  - Use print statements to debug tensor dimensions
  - The baseline achieves ~450 distance; your GNN should match this
  - If accuracy plateaus below 80%, check your masking logic

Author: [Course Instructor]
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==============================================================================
# SECTION 1: ENVIRONMENT (PROVIDED - DO NOT MODIFY)
# ==============================================================================

class UAVEnvironment:
    """
    Multi-UAV Target Assignment Environment
    
    Problem Setup:
    - N UAVs start at a central depot
    - M targets are scattered in a 2D grid
    - Each target must be visited exactly once by some UAV
    - Goal: Minimize total distance traveled by all UAVs
    
    State Representation:
    - UAV positions: [n_uavs, 2] - current (x, y) coordinates
    - Target positions: [n_targets, 2] - fixed (x, y) coordinates  
    - Visited mask: [n_targets] - boolean, True if target already assigned
    
    Action Space:
    - Each action is a (uav_idx, target_idx) pair
    - Assigns target_idx to uav_idx
    - UAV "moves" to that target (updates position)
    """
    
    def __init__(self, n_uavs: int = 3, n_targets: int = 15, grid_size: float = 100.0):
        self.n_uavs = n_uavs
        self.n_targets = n_targets
        self.grid_size = grid_size
        self.depot = np.array([grid_size / 2, grid_size / 2])
        
        # State variables (initialized in reset)
        self.target_positions = None
        self.uav_positions = None
        self.visited = None
        self.assignment_order = None
        
    def reset(self, seed: Optional[int] = None) -> 'UAVEnvironment':
        """Reset environment with random target positions."""
        if seed is not None:
            np.random.seed(seed)
            
        self.target_positions = np.random.uniform(0, self.grid_size, (self.n_targets, 2))
        self.uav_positions = np.tile(self.depot, (self.n_uavs, 1)).astype(float)
        self.visited = np.zeros(self.n_targets, dtype=bool)
        self.assignment_order = [[] for _ in range(self.n_uavs)]
        return self
    
    def get_unvisited_targets(self) -> List[int]:
        """Return indices of unvisited targets."""
        return np.where(~self.visited)[0].tolist()
    
    def assign_target(self, uav_idx: int, target_idx: int) -> float:
        """Assign a target to a UAV. Returns distance cost."""
        if self.visited[target_idx]:
            raise ValueError(f"Target {target_idx} already visited!")
            
        dist = np.linalg.norm(self.uav_positions[uav_idx] - self.target_positions[target_idx])
        self.uav_positions[uav_idx] = self.target_positions[target_idx].copy()
        self.visited[target_idx] = True
        self.assignment_order[uav_idx].append(target_idx)
        return dist
    
    def get_total_distance(self, include_return: bool = True) -> Tuple[float, np.ndarray]:
        """Calculate total and per-UAV distances (including return to depot)."""
        uav_distances = np.zeros(self.n_uavs)
        
        for uav_idx in range(self.n_uavs):
            path = self.assignment_order[uav_idx]
            if not path:
                continue
            
            # Depot -> first target
            uav_distances[uav_idx] += np.linalg.norm(self.depot - self.target_positions[path[0]])
            
            # Between consecutive targets
            for i in range(1, len(path)):
                uav_distances[uav_idx] += np.linalg.norm(
                    self.target_positions[path[i-1]] - self.target_positions[path[i]]
                )
            
            # Last target -> depot
            if include_return:
                uav_distances[uav_idx] += np.linalg.norm(self.target_positions[path[-1]] - self.depot)
        
        return np.sum(uav_distances), uav_distances
    
    def get_state_features(self) -> Dict[str, torch.Tensor]:
        """
        Get current state as tensors for the neural network.
        
        Returns:
            Dictionary containing:
            - 'uav_features': [n_uavs, 2] normalized UAV positions
            - 'target_features': [n_targets, 3] normalized target positions + visited flag
            - 'visited_mask': [n_targets] boolean mask (True = already visited)
        """
        uav_pos = torch.tensor(self.uav_positions / self.grid_size, dtype=torch.float32)
        target_pos = torch.tensor(self.target_positions / self.grid_size, dtype=torch.float32)
        visited = torch.tensor(self.visited, dtype=torch.float32).unsqueeze(-1)
        
        target_features = torch.cat([target_pos, visited], dim=-1)
        
        return {
            'uav_features': uav_pos,           # Shape: [n_uavs, 2]
            'target_features': target_features, # Shape: [n_targets, 3]
            'visited_mask': torch.tensor(self.visited, dtype=torch.bool)  # Shape: [n_targets]
        }
    
    def render(self, ax=None, title: str = "") -> plt.Axes:
        """Visualize the current state and UAV paths."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(*self.depot, s=400, c='green', marker='s', 
                   label='Depot', zorder=5, edgecolors='black', linewidth=2)
        
        colors = ['red' if not v else 'lightgray' for v in self.visited]
        ax.scatter(self.target_positions[:, 0], self.target_positions[:, 1],
                  s=150, c=colors, marker='o', zorder=3, edgecolors='black')
        
        uav_colors = plt.cm.Set1(np.linspace(0, 1, self.n_uavs))
        for uav_idx in range(self.n_uavs):
            path = self.assignment_order[uav_idx]
            if path:
                points = [self.depot] + [self.target_positions[t] for t in path]
                points = np.array(points)
                ax.plot(points[:, 0], points[:, 1], c=uav_colors[uav_idx], 
                       linewidth=2, alpha=0.7, label=f'UAV {uav_idx+1}')
                ax.plot([points[-1, 0], self.depot[0]], [points[-1, 1], self.depot[1]], 
                       c=uav_colors[uav_idx], linewidth=2, alpha=0.4, linestyle='--')
        
        ax.set_xlim(-5, self.grid_size + 5)
        ax.set_ylim(-5, self.grid_size + 5)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        return ax


# ==============================================================================
# SECTION 2: GREEDY BASELINE (PROVIDED - DO NOT MODIFY)
# ==============================================================================

class GreedyPolicy:
    """
    Greedy Nearest Neighbor Baseline
    
    Strategy: At each step, find the (UAV, target) pair with minimum distance
    and assign that target to that UAV.
    
    This is a simple but effective heuristic that serves as:
    1. A performance baseline for comparison
    2. An "expert" that provides training data for imitation learning
    """
    
    def __init__(self):
        self.name = "Greedy Nearest Neighbor"
    
    def get_expert_action(self, env: UAVEnvironment) -> Tuple[int, int]:
        """
        Get the greedy action: (uav_idx, target_idx) with minimum distance.
        """
        best_uav, best_target, best_dist = None, None, float('inf')
        unvisited = env.get_unvisited_targets()
        
        if not unvisited:
            return None, None
            
        for uav_idx in range(env.n_uavs):
            for target_idx in unvisited:
                dist = np.linalg.norm(
                    env.uav_positions[uav_idx] - env.target_positions[target_idx]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_uav = uav_idx
                    best_target = target_idx
        
        return best_uav, best_target


# ==============================================================================
# SECTION 3A: ATTENTION LAYER (TODO - IMPLEMENT THIS)
# ==============================================================================

class AttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer
    
    This layer computes attention between two sets of vectors (e.g., UAVs and targets).
    It allows information to flow between nodes based on learned attention weights.
    
    Args:
        embed_dim: Dimension of input embeddings (e.g., 64)
        n_heads: Number of attention heads (e.g., 4)
    
    Forward pass:
        Input:
            - query: [batch, n_query, embed_dim] (e.g., UAV embeddings)
            - key: [batch, n_key, embed_dim] (e.g., target embeddings)
            - value: [batch, n_key, embed_dim] (e.g., target embeddings)
            - mask: [batch, n_key] boolean mask (True = ignore this position)
        
        Output:
            - out: [batch, n_query, embed_dim] updated query representations
    
    Implementation Steps:
        1. Project query, key, value using linear layers
        2. Reshape for multi-head attention
        3. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
        4. Apply mask (set masked positions to -inf before softmax)
        5. Apply softmax to get attention weights
        6. Compute weighted sum of values
        7. Reshape and project output
    """
    
    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        
        # TODO: Define the following linear layers:
        # - self.q_proj: Linear layer for query projection (embed_dim -> embed_dim)
        # - self.k_proj: Linear layer for key projection (embed_dim -> embed_dim)
        # - self.v_proj: Linear layer for value projection (embed_dim -> embed_dim)
        # - self.out_proj: Linear layer for output projection (embed_dim -> embed_dim)
        
        # ============ YOUR CODE HERE ============
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # ========================================
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            query: [batch, n_query, embed_dim]
            key: [batch, n_key, embed_dim]
            value: [batch, n_key, embed_dim]
            mask: [batch, n_key] boolean, True means IGNORE this position
            
        Returns:
            out: [batch, n_query, embed_dim]
        """
        batch_size = query.size(0)
        
        # TODO: Implement multi-head attention
        # 
        # Step 1: Project Q, K, V using the linear layers
        #         Q = self.q_proj(query)  # [batch, n_query, embed_dim]
        #         K = self.k_proj(key)    # [batch, n_key, embed_dim]
        #         V = self.v_proj(value)  # [batch, n_key, embed_dim]
        #
        # Step 2: Reshape for multi-head attention
        #         Q: [batch, n_query, embed_dim] -> [batch, n_heads, n_query, head_dim]
        #         K: [batch, n_key, embed_dim] -> [batch, n_heads, n_key, head_dim]
        #         V: [batch, n_key, embed_dim] -> [batch, n_heads, n_key, head_dim]
        #         Hint: Use view() and transpose()
        #
        # Step 3: Compute attention scores
        #         scores = Q @ K^T / sqrt(head_dim)
        #         Shape: [batch, n_heads, n_query, n_key]
        #
        # Step 4: Apply mask (if provided)
        #         - Expand mask to [batch, 1, 1, n_key]
        #         - Use masked_fill(mask, float('-inf')) to ignore masked positions
        #
        # Step 5: Apply softmax over the last dimension (n_key)
        #         attn_weights = softmax(scores, dim=-1)
        #
        # Step 6: Compute weighted sum of values
        #         out = attn_weights @ V
        #         Shape: [batch, n_heads, n_query, head_dim]
        #
        # Step 7: Reshape back and apply output projection
        #         out: [batch, n_heads, n_query, head_dim] -> [batch, n_query, embed_dim]
        #         out = self.out_proj(out)
        
        # ============ YOUR CODE HERE ============
        # Step 1: Project Q, K, V using the linear layers
        Q= self.q_proj(query)  # [batch, n_query, embed_dim]
        K = self.k_proj(key)    # [batch, n_key, embed_dim]
        V = self.v_proj(value)  # [batch, n_key, embed_dim]

        # Step 2: Reshape for multi-head attention
        Q = Q.view(batch_size, n_query, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_key, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_key, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention scores
        scores = Q @ K.transpose(-1, -2) / np.sqrt(self.head_dim)

        # Step 4: Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # Step 5: Apply softmax over the last dimension (n_key)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Step 6: Compute weighted sum of values
        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)

        # Step 7: Reshape back and apply output projection
        out = self.out_proj(out)
        return out
        # ========================================


# ==============================================================================
# SECTION 3B: GNN ENCODER (TODO - IMPLEMENT THIS)
# ==============================================================================

class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder using Cross-Attention
    
    This encoder processes two types of nodes (UAVs and targets) and allows
    them to exchange information through cross-attention layers.
    
    Architecture:
        1. Embed UAV features (2D position) into embed_dim
        2. Embed target features (2D position + visited flag) into embed_dim
        3. Apply n_layers of cross-attention:
           - UAVs attend to targets (to understand which targets are good)
           - Targets attend to UAVs (to understand which UAV might visit them)
    
    Args:
        uav_dim: Input dimension for UAVs (default: 2 for x, y position)
        target_dim: Input dimension for targets (default: 3 for x, y, visited)
        embed_dim: Hidden dimension (default: 64)
        n_layers: Number of cross-attention layers (default: 2)
    """
    
    def __init__(self, uav_dim: int = 2, target_dim: int = 3, 
                 embed_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # TODO: Define the following components:
        #
        # 1. UAV embedding network (MLP: uav_dim -> embed_dim -> embed_dim)
        #    self.uav_embed = nn.Sequential(
        #        nn.Linear(uav_dim, embed_dim),
        #        nn.ReLU(),
        #        nn.Linear(embed_dim, embed_dim)
        #    )
        #
        # 2. Target embedding network (MLP: target_dim -> embed_dim -> embed_dim)
        #    self.target_embed = nn.Sequential(...)
        #
        # 3. Cross-attention layers (use nn.ModuleList for multiple layers)
        #    self.uav_to_target_attn = nn.ModuleList([AttentionLayer(embed_dim) for _ in range(n_layers)])
        #    self.target_to_uav_attn = nn.ModuleList([AttentionLayer(embed_dim) for _ in range(n_layers)])
        #
        # 4. Layer normalization for residual connections
        #    self.uav_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        #    self.target_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        
        # ============ YOUR CODE HERE ============
        raise NotImplementedError("TODO: Initialize GNN encoder components")
        # ========================================
        
    def forward(self, uav_features: torch.Tensor, target_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode UAVs and targets with bidirectional cross-attention.
        
        Args:
            uav_features: [batch, n_uavs, uav_dim] UAV positions
            target_features: [batch, n_targets, target_dim] target positions + visited
            mask: [batch, n_targets] visited mask (True = ignore)
            
        Returns:
            uav_h: [batch, n_uavs, embed_dim] UAV embeddings
            target_h: [batch, n_targets, embed_dim] target embeddings
        """
        # TODO: Implement the forward pass
        #
        # Step 1: Embed inputs
        #         uav_h = self.uav_embed(uav_features)       # [batch, n_uavs, embed_dim]
        #         target_h = self.target_embed(target_features)  # [batch, n_targets, embed_dim]
        #
        # Step 2: Apply cross-attention layers with residual connections
        #         for i in range(self.n_layers):
        #             # UAVs attend to targets (use mask to ignore visited targets)
        #             uav_update = self.uav_to_target_attn[i](uav_h, target_h, target_h, mask)
        #             uav_h = self.uav_norms[i](uav_h + uav_update)  # Residual + LayerNorm
        #
        #             # Targets attend to UAVs (no mask needed)
        #             target_update = self.target_to_uav_attn[i](target_h, uav_h, uav_h, None)
        #             target_h = self.target_norms[i](target_h + target_update)
        #
        # Step 3: Return both embeddings
        #         return uav_h, target_h
        
        # ============ YOUR CODE HERE ============
        raise NotImplementedError("TODO: Implement GNN encoder forward pass")
        # ========================================


# ==============================================================================
# SECTION 3C: ASSIGNMENT NETWORK (TODO - IMPLEMENT THIS)
# ==============================================================================

class AssignmentNetwork(nn.Module):
    """
    Complete Network for UAV-Target Assignment
    
    This network takes the current state (UAV positions, target positions, 
    visited mask) and outputs scores for each possible (UAV, target) assignment.
    
    Architecture:
        1. Use GNNEncoder to get UAV and target embeddings
        2. Compute pairwise scores between all UAVs and all targets
        3. Mask out already-visited targets
        4. Return flattened scores for action selection
    
    Args:
        n_uavs: Number of UAVs
        n_targets: Number of targets
        embed_dim: Embedding dimension
    """
    
    def __init__(self, n_uavs: int = 3, n_targets: int = 15, embed_dim: int = 64):
        super().__init__()
        self.n_uavs = n_uavs
        self.n_targets = n_targets
        
        # TODO: Define the following components:
        #
        # 1. GNN Encoder
        #    self.encoder = GNNEncoder(uav_dim=2, target_dim=3, embed_dim=embed_dim)
        #
        # 2. Scoring head: Takes concatenated [UAV_embedding, target_embedding] 
        #    and outputs a scalar score
        #    self.score_head = nn.Sequential(
        #        nn.Linear(2 * embed_dim, embed_dim),
        #        nn.ReLU(),
        #        nn.Linear(embed_dim, 1)
        #    )
        
        # ============ YOUR CODE HERE ============
        raise NotImplementedError("TODO: Initialize assignment network components")
        # ========================================
        
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute assignment scores for all (UAV, target) pairs.
        
        Args:
            state: Dictionary containing:
                - 'uav_features': [batch, n_uavs, 2] or [n_uavs, 2]
                - 'target_features': [batch, n_targets, 3] or [n_targets, 3]
                - 'visited_mask': [batch, n_targets] or [n_targets]
            
        Returns:
            scores: [batch, n_uavs * n_targets] flattened assignment scores
                    Higher score = better assignment
                    Visited targets have score = -inf
        """
        uav_feat = state['uav_features']
        target_feat = state['target_features']
        mask = state['visited_mask']
        
        # Add batch dimension if needed
        if uav_feat.dim() == 2:
            uav_feat = uav_feat.unsqueeze(0)
            target_feat = target_feat.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        batch_size = uav_feat.size(0)
        
        # TODO: Implement the forward pass
        #
        # Step 1: Encode using GNN
        #         uav_h, target_h = self.encoder(uav_feat, target_feat, mask)
        #         uav_h: [batch, n_uavs, embed_dim]
        #         target_h: [batch, n_targets, embed_dim]
        #
        # Step 2: Compute pairwise features
        #         We need to score every (UAV, target) pair.
        #         Expand dimensions to create all pairs:
        #         uav_expanded: [batch, n_uavs, 1, embed_dim] -> [batch, n_uavs, n_targets, embed_dim]
        #         target_expanded: [batch, 1, n_targets, embed_dim] -> [batch, n_uavs, n_targets, embed_dim]
        #         pair_features = concat([uav_expanded, target_expanded], dim=-1)
        #         Shape: [batch, n_uavs, n_targets, 2*embed_dim]
        #
        # Step 3: Score each pair
        #         scores = self.score_head(pair_features).squeeze(-1)
        #         Shape: [batch, n_uavs, n_targets]
        #
        # Step 4: Mask visited targets
        #         mask_expanded = mask.unsqueeze(1).expand(-1, self.n_uavs, -1)
        #         scores = scores.masked_fill(mask_expanded, float('-inf'))
        #
        # Step 5: Flatten for output
        #         scores_flat = scores.view(batch_size, -1)
        #         Shape: [batch, n_uavs * n_targets]
        #
        # return scores_flat
        
        # ============ YOUR CODE HERE ============
        raise NotImplementedError("TODO: Implement assignment network forward pass")
        # ========================================
    
    def get_action(self, state: Dict[str, torch.Tensor], 
                   deterministic: bool = True) -> Tuple[int, int]:
        """
        Select a (UAV, target) assignment based on scores.
        
        This method is provided - no need to modify.
        """
        with torch.no_grad():
            scores = self.forward(state)
            
            if deterministic:
                action_idx = scores.argmax(dim=-1).item()
            else:
                probs = F.softmax(scores, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            
            uav_idx = action_idx // self.n_targets
            target_idx = action_idx % self.n_targets
            
            return uav_idx, target_idx


# ==============================================================================
# SECTION 4: IMITATION LEARNING TRAINER (PROVIDED - DO NOT MODIFY)
# ==============================================================================

class ImitationLearningTrainer:
    """
    Train the GNN via imitation learning from the greedy expert.
    
    Process:
    1. Collect demonstrations: Run greedy policy, record (state, action) pairs
    2. Train network: Minimize cross-entropy between predicted and expert actions
    
    This is more stable than reinforcement learning because we have direct
    supervision from the expert's actions.
    """
    
    def __init__(self, model: AssignmentNetwork, lr: float = 1e-3):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.expert = GreedyPolicy()
        self.losses = []
        self.accuracies = []
        
    def collect_demonstrations(self, env: UAVEnvironment, n_episodes: int = 100) -> List[Tuple]:
        """Collect (state, expert_action) pairs by running greedy policy."""
        demonstrations = []
        
        for ep in range(n_episodes):
            env.reset(seed=SEED + ep)
            
            while not np.all(env.visited):
                state = env.get_state_features()
                uav_idx, target_idx = self.expert.get_expert_action(env)
                action_idx = uav_idx * env.n_targets + target_idx
                demonstrations.append((state, action_idx))
                env.assign_target(uav_idx, target_idx)
        
        return demonstrations
    
    def train_epoch(self, demonstrations: List[Tuple], batch_size: int = 64) -> Dict:
        """Train one epoch on collected demonstrations."""
        self.model.train()
        random.shuffle(demonstrations)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(demonstrations), batch_size):
            batch = demonstrations[i:i+batch_size]
            
            uav_feats = torch.stack([d[0]['uav_features'] for d in batch]).to(DEVICE)
            target_feats = torch.stack([d[0]['target_features'] for d in batch]).to(DEVICE)
            masks = torch.stack([d[0]['visited_mask'] for d in batch]).to(DEVICE)
            actions = torch.tensor([d[1] for d in batch], dtype=torch.long).to(DEVICE)
            
            state = {
                'uav_features': uav_feats,
                'target_features': target_feats,
                'visited_mask': masks
            }
            
            scores = self.model(state)
            loss = F.cross_entropy(scores, actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(batch)
            preds = scores.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += len(batch)
        
        avg_loss = total_loss / len(demonstrations)
        accuracy = correct / total
        
        self.losses.append(avg_loss)
        self.accuracies.append(accuracy)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, env: UAVEnvironment, n_epochs: int = 50, 
              n_demo_episodes: int = 200, print_every: int = 10):
        """Full training loop."""
        print("="*60)
        print("COLLECTING EXPERT DEMONSTRATIONS")
        print("="*60)
        
        demonstrations = self.collect_demonstrations(env, n_demo_episodes)
        print(f"Collected {len(demonstrations)} state-action pairs")
        
        print("\n" + "="*60)
        print("TRAINING VIA IMITATION LEARNING")
        print("="*60)
        
        for epoch in range(n_epochs):
            stats = self.train_epoch(demonstrations)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {stats['loss']:.4f} | "
                      f"Accuracy: {stats['accuracy']*100:.1f}%")
        
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)


# ==============================================================================
# SECTION 5: EVALUATION AND VISUALIZATION (PROVIDED - DO NOT MODIFY)
# ==============================================================================

def evaluate_policy(env: UAVEnvironment, policy, n_episodes: int = 50, 
                    policy_name: str = "Policy") -> Dict:
    """Evaluate a policy over multiple episodes."""
    distances = []
    
    for ep in range(n_episodes):
        env.reset(seed=SEED + 1000 + ep)
        
        while not np.all(env.visited):
            if isinstance(policy, GreedyPolicy):
                uav_idx, target_idx = policy.get_expert_action(env)
            else:
                state = env.get_state_features()
                state = {k: v.to(DEVICE) for k, v in state.items()}
                uav_idx, target_idx = policy.get_action(state, deterministic=True)
            
            if target_idx is None:
                break
            env.assign_target(uav_idx, target_idx)
        
        total_dist, _ = env.get_total_distance()
        distances.append(total_dist)
    
    return {
        'name': policy_name,
        'avg_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'distances': distances
    }


def compare_policies(env: UAVEnvironment, gnn_model: AssignmentNetwork,
                     greedy: GreedyPolicy, n_episodes: int = 50):
    """Compare GNN and greedy policies."""
    print("\n" + "="*60)
    print("EVALUATION: GNN vs Greedy Baseline")
    print("="*60)
    
    gnn_stats = evaluate_policy(env, gnn_model, n_episodes, "GNN")
    greedy_stats = evaluate_policy(env, greedy, n_episodes, "Greedy")
    
    improvement = (greedy_stats['avg_distance'] - gnn_stats['avg_distance']) / greedy_stats['avg_distance'] * 100
    
    print(f"\nResults over {n_episodes} episodes:")
    print("-"*60)
    print(f"{'Metric':<25} {'Greedy':>15} {'GNN':>15}")
    print("-"*60)
    print(f"{'Avg Total Distance':<25} {greedy_stats['avg_distance']:>15.2f} {gnn_stats['avg_distance']:>15.2f}")
    print(f"{'Std Total Distance':<25} {greedy_stats['std_distance']:>15.2f} {gnn_stats['std_distance']:>15.2f}")
    print("-"*60)
    
    # Clearer interpretation of the difference
    abs_diff = abs(improvement)
    if improvement > 0:
        print(f"Distance Difference: {abs_diff:.2f}% (GNN is better)")
    elif improvement < 0:
        print(f"Distance Difference: {abs_diff:.2f}% (GNN slightly worse)")
    else:
        print(f"Distance Difference: 0.00% (exactly equal)")
    
    if abs(improvement) < 5:
        print("\n✓ SUCCESS: GNN matches the greedy baseline (within 5%)!")
    elif improvement > 0:
        print(f"\n✓ EXCELLENT: GNN beats greedy by {improvement:.1f}%!")
    else:
        print(f"\n⚠ GNN is {-improvement:.1f}% worse (check your implementation)")
    
    print("="*60)
    return gnn_stats, greedy_stats, improvement


def plot_training_curves(trainer: ImitationLearningTrainer):
    """Plot training loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(trainer.losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([a*100 for a in trainer.accuracies], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Action Prediction Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: training_curves.png")
    plt.close()


def visualize_comparison(env: UAVEnvironment, gnn_model: AssignmentNetwork,
                         greedy: GreedyPolicy, seed: int = 999):
    """Create side-by-side trajectory visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    env.reset(seed=seed)
    while not np.all(env.visited):
        state = env.get_state_features()
        state = {k: v.to(DEVICE) for k, v in state.items()}
        uav_idx, target_idx = gnn_model.get_action(state, deterministic=True)
        env.assign_target(uav_idx, target_idx)
    gnn_dist, _ = env.get_total_distance()
    env.render(ax=axes[0], title=f"Your GNN Policy\nTotal Distance: {gnn_dist:.2f}")
    
    env.reset(seed=seed)
    while not np.all(env.visited):
        uav_idx, target_idx = greedy.get_expert_action(env)
        env.assign_target(uav_idx, target_idx)
    greedy_dist, _ = env.get_total_distance()
    env.render(ax=axes[1], title=f"Greedy Baseline\nTotal Distance: {greedy_dist:.2f}")
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: trajectory_comparison.png")
    plt.close()


# ==============================================================================
# SECTION 6: MAIN FUNCTION
# ==============================================================================

def main():
    """Main function to run the assignment."""
    print("\n" + "="*70)
    print("   GNN-BASED UAV TRAJECTORY DESIGN: Student Assignment")
    print("="*70)
    
    # Configuration
    N_UAVS = 3
    N_TARGETS = 15
    GRID_SIZE = 100.0
    N_TRAINING_EPOCHS = 100
    N_DEMO_EPISODES = 300
    N_EVAL_EPISODES = 50
    
    print(f"\nConfiguration:")
    print(f"  - UAVs: {N_UAVS}")
    print(f"  - Targets: {N_TARGETS}")
    print(f"  - Grid: {GRID_SIZE}x{GRID_SIZE}")
    
    # Create environment and policies
    env = UAVEnvironment(N_UAVS, N_TARGETS, GRID_SIZE)
    greedy = GreedyPolicy()
    
    # Create your GNN model
    print("\nInitializing GNN model...")
    try:
        gnn_model = AssignmentNetwork(N_UAVS, N_TARGETS, embed_dim=64)
        print("✓ Model initialized successfully!")
    except NotImplementedError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease implement the TODO sections in:")
        print("  - AttentionLayer (Section 3A)")
        print("  - GNNEncoder (Section 3B)")
        print("  - AssignmentNetwork (Section 3C)")
        return
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        env.reset(seed=0)
        state = env.get_state_features()
        state = {k: v.to(DEVICE) for k, v in state.items()}
        scores = gnn_model(state)
        print(f"✓ Forward pass successful! Output shape: {scores.shape}")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        print("Please check your implementation.")
        return
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    trainer = ImitationLearningTrainer(gnn_model, lr=1e-3)
    trainer.train(env, n_epochs=N_TRAINING_EPOCHS, 
                  n_demo_episodes=N_DEMO_EPISODES, print_every=20)
    
    # Evaluate
    gnn_stats, greedy_stats, improvement = compare_policies(
        env, gnn_model, greedy, N_EVAL_EPISODES
    )
    
    # Visualize
    print("\n[Generating Visualizations]")
    plot_training_curves(trainer)
    visualize_comparison(env, gnn_model, greedy)
    
    # Summary
    print("\n" + "="*70)
    print("   ASSIGNMENT SUMMARY")
    print("="*70)
    
    final_accuracy = trainer.accuracies[-1] * 100 if trainer.accuracies else 0
    
    print(f"\nYour Results:")
    print(f"  - Final Training Accuracy: {final_accuracy:.1f}%")
    print(f"  - Greedy Distance: {greedy_stats['avg_distance']:.2f}")
    print(f"  - Your GNN Distance: {gnn_stats['avg_distance']:.2f}")
    print(f"  - Improvement: {improvement:+.2f}%")
    
    print(f"\nGrading Criteria:")
    if final_accuracy >= 90:
        print("  ✓ Accuracy >= 90% - EXCELLENT")
    elif final_accuracy >= 80:
        print("  ~ Accuracy 80-90% - GOOD")
    else:
        print("  ✗ Accuracy < 80% - NEEDS WORK")
    
    if abs(improvement) < 5:
        print("  ✓ GNN matches baseline - PASS")
    elif improvement > 0:
        print("  ✓ GNN beats baseline - BONUS POINTS!")
    else:
        print("  ✗ GNN worse than baseline - CHECK IMPLEMENTATION")
    
    print("\nFiles generated:")
    print("  - training_curves.png")
    print("  - trajectory_comparison.png")
    print("="*70)
    
    return env, gnn_model, greedy, trainer


if __name__ == "__main__":
    main()
