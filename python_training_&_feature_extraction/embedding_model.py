"""
embedding_model.py
==================
The core neural network architecture for few-shot wake word detection.

This file is imported by:
  - train_embedding.py       (training phase)
  - register_wake_word.py    (registration phase)

Architecture:
  Input:  (batch, 1, 40, 196) MFCC
  Output: (batch, 128) L2-normalized embedding vector

Key idea:
  Same word spoken by different people  → embeddings are CLOSE
  Different words                        → embeddings are FAR APART
  This is learned via episodic Prototypical-Network training (see
  train_embedding.py: prototypical_loss()), NOT the TripletLoss class
  below — that class is currently unused legacy code, kept in case you
  want to experiment with triplet-based training later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordEmbedder(nn.Module):
    """
    Lightweight CNN that converts a 2-second MFCC into a 128-dim embedding.

    Architecture mirrors your existing WakeWordCNN so the same
    mfcc.cpp / cnn_inference.cpp structure still works — only the
    final layer changes from sigmoid → L2-normalized linear projection.
    """

    def __init__(self, embedding_dim=128):
        super(WakeWordEmbedder, self).__init__()

        # ── Feature Extractor (same as your existing CNN) ──────────────
        self.features = nn.Sequential(
            # Block 1: (1, 40, 196) → (16, 20, 98)
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: (16, 20, 98) → (32, 10, 49)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: (32, 10, 49) → (64, 5, 24)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # ── Pooling: collapse spatial dims → (64,) ─────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Projection: 64 → embedding_dim ─────────────────────────────
        # NO sigmoid here — we want raw values that we then L2-normalize
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        x: (batch, 1, 40, 196)  — one channel, 40 MFCC coeffs, 196 frames
        returns: (batch, 128)   — L2-normalized embedding
        """
        x = self.features(x)           # (batch, 64, 5, 24)
        x = self.global_pool(x)        # (batch, 64, 1, 1)
        x = self.projector(x)          # (batch, 128)

        # L2 normalization — CRITICAL
        # After this, all embeddings live on a unit hypersphere.
        # Distance between embeddings = cosine distance (simple dot product)
        x = F.normalize(x, p=2, dim=1)
        return x


class TripletLoss(nn.Module):
    """
    NOTE: currently unused. train_embedding.py trains with
    prototypical_loss() (episodic Prototypical Networks), not this class.
    Kept here as a legacy/optional alternative — delete if you don't plan
    to use it, to avoid confusing future readers about what's actually
    driving training.

    Triplet Loss — teaches the model that:
      distance(anchor, positive) + margin < distance(anchor, negative)

    anchor   = "Hey Saya" spoken by person A
    positive = "Hey Saya" spoken by person B  (same word, different person)
    negative = "dog" spoken by person A        (different word)

    margin: how far apart positives vs negatives must be.
            0.3 is a good starting point for audio.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Since embeddings are L2-normalized, we use squared Euclidean distance
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        # relu clamps to 0 when the triplet is already satisfied
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()