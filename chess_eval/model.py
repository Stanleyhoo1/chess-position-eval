"""Neural network for evaluating chess positions.

Architecture overview
---------------------
1. Board: each of the 64 squares holds a piece ID (0 = empty, 1-12 = pieces).
   A shared nn.Embedding(13, 16) maps every square to a 16-dim vector, giving a
   16-channel 8x8 "image". Two 3x3 convolutions (16 -> 32 -> 64 channels,
   padding 1, no pooling) keep the 8x8 geometry. The result is flattened to
   4096 features and reduced by two linear layers to a 128-dim board vector.
2. Metadata: side to move (1 float), castling rights (4 floats), and an
   nn.Embedding(17, 4) of the en-passant state (4 floats).
3. Head: the 128 + 1 + 4 + 4 = 137 features go through an MLP
   137 -> 64 -> 32 -> 1 producing one unbounded evaluation score.

Only the model and a smoke test live here. Data loading, training, losses and
optimizers are intentionally left for later.
"""

import torch
import torch.nn as nn


class ChessEvaluationModel(nn.Module):
    """CNN + MLP that maps an encoded chess position to a scalar evaluation."""

    # regular pieces
    NUM_PIECE_IDS = 13      # 0 = empty, 1..12 = six white + six black piece types
    PIECE_EMBED_DIM = 16
    # en passant
    NUM_EP_IDS = 17         # 0 = none, 1..16 = the sixteen squares on ranks 3 and 6
    EP_EMBED_DIM = 4

    def __init__(self):
        super().__init__()

        # --- Board branch -------------------------------------------------
        # piece embeddings
        self.piece_embedding = nn.Embedding(self.NUM_PIECE_IDS, self.PIECE_EMBED_DIM)

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Flatten CNN layers to one vector
        self.board_fc1 = nn.Linear(64 * 8 * 8, 256)   # 4096 -> 256
        self.board_fc2 = nn.Linear(256, 128)

        # --- Metadata branch ----------------------------------------------
        self.en_passant_embedding = nn.Embedding(self.NUM_EP_IDS, self.EP_EMBED_DIM)

        # --- Combined head ------------------------------------------------
        # 1 for player move, 4 for castle status, + embedding dimestions for en passant
        combined_dim = 128 + 1 + 4 + self.EP_EMBED_DIM   # = 137

        # Add the metadata then get predicted eval score
        self.head_fc1 = nn.Linear(combined_dim, 64)
        self.head_fc2 = nn.Linear(64, 32)
        self.head_out = nn.Linear(32, 1)

        # activation function for non-linearity
        self.relu = nn.ReLU()

    def forward(self, board, side_to_move, castling_rights, en_passant):
        """
        Args:
            board:           (batch, 8, 8) integer piece IDs in [0, 12]
            side_to_move:    (batch, 1)    1 = White to move, 0 = Black to move
            castling_rights: (batch, 4)    [W kingside, W queenside, B kingside, B queenside]
            en_passant:      (batch,)      integer en-passant ID in [0, 16]

        Returns:
            (batch, 1) unbounded evaluation score.
        """
        # Embedding layers need integer (long) indices.
        board = board.long()
        en_passant = en_passant.long()
        # Linear layers need floating-point inputs.
        side_to_move = side_to_move.float()
        castling_rights = castling_rights.float()

        # --- Board branch -------------------------------------------------
        x = self.piece_embedding(board)        # (batch, 8, 8, 16)
        x = x.permute(0, 3, 1, 2)              # (batch, 16, 8, 8)  channels-first for Conv2d

        x = self.relu(self.conv1(x))           # (batch, 32, 8, 8)
        x = self.relu(self.conv2(x))           # (batch, 64, 8, 8)

        x = x.flatten(start_dim=1)             # (batch, 4096)
        x = self.relu(self.board_fc1(x))       # (batch, 256)
        board_features = self.relu(self.board_fc2(x))   # (batch, 128)

        # --- Metadata branch ----------------------------------------------
        ep_features = self.en_passant_embedding(en_passant)   # (batch, 4)

        # --- Combine ------------------------------------------------------
        combined = torch.cat(
            [board_features, side_to_move, castling_rights, ep_features], dim=1
        )                                      # (batch, 128 + 1 + 4 + 4) = (batch, 137)

        h = self.relu(self.head_fc1(combined))  # (batch, 64)
        h = self.relu(self.head_fc2(h))         # (batch, 32)
        out = self.head_out(h)                  # (batch, 1), no final activation

        return out



class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and a skip connection; keeps 8x8."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)                # (batch, channels, 8, 8)


class ChessEvaluationResNet(nn.Module):
    """Larger evaluator: a residual convolutional trunk over the board.

    Differences from ChessEvaluationModel:
      * side to move, castling rights and the en-passant square are fed in as
        extra 8x8 planes next to the piece embedding, so the convolutions can
        see whose move it is instead of only the dense head seeing it;
      * a stack of residual blocks (default 6 x 128 channels, 13 convolutions
        deep) gives every square a view of the whole board before flattening;
      * a 1x1 convolution squeezes channels before the dense head so the
        memorization-prone fully connected layer stays small.

    Inputs and output have exactly the same shapes and meanings as
    ChessEvaluationModel, so training/train.py and chess_eval/load_model.py work unchanged.
    """

    NUM_PIECE_IDS = 13
    META_PLANES = 1 + 4 + 1          # side to move, 4 castling flags, en-passant square

    def __init__(self, blocks=6, channels=128, embed_dim=32, head_channels=32, head_hidden=256):
        super().__init__()
        self.cfg = dict(blocks=blocks, channels=channels, embed_dim=embed_dim,
                        head_channels=head_channels, head_hidden=head_hidden)

        self.piece_embedding = nn.Embedding(self.NUM_PIECE_IDS, embed_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim + self.META_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.head_conv = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
        )
        self.head_fc1 = nn.Linear(head_channels * 8 * 8, head_hidden)
        self.head_out = nn.Linear(head_hidden, 1)
        self.relu = nn.ReLU()

    def config(self):
        return dict(self.cfg)

    @staticmethod
    def _en_passant_plane(en_passant):
        """(batch,) en-passant IDs -> (batch, 1, 8, 8) plane with a 1 on the target square.
        IDs 1-8 are a3..h3 (board row 5, since row 0 is rank 8); 9-16 are a6..h6 (row 2)."""
        batch = en_passant.shape[0]
        plane = torch.zeros(batch, 64, device=en_passant.device)
        has_ep = en_passant > 0
        file_idx = (en_passant - 1) % 8
        row = torch.where(en_passant >= 9, 2, 5)
        square = row * 8 + file_idx
        plane[has_ep, square[has_ep]] = 1.0
        return plane.view(batch, 1, 8, 8)

    def forward(self, board, side_to_move, castling_rights, en_passant):
        board = board.long()
        en_passant = en_passant.long()
        side_to_move = side_to_move.float()
        castling_rights = castling_rights.float()
        batch = board.shape[0]

        x = self.piece_embedding(board)                      # (batch, 8, 8, embed_dim)
        x = x.permute(0, 3, 1, 2)                            # (batch, embed_dim, 8, 8)

        meta = torch.cat([side_to_move, castling_rights], dim=1)           # (batch, 5)
        meta_planes = meta.view(batch, 5, 1, 1).expand(batch, 5, 8, 8)     # (batch, 5, 8, 8)
        ep_plane = self._en_passant_plane(en_passant)                      # (batch, 1, 8, 8)
        x = torch.cat([x, meta_planes, ep_plane], dim=1)     # (batch, embed_dim + 6, 8, 8)

        x = self.stem(x)                                     # (batch, channels, 8, 8)
        x = self.blocks(x)                                   # (batch, channels, 8, 8)
        x = self.head_conv(x)                                # (batch, head_channels, 8, 8)
        x = x.flatten(start_dim=1)                           # (batch, head_channels * 64)
        x = self.relu(self.head_fc1(x))                      # (batch, head_hidden)
        return self.head_out(x)                              # (batch, 1), no final activation


MODEL_CLASSES = {
    "ChessEvaluationModel": ChessEvaluationModel,
    "ChessEvaluationResNet": ChessEvaluationResNet,
}


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = torch.randint(0, 13, (batch_size, 8, 8), device=device)        # piece IDs 0..12
    side_to_move = torch.randint(0, 2, (batch_size, 1), device=device)     # 0 or 1
    castling_rights = torch.randint(0, 2, (batch_size, 4), device=device)  # four binary flags
    en_passant = torch.randint(0, 17, (batch_size,), device=device)        # 0..16

    model = ChessEvaluationModel().to(device)
    output = model(board, side_to_move, castling_rights, en_passant)

    print("device:               ", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print("board shape:          ", tuple(board.shape))
    print("side_to_move shape:   ", tuple(side_to_move.shape))
    print("castling_rights shape:", tuple(castling_rights.shape))
    print("en_passant shape:     ", tuple(en_passant.shape))
    print("output shape:         ", tuple(output.shape))
    print("predicted outputs:    ", output.detach().cpu().squeeze(1).tolist())

    assert output.shape == (batch_size, 1), f"expected (4, 1), got {tuple(output.shape)}"
    print("OK: output shape is (4, 1)")
    # Same dummy batch through the larger residual model.
    big = ChessEvaluationResNet().to(device)
    big_out = big(board, side_to_move, castling_rights, en_passant)
    n_small = sum(p.numel() for p in model.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    print(f"ChessEvaluationModel params:  {n_small:,}")
    print(f"ChessEvaluationResNet params: {n_big:,}  output shape {tuple(big_out.shape)}")
    assert big_out.shape == (batch_size, 1)
    # en-passant plane: id 5 = e3 -> row 5, file 4; id 12 = d6 -> row 2, file 3
    plane = ChessEvaluationResNet._en_passant_plane(torch.tensor([0, 5, 12], device=device))
    assert plane[0].sum() == 0 and plane[1, 0, 5, 4] == 1 and plane[2, 0, 2, 3] == 1 and plane.sum() == 2
    print("OK: ResNet forward pass and en-passant plane")
