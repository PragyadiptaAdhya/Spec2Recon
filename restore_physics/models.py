from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MLP2d(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class WindowAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.proj(out)
        return self.proj_drop(out)


class SwinBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = LayerNorm2d(channels)
        self.attn = WindowAttention2d(channels, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = MLP2d(channels, mlp_ratio=mlp_ratio, drop=drop)

    def _window_partition(self, x: torch.Tensor, window: int) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h // window, window, w // window, window)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, window * window, c)

    def _window_reverse(self, windows: torch.Tensor, window: int, h: int, w: int, b: int) -> torch.Tensor:
        x = windows.view(b, h // window, w // window, window, window, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        b, c, h, w = x.shape
        window = min(self.window_size, h, w)
        if window < 1:
            return shortcut
        pad_h = (window - (h % window)) % window
        pad_w = (window - (w % window)) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape

        if self.shift_size > 0 and self.shift_size < window:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        windows = self._window_partition(x, window)
        windows = self.attn(windows)
        x = self._window_reverse(windows, window, hp, wp, b)

        if self.shift_size > 0 and self.shift_size < window:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        if pad_h or pad_w:
            x = x[:, :, :h, :w]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinBlock2d(
                    channels=channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Downsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Upsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class SwinUNetDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 64,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (2, 4, 8, 16),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths and num_heads must each have four values")

        c1 = embed_dim
        c2 = embed_dim * 2
        c3 = embed_dim * 4
        c4 = embed_dim * 8

        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.enc1 = SwinStage(c1, depths[0], num_heads[0], window_size, mlp_ratio, drop_rate, attn_drop_rate)
        self.down1 = Downsample2d(c1, c2)
        self.enc2 = SwinStage(c2, depths[1], num_heads[1], window_size, mlp_ratio, drop_rate, attn_drop_rate)
        self.down2 = Downsample2d(c2, c3)
        self.enc3 = SwinStage(c3, depths[2], num_heads[2], window_size, mlp_ratio, drop_rate, attn_drop_rate)
        self.down3 = Downsample2d(c3, c4)
        self.bottleneck = SwinStage(c4, depths[3], num_heads[3], window_size, mlp_ratio, drop_rate, attn_drop_rate)

        self.up3 = Upsample2d(c4, c3)
        self.dec3_merge = nn.Conv2d(c3 + c3, c3, kernel_size=1)
        self.dec3 = SwinStage(c3, depths[2], num_heads[2], window_size, mlp_ratio, drop_rate, attn_drop_rate)

        self.up2 = Upsample2d(c3, c2)
        self.dec2_merge = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.dec2 = SwinStage(c2, depths[1], num_heads[1], window_size, mlp_ratio, drop_rate, attn_drop_rate)

        self.up1 = Upsample2d(c2, c1)
        self.dec1_merge = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1 = SwinStage(c1, depths[0], num_heads[0], window_size, mlp_ratio, drop_rate, attn_drop_rate)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.bottleneck(self.down3(s3))

        u3 = self.up3(b)
        u3 = self.dec3(self.dec3_merge(torch.cat([u3, s3], dim=1)))

        u2 = self.up2(u3)
        u2 = self.dec2(self.dec2_merge(torch.cat([u2, s2], dim=1)))

        u1 = self.up1(u2)
        u1 = self.dec1(self.dec1_merge(torch.cat([u1, s1], dim=1)))

        noise = self.head(u1)
        return torch.clamp(x - noise, min=-1.0, max=1.0)
