"""Registry for the seven supported Step-5 strategies."""

from .linear import (
    CenterBaseLinearRaw768, Center5merLinearRaw768,
    Center5merGaussianPoolLinearRaw768, Center5merLearnedPoolLinearRaw768,
)
from .nonlinear import Center5merMLPRaw768, Center5merResidualDepthwiseConvRaw768
from .deepplant import DeepPlantSignalBaseBiLSTM

FUSION_STRATEGY = "center_5mer_emb8_none"
STRATEGIES = (
    "center_base_linear_raw768",
    "center_5mer_linear_raw768",
    "center_5mer_gaussian_pool_linear_raw768",
    "center_5mer_learned_pool_linear_raw768",
    "center_5mer_mlp_raw768",
    "center_5mer_residual_dwconv_raw768",
    FUSION_STRATEGY,
)


def build_model(strategy, hidden_dim, source_window_size, settings):
    if source_window_size < 5 or source_window_size % 2 != 1:
        raise ValueError("Features must have an odd window size >= 5")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")
    common = dict(hidden_dim=hidden_dim, window_size=source_window_size)
    if strategy == STRATEGIES[0]:
        return CenterBaseLinearRaw768(**common)
    if strategy == STRATEGIES[1]:
        return Center5merLinearRaw768(**common)
    if strategy == STRATEGIES[2]:
        sigma = settings["gaussian_sigma"]
        if sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        return Center5merGaussianPoolLinearRaw768(**common, sigma=sigma)
    if strategy == STRATEGIES[3]:
        return Center5merLearnedPoolLinearRaw768(**common)
    if strategy == STRATEGIES[4]:
        return Center5merMLPRaw768(**common, hidden_dims=tuple(settings["mlp_hidden_dims"]),
                                 dropout=settings["dropout"])
    if strategy == STRATEGIES[5]:
        return Center5merResidualDepthwiseConvRaw768(
            **common, channels=settings["conv_channels"],
            head_dim=settings["conv_head_dim"], dropout=settings["dropout"])
    if strategy == FUSION_STRATEGY:
        return DeepPlantSignalBaseBiLSTM(hidden_dim=hidden_dim, **settings["deepplant"])
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
