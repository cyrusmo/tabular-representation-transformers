# Architecture

`TabularStateTransformer` is a modular research architecture for testing whether feature identity, sparse gates, spectral/wavelet expansion, interaction blocks, and regime-gated heads help neural tabular models on non-smooth tasks.

The default model keeps mean pooling over final feature tokens. The post-P1 architecture probe adds
two opt-in readouts:

- `pooling="cls"` prepends a learned CLS token before the interaction block and reads that token.
- `pooling="attention"` uses a learned query to attention-pool the final token states.

The probe also adds opt-in lightweight feature-cross tokens with `use_feature_crosses=true` and
`cross_max_features=16`. Crosses use only the first `min(n_features, cross_max_features)` processed
features. Each pair creates `x_i * x_j`, then scales and shifts that scalar with learned
pair-specific vectors plus a pair identity embedding. This is a controlled synthetic-task probe for
XOR/noise failure modes, not a general real-data feature-selection method.
