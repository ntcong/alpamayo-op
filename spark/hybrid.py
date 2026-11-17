"""Move the Edge vision encoder and projector to the CPU so the model fits an 8 GB card.

Same change as the launcher in edge-setup.sh (ran on an RTX 2060 Super, 2026-11-11). Not NVIDIA's validated
setup: the encoder runs in float32 on the CPU, so sampled text and trajectories can differ slightly.
Written against transformers 5.17.0 (Cosmos3EdgeModel: visual, projector, language_model).
"""

import torch


def to_hybrid(model):
    core = model.vlm.model
    visual = core.visual.to("cpu").float()
    projector = core.projector.to("cpu").float()
    # Unregister both: model.dtype reads the first parameter and must stay bfloat16 for autocast
    del core.visual, core.projector

    def get_image_features(pixel_values, image_grid_thw=None, **_):
        # Same steps as Cosmos3EdgeModel.get_image_features; forward() moves the result to the GPU
        grid = image_grid_thw.cpu()
        out = visual(
            pixel_values.to("cpu", torch.float32), grid_thw=grid, return_dict=True
        )
        embeds = projector(out.last_hidden_state)
        out.pooler_output = torch.split(
            embeds, (grid.prod(-1) // projector.spatial_merge_size**2).tolist()
        )
        return out

    core.get_image_features = get_image_features
    torch.cuda.empty_cache()
