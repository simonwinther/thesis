import argparse
import inspect

from .unet import SuperResModel, UNetModel, EncoderUNetModel
from .wunet import WavUNetModel

NUM_CLASSES = 4  # t1n, t1c, t2w, t2f modalities


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
    num_groups=32,
    dims=2,
    in_channels=1,
    out_channels=0,  # automatically determine if 0
    bottleneck_attention=True,
    resample_2d=True,
    additive_skips=False,
    use_freq=False,
):
    if not channel_mult:
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 2, 2, 4, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"[MODEL] Unsupported image size: {image_size}")
    else:
        if isinstance(channel_mult, str):
            from ast import literal_eval

            channel_mult = literal_eval(channel_mult)
        elif isinstance(channel_mult, tuple):  # do nothing
            pass
        else:
            raise ValueError(f"[MODEL] Value for {channel_mult=} not supported")

    attention_ds = []
    if attention_resolutions:
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
    if out_channels == 0:
        out_channels = 2 * in_channels if learn_sigma else in_channels

    if not use_freq:
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=out_channels * (1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            dims=dims,
            num_groups=num_groups,
            bottleneck_attention=bottleneck_attention,
            additive_skips=additive_skips,
            resample_2d=resample_2d,
        )
    else:
        return WavUNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=out_channels * (1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            dims=dims,
            num_groups=num_groups,
            bottleneck_attention=bottleneck_attention,
            additive_skips=additive_skips,
            use_freq=use_freq,
        )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
