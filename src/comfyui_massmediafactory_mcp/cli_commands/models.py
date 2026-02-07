"""Model commands: list, constraints, compatibility, optimize, search, install."""

from ..cli_utils import EXIT_ERROR, EXIT_OK, _add_common_args, _is_pretty, _output


def cmd_models_list(args):
    """List installed models."""
    from .. import discovery

    pretty = args.pretty or _is_pretty()
    model_type = args.type or "all"
    result = discovery.list_models(model_type)
    _output(result, pretty)
    return EXIT_OK


def cmd_models_constraints(args):
    """Get model constraints."""
    from ..model_registry import get_model_constraints

    pretty = args.pretty or _is_pretty()
    result = get_model_constraints(args.model)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_models_compatibility(args):
    """Model compatibility matrix."""
    from ..compatibility import get_compatibility_matrix

    pretty = args.pretty or _is_pretty()
    result = get_compatibility_matrix()
    _output(result, pretty)
    return EXIT_OK


def cmd_models_optimize(args):
    """Hardware-optimized params."""
    from ..optimization import get_optimal_workflow_params

    pretty = args.pretty or _is_pretty()
    result = get_optimal_workflow_params(args.model, task=args.task or "i2v")
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_search_model(args):
    """Search Civitai for models."""
    from .. import models as models_mod

    pretty = args.pretty or _is_pretty()
    result = models_mod.search_civitai(args.query, model_type=args.type, limit=args.limit)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_install_model(args):
    """Download and install a model."""
    from .. import models as models_mod

    pretty = args.pretty or _is_pretty()
    result = models_mod.download_model(args.url, model_type=args.type, filename=args.filename)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def register_commands(sub, add_common=_add_common_args, **_kwargs):
    """Register model subcommands."""
    p_models = sub.add_parser("models", help="Model operations")
    models_sub = p_models.add_subparsers(dest="models_command")

    p_ml = models_sub.add_parser("list", help="List installed models")
    p_ml.add_argument("--type", help="Model type: checkpoint, lora, unet, etc.")
    add_common(p_ml)
    p_ml.set_defaults(func=cmd_models_list)

    p_mc = models_sub.add_parser("constraints", help="Model constraints")
    p_mc.add_argument("model", help="Model name: flux, wan, ltx, qwen")
    add_common(p_mc)
    p_mc.set_defaults(func=cmd_models_constraints)

    p_mcompat = models_sub.add_parser("compatibility", help="Compatibility matrix")
    add_common(p_mcompat)
    p_mcompat.set_defaults(func=cmd_models_compatibility)

    p_mo = models_sub.add_parser("optimize", help="Hardware-optimal params")
    p_mo.add_argument("model", help="Model name")
    p_mo.add_argument("--task", help="Task type: i2v, t2v, t2i")
    add_common(p_mo)
    p_mo.set_defaults(func=cmd_models_optimize)

    # search-model
    p_sm = sub.add_parser("search-model", help="Search Civitai for models")
    p_sm.add_argument("query", help="Search query")
    p_sm.add_argument("--type", help="Model type: checkpoint, lora, controlnet, etc.")
    p_sm.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    add_common(p_sm)
    p_sm.set_defaults(func=cmd_search_model)

    # install-model
    p_im = sub.add_parser("install-model", help="Download and install a model")
    p_im.add_argument("url", help="Download URL (Civitai or HuggingFace)")
    p_im.add_argument("--type", required=True, help="Model type: checkpoint|unet|lora|vae|controlnet|clip")
    p_im.add_argument("--filename", help="Target filename (auto-detected if omitted)")
    add_common(p_im)
    p_im.set_defaults(func=cmd_install_model)
