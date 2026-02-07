"""Template commands: list, get, create."""

from ..cli_utils import EXIT_ERROR, EXIT_OK, _add_common_args, _is_pretty, _output, _parse_json_arg


def cmd_templates_list(args):
    """List available templates."""
    from .. import templates as tmpl

    pretty = args.pretty or _is_pretty()
    tags = args.tags.split(",") if args.tags else None
    result = tmpl.list_templates(only_installed=args.installed, model_type=args.model, tags=tags)
    _output(result, pretty)
    return EXIT_OK


def cmd_templates_get(args):
    """Get raw template JSON."""
    from .. import templates as tmpl

    pretty = args.pretty or _is_pretty()
    result = tmpl.load_template(args.name)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_templates_create(args):
    """Create workflow from template with injected params."""
    from .. import templates as tmpl

    pretty = args.pretty or _is_pretty()
    template = tmpl.load_template(args.name)
    if "error" in template:
        _output(template, pretty)
        return EXIT_ERROR

    params = _parse_json_arg(args.params) if args.params else {}
    wf = tmpl.inject_parameters(template, params)
    _output(wf, pretty)
    return EXIT_OK


def register_commands(sub, add_common=_add_common_args, **_kwargs):
    """Register template subcommands."""
    p_tmpl = sub.add_parser("templates", help="Template operations")
    tmpl_sub = p_tmpl.add_subparsers(dest="templates_command")

    p_tl = tmpl_sub.add_parser("list", help="List templates")
    p_tl.add_argument("--installed", action="store_true", help="Only installed models")
    p_tl.add_argument("--model", help="Filter by model type")
    p_tl.add_argument("--tags", help="Filter by tags (comma-separated)")
    add_common(p_tl)
    p_tl.set_defaults(func=cmd_templates_list)

    p_tg = tmpl_sub.add_parser("get", help="Get raw template JSON")
    p_tg.add_argument("name", help="Template name")
    add_common(p_tg)
    p_tg.set_defaults(func=cmd_templates_get)

    p_tc = tmpl_sub.add_parser("create", help="Create workflow from template")
    p_tc.add_argument("name", help="Template name")
    p_tc.add_argument("--params", help="JSON params (or @file.json)")
    add_common(p_tc)
    p_tc.set_defaults(func=cmd_templates_create)
