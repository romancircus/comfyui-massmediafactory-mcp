"""Asset commands: list, metadata, publish."""

from ..cli_utils import EXIT_ERROR, EXIT_OK, _add_common_args, _is_pretty, _output, _parse_json_arg


def cmd_assets_list(args):
    """List generated assets."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.list_assets(asset_type=args.type, limit=args.limit)
    _output(result, pretty)
    return EXIT_OK


def cmd_assets_metadata(args):
    """Get full asset metadata."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.get_asset_metadata(args.asset_id)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_publish(args):
    """Publish asset to web directory."""
    from ..publish import publish_asset

    pretty = args.pretty or _is_pretty()
    result = publish_asset(
        asset_id=args.asset_id,
        target_filename=args.filename,
        manifest_key=args.manifest_key,
    )
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_workflow_lib(args):
    """Workflow library operations."""
    from .. import persistence

    pretty = args.pretty or _is_pretty()
    action = args.action

    workflow_data = None
    if args.workflow:
        workflow_data = _parse_json_arg(args.workflow)

    if action == "save":
        if not args.name or not workflow_data:
            from ..cli_utils import _error

            _output(_error("--name and --workflow required for action 'save'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.save_workflow(args.name, workflow_data)
    elif action == "load":
        if not args.name:
            from ..cli_utils import _error

            _output(_error("--name required for action 'load'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.load_workflow(args.name)
    elif action == "list":
        result = persistence.list_workflows()
    elif action == "delete":
        if not args.name:
            from ..cli_utils import _error

            _output(_error("--name required for action 'delete'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.delete_workflow(args.name)
    else:
        from ..cli_utils import _error

        _output(_error(f"Unknown action: {action}. Use save|load|list|delete", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def register_commands(sub, add_common=_add_common_args, **_kwargs):
    """Register asset subcommands."""
    p_assets = sub.add_parser("assets", help="Asset operations")
    assets_sub = p_assets.add_subparsers(dest="assets_command")

    p_al = assets_sub.add_parser("list", help="List generated assets")
    p_al.add_argument("--type", help="Asset type: images, video, audio")
    p_al.add_argument("--limit", type=int, default=20)
    add_common(p_al)
    p_al.set_defaults(func=cmd_assets_list)

    p_am = assets_sub.add_parser("metadata", help="Full asset metadata")
    p_am.add_argument("asset_id", help="Asset ID")
    add_common(p_am)
    p_am.set_defaults(func=cmd_assets_metadata)

    # publish
    p_pub = sub.add_parser("publish", help="Publish asset to web directory")
    p_pub.add_argument("asset_id", help="Asset ID")
    p_pub.add_argument("--filename", help="Target filename")
    p_pub.add_argument("--manifest-key", help="Manifest key for tracking")
    add_common(p_pub)
    p_pub.set_defaults(func=cmd_publish)

    # workflow-lib
    p_wl = sub.add_parser("workflow-lib", help="Workflow library operations")
    p_wl.add_argument("action", choices=["save", "load", "list", "delete"], help="Library action")
    p_wl.add_argument("--name", help="Workflow name")
    p_wl.add_argument("--workflow", help="Workflow JSON string or @file.json (for save)")
    add_common(p_wl)
    p_wl.set_defaults(func=cmd_workflow_lib)
