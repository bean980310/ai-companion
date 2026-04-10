from ai_companion_core.detect_language import detect_system_language
from src.common.args import parse_args

args = parse_args()

if args.language:
    default_language = args.language
else:
    default_language = detect_system_language()
