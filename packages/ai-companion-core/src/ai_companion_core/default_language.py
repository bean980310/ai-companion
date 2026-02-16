from .args import parse_args
from .detect_language import detect_system_language

args=parse_args()

if args.language:
    default_language = args.language
else:
    default_language = detect_system_language()