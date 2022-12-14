#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser(
    """Take Kaldi train text file and extra text files and check that 
    the extra text files do not contain unseen graphemes."""
)
parser.add_argument("train_text", help="The train text file")
parser.add_argument("extra_text", nargs="*", help="The extra files, checked for unseen graphemes")
parser.add_argument("--unk-token", help="Unknown token symbol", default="<UNK>")
args = parser.parse_args()

if not args.extra_text:
    # Nothing to check
    sys.exit(0)

train_graphemes = set()
with open(args.train_text, encoding="utf-8") as fi:
    for line in fi:
        uttid, *words = line.strip().split()
        for word in words:
            if word == args.unk_token:
                pass
            else:
                train_graphemes |= set(word)

for textfile in args.extra_text:
    with open(textfile, encoding="utf-8") as fi:
        for line in fi:
            uttid, *words = line.strip().split()
            for word in words:
                if word == args.unk_token:
                    pass
                elif not train_graphemes.issuperset(word):
                    print("FOUND OFFENDING LINE!:")
                    print(line)
                    # Not good!
                    sys.exit(1)

# All good
sys.exit(0)
