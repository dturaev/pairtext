import logging
from pathlib import Path

from pairtext.aligner import align_sentences
from pairtext.gaps import detect_gaps
from pairtext.tokenize import sentence_tokenize
from pairtext.utils import write_aligned_sentences_to_file

logger = logging.getLogger(__name__)


def run_alignment(args):

    if not args.tokenized:

        src_file_name = Path(args.file1).name
        tgt_file_name = Path(args.file2).name
        with open(args.file1) as f1, open(args.file2) as f2:
            logger.info(f"Tokenizing {src_file_name}...")
            src_sentences = sentence_tokenize(f1.read())
            logger.info(f"Tokenizing {tgt_file_name}...")
            tgt_sentences = sentence_tokenize(f2.read())

    else:

        with open(args.file1) as f1, open(args.file2) as f2:
            src_sentences = f1.read().splitlines()
            tgt_sentences = f2.read().splitlines()

    alignment = align_sentences(src_sentences, tgt_sentences)

    if args.gapped:
        alignment = detect_gaps(
            alignment,
            minscore_translation=args.minscore_translation,
            special_penalty=args.special_penalty,
            concat_preference=args.concat_preference,
        )
    print(alignment.result)

    if args.output_file:
        write_aligned_sentences_to_file(
            src_sentences, tgt_sentences, alignment, args.output_file
        )
