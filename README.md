# Pairtext

This is a small Python package to align [parallel text](https://en.wikipedia.org/wiki/Parallel_text).

-----

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Description

This Python package was designed for aligning machine-translated text to the source text, optimized for simplicity and ease of use. It creates 1:1, 1:n, n:1, 1:0, and 0:1 sentence alignments between bilingual documents.

It is based on a simple dynamic programming algorithm that traverses a cosine similarity matrix of [LaBSE embeddings](https://huggingface.co/sentence-transformers/LaBSE) ([Feng et al. 2020](https://arxiv.org/abs/2007.01852)) to identify an optimal alignment path, resulting in 1:1, 1:n, and n:1 alignments. It runs in quadratic time, so it’s best suited for small documents up to ~1000–2000 sentences. (It should be possible to achieve near-linear performance with banded DP and anchoring.)

The tool also allows for 1:0 and 0:1 alignments, by re-aligning 1:n and n:1 segments in two additional passes. This approach wasn't extensively tested, but works reasonably well.

While several excellent alignment tools exist, [Bertalign](https://github.com/bfsujason/bertalign) didn't produce the desired results for my use case, and [Vecalign](https://github.com/thompsonb/vecalign) offered more features than necessary and had a steeper learning curve.

### Limitations

- n:m alignments are not supported
- Mistranslations/gaps in between correct sentences in 1:n segments are not identified (this should be possible based on pairwise sentence concatenations)
- Gaps are not allowed at the beginning and at the end of the text; i.e., the beginning and the end of the parallel text must align
- The algorithm assumes that the sentence order is preserved, and works best if there are not too many gaps

## Installation

The package can be installed from GitHub via:

```bash
pip install git+https://github.com/dturaev/pairtext
```

It depends on [`wtpsplit`](https://github.com/segment-any-text/wtpsplit) (for sentence splitting) and [`sentence-transformers`](https://www.sbert.net) (for sentence similarity). It’s recommended to install it in a virtual environment, e.g. using [uv](https://github.com/astral-sh/uv), [venv](https://docs.python.org/3/library/venv.html), or conda.

## Usage

The tool can be used as a stand-alone executable:

```bash
echo "One dollar and eighty-seven cents. That was all. And sixty cents of it was in pennies." > source.txt
echo "Ein Dollar und siebenundachtzig Cent. Und sechzig Cents davon waren in Pennys." > target.txt
pairtext --gapped source.txt target.txt
[([0], [0]), ([1], []), ([2], [1])]
```

A short excerpt from "The Adventures of Tom Sawyer" by Mark Twain, from Project Gutenberg ([english version](https://www.gutenberg.org/ebooks/74), [german version](https://www.gutenberg.org/ebooks/30165)) is included in the `tests/` directory. After `git clone https://github.com/dturaev/pairtext.git && cd pairtext`, you can run:

```bash
# `-m 0.4` option is used because of lower similarity of literary translation,
# and it's still not low enough for two sentence pairs in this example
pairtext --gapped -m 0.4 tests/tom_sawyer_en.txt tests/tom_sawyer_de.txt
[([0], [0]), ([1], [1]), ([2, 3], [2]), ...
```

A help message can be obtained via `pairtext -h`.

The package can also be used as a Python library:

```python
>>> from pairtext import align_sentences, detect_gaps, sentence_tokenize
>>> src = ["One dollar and eighty-seven cents.", "That was all.", "And sixty cents of it was in pennies."]
>>> tgt = "Ein Dollar und siebenundachtzig Cent. Und sechzig Cents davon waren in Pennys."
>>> tgt = sentence_tokenize(tgt)
>>> tgt
['Ein Dollar und siebenundachtzig Cent.', 'Und sechzig Cents davon waren in Pennys.']
>>> alignment = align_sentences(src, tgt)
>>> alignment.result
[([0], [0]), ([1, 2], [1])]
>>> gapped = detect_gaps(alignment)
>>> gapped.result
[([0], [0]), ([1], []), ([2], [1])]
```

## License

The package is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
