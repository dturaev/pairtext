import numpy as np
import pytest

from pairtext.aligner import align_sentences
from pairtext.gaps import detect_gaps

sentence_pairs_gapped = [
    {
        "source": [
            "One dollar and eighty-seven cents.",
            "That was all.",
            "And sixty cents of it was in pennies.",
        ],
        "target": [
            "Ein Dollar und siebenundachtzig Cent.",
            "Und sechzig Cents davon waren in Pennys.",
        ],
        "expected": [([0], [0]), ([1], []), ([2], [1])],
    },
    {
        # This example demonstrates the limits of the `max_path` function.
        # A correct alignment is only possible with `-g` <= 0.28
        "source": [
            "But old fools is the biggest fools there is.",
            "Can’t learn an old dog new tricks, as the saying is.",
            "I ain’t doing my duty by that boy, and that’s the Lord’s truth, goodness knows.",
        ],
        "target": [
            "Aber alte Torheit ist die größte Torheit, und ein alter Hund lernt keine neuen Kunststücke mehr.",
            "Aber, du lieber Gott, er macht jeden Tag neue, und wie kann jemand bei ihm wissen, was kommt!",
            "Es scheint, er weiß ganz genau, wie lange er mich quälen kann, bis ich dahinter komme, und ist gar zu gerissen, wenn es gilt, etwas ausfindig zu machen, um mich für einen Augenblick zu verblüffen oder mich wider Willen lachen zu machen, es ist immer dieselbe Geschichte, und ich bringe es nicht fertig, ihn zu prügeln.",
            "Ich tue meine Pflicht nicht an dem Knaben, wie ich sollte, Gott weiß es.",
        ],
        "expected": [([0], [0]), ([1], []), ([], [1, 2]), ([2], [3])],
        "expected_correct": [([0, 1], [0]), ([], [1, 2]), ([2], [3])],
        "matrix": np.array(
            [
                [0.62457067, 0.22217272, 0.10775606, 0.08698118],
                [0.55267566, 0.27590013, 0.29863203, 0.32706615],
                [0.2689323, 0.34125045, 0.24806654, 0.7923431],
            ],
            dtype=np.float32,
        ),
    },
    {
        # This example demonstrates the limits of the `max_path` function.
        # There is no parameter combination that allows for a correct alignment.
        "source": [
            "But old fools is the biggest fools there is.",
            "Can’t learn an old dog new tricks, as the saying is.",
            "Spare the rod and spile the child, as the Good Book says.",
        ],
        "target": [
            "Aber alte Torheit ist die größte Torheit, und ein alter Hund lernt keine neuen Kunststücke mehr.",
            "Aber, du lieber Gott, er macht jeden Tag neue, und wie kann jemand bei ihm wissen, was kommt!",
            "Es scheint, er weiß ganz genau, wie lange er mich quälen kann, bis ich dahinter komme, und ist gar zu gerissen, wenn es gilt, etwas ausfindig zu machen, um mich für einen Augenblick zu verblüffen oder mich wider Willen lachen zu machen, es ist immer dieselbe Geschichte, und ich bringe es nicht fertig, ihn zu prügeln.",
            "Ich tue meine Pflicht nicht an dem Knaben, wie ich sollte, Gott weiß es.",
            "‚Spare die Rute, und du verdirbst dein Kind‘, heißt es.",
        ],
        "expected": [([0], [0]), ([1], []), ([], [1, 2, 3]), ([2], [4])],
        "matrix": np.array(
            [
                [0.62457067, 0.22217272, 0.10775606, 0.08698118, 0.04378524],
                [0.55267566, 0.27590013, 0.29863203, 0.32706615, 0.246663],
                [0.1258886, 0.11385883, -0.00681811, 0.36908442, 0.5740005],
            ],
            dtype=np.float32,
        ),
    },
]


@pytest.mark.parametrize("dataset", sentence_pairs_gapped)
def test_detect_gaps(dataset):
    source = dataset["source"]
    target = dataset["target"]
    expected = dataset["expected"]
    alignment = align_sentences(source, target)
    alignment = detect_gaps(alignment, minscore_translation=0.5)
    assert alignment.result == expected


def test_align_sentences_with_parameters():
    dataset = sentence_pairs_gapped[1]
    source = dataset["source"]
    target = dataset["target"]

    alignment1 = align_sentences(source, target, group_penalty=0.29)
    alignment1 = detect_gaps(alignment1)
    assert alignment1.result == dataset["expected"]

    alignment2 = align_sentences(source, target, group_penalty=0.28)
    alignment2 = detect_gaps(alignment2)
    assert alignment2.result == dataset["expected_correct"]
