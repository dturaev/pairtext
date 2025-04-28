import pytest

from pairtext.aligner import align_sentences

sentence_pairs = [
    {
        "source": ["The cat", "is black"],
        "target": ["Die Katze", "ist schwarz"],
        "expected": [([0], [0]), ([1], [1])],
    },
    {
        "source": [
            "Der Fuchs hüpft geschwind.",
            "Er springt über den faulen Hund.",
        ],
        "target": [
            "The quick brown fox jumps over the lazy dog.",
        ],
        "expected": [([0, 1], [0])],
    },
    {
        "source": [
            "The quick brown fox jumps over the lazy dog.",
            "It's a beautiful day.",
        ],
        "target": [
            "Der Fuchs hüpft geschwind.",
            "Er springt über den faulen Hund.",
            "Es ist ein schöner Tag.",
        ],
        "expected": [([0], [0, 1]), ([1], [2])],
    },
    {
        # https://www.gutenberg.org/ebooks/74
        "source": [
            "The old lady whirled round, and snatched her skirts out of danger.",
            "The lad fled on the instant, scrambled up the high board-fence, and disappeared over it.",
            "His aunt Polly stood surprised a moment, and then broke into a gentle laugh.",
            "“Hang the boy, can’t I never learn anything?",
            "Ain’t he played me tricks enough like that for me to be looking out for him by this time?",
            "But old fools is the biggest fools there is.",
            "Can’t learn an old dog new tricks, as the saying is.",
            "But my goodness, he never plays them alike, two days, and how is a body to know what’s coming?",
            "He ’pears to know just how long he can torment me before I get my dander up, and he knows if he can make out to put me off for a minute or make me laugh, it’s all down again and I can’t hit him a lick.",
            "I ain’t doing my duty by that boy, and that’s the Lord’s truth, goodness knows.",
            "Spare the rod and spile the child, as the Good Book says.",
            "I’m a laying up sin and suffering for us both, I know.",
            "He’s full of the Old Scratch, but laws-a-me!",
            "he’s my own dead sister’s boy, poor thing, and I ain’t got the heart to lash him, somehow.",
            "Every time I let him off, my conscience does hurt me so, and every time I hit him my old heart most breaks.",
            "Well-a-well, man that is born of woman is of few days and full of trouble, as the Scripture says, and I reckon it’s so.",
            "He’ll play hookey this evening,[*] and I’ll just be obleeged to make him work, tomorrow, to punish him.",
            "It’s mighty hard to make him work Saturdays, when all the boys is having holiday, but he hates work more than he hates anything else, and I’ve got to do some of my duty by him, or I’ll be the ruination of the child.”",
        ],
        # https://www.gutenberg.org/ebooks/30165
        "target": [
            "Die alte Dame fuhr herum und brachte ihre Röcke in Sicherheit, während der Bursche, den Augenblick wahrnehmend, auf den hohen Bretterzaun kletterte und jenseits verschwand.",
            "Tante Polly stand sprachlos, dann begann sie gutmütig zu lächeln.",
            "„Der Kuckuck hole den Jungen!",
            "Werde ich denn das niemals lernen?",
            "Hat er mir denn nicht schon Streiche genug gespielt, daß ich immer wieder auf den Leim krieche?",
            "Aber alte Torheit ist die größte Torheit, und ein alter Hund lernt keine neuen Kunststücke mehr.",
            "Aber, du lieber Gott, er macht jeden Tag neue, und wie kann jemand bei ihm wissen, was kommt!",
            "Es scheint, er weiß ganz genau, wie lange er mich quälen kann, bis ich dahinter komme, und ist gar zu gerissen, wenn es gilt, etwas ausfindig zu machen, um mich für einen Augenblick zu verblüffen oder mich wider Willen lachen zu machen, es ist immer dieselbe Geschichte, und ich bringe es nicht fertig, ihn zu prügeln.",
            "Ich tue meine Pflicht nicht an dem Knaben, wie ich sollte, Gott weiß es.",
            "‚Spare die Rute, und du verdirbst dein Kind‘, heißt es.",
            "Ich begehe vielleicht unrecht und kann es vor mir und ihm nicht verantworten, fürcht‘ ich.",
            "Er steckt voller Narrenspossen und allerhand Unsinn — aber einerlei!",
            "Er ist meiner toten Schwester Kind, ein armes Kind, und ich habe nicht das Herz, ihn irgendwie am Gängelband zu führen.",
            "Wenn ich ihn sich selbst überlasse, drückt mich mein Gewissen, und so oft ich ihn schlagen muß, möchte mit das alte Herz brechen.",
            "Nun, mag‘s drum sein, der weibgeborene Mensch bleibt halt sein ganzes Leben durch in Zweifel und Irrtum, wie die heilige Schrift sagt, und ich denke, es ist so.",
            "Er wird wieder den ganzen Abend Blindekuh spielen, und ich sollte ihn von Rechts wegen, um ihn zu strafen, morgen arbeiten lassen.",
            "Es ist wohl hart für ihn, am Samstag stillzusitzen, wenn alle anderen Knaben Feiertag haben, aber er haßt Arbeit mehr als irgend sonst was, und ich will meine Pflicht an ihm tun, oder ich würde das Kind zu Grunde richten.“",
        ],
        "expected": [
            ([0, 1], [0]),
            ([2], [1]),
            ([3], [2, 3]),
            ([4], [4]),
            ([5, 6], [5]),
            ([7], [6]),
            ([8], [7]),
            ([9], [8]),
            ([10], [9]),
            ([11], [10]),
            ([12], [11]),
            ([13], [12]),
            ([14], [13]),
            ([15], [14]),
            ([16], [15]),
            ([17], [16]),
        ],
    },
]


@pytest.mark.parametrize("dataset", sentence_pairs)
def test_align_sentences(dataset):
    source, target = dataset["source"], dataset["target"]
    alignment = align_sentences(source, target, group_penalty=0.4)
    assert alignment.result == dataset["expected"]
