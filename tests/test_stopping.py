import torch

from warehouse_planning.llm import StopWhenVerified


class CharTokenizer:
    """One token per character, enough to exercise line tracking."""

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["".join(chr(int(c)) for c in row) for row in sequences]


def tokens(*rows):
    width = max(map(len, rows))
    return torch.tensor([[ord(c) for c in row.ljust(width)] for row in rows])


def test_verifier_only_sees_finished_lines():
    seen = []
    stop = StopWhenVerified(CharTokenizer(), 0, [lambda text: seen.append(text) or False])
    stop(tokens("(move a"), None)
    assert seen == []
    stop(tokens("(move a)\n(pi"), None)
    assert seen == ["(move a)"]


def test_each_sequence_stops_on_its_own():
    stop = StopWhenVerified(CharTokenizer(), 0, [lambda _: True, lambda _: False])
    assert stop(tokens("bad\n", "ok\n "), None).tolist() == [True, False]


def test_a_stopped_sequence_stays_stopped():
    answers = iter([True])
    stop = StopWhenVerified(CharTokenizer(), 0, [lambda _: next(answers)])
    stop(tokens("bad\n"), None)
    assert stop(tokens("bad\nmore\n"), None).tolist() == [True]


def test_prompt_tokens_are_skipped():
    seen = []
    stop = StopWhenVerified(CharTokenizer(), 7, [lambda text: seen.append(text) or False])
    stop(tokens("prompt\nplan\n"), None)
    assert seen == ["plan"]
