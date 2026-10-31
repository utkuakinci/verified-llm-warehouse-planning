from types import SimpleNamespace

from warehouse_planning.llm import ClaudeLLM


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get_final_message(self):
        return self.response


class FakeClient:
    def __init__(self, stop_reason="end_turn", text="(move robot1 a00 a10)"):
        self.requests = []
        blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
        self.response = SimpleNamespace(stop_reason=stop_reason, content=blocks)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **request):
        self.requests.append(request)
        return FakeStream(self.response)


MESSAGES = [{"role": "system", "content": "plan"}, {"role": "user", "content": "go"}]


def test_system_prompt_moves_out_of_the_conversation():
    client = FakeClient()
    ClaudeLLM(client=client).generate([MESSAGES], 0.3, 1, 100)
    request = client.requests[0]
    assert request["system"] == "plan"
    assert request["messages"] == [{"role": "user", "content": "go"}]


def test_sampling_parameters_are_not_sent():
    """Current Claude models reject temperature with a 400."""
    client = FakeClient()
    ClaudeLLM(client=client).generate([MESSAGES], 0.3, 1, 100)
    assert not {"temperature", "top_p", "top_k", "seed"} & set(client.requests[0])


def test_only_text_blocks_reach_the_validator():
    outputs = ClaudeLLM(client=FakeClient()).generate([MESSAGES], 0, 0, 100)
    assert outputs == ["(move robot1 a00 a10)"]


def test_a_refusal_counts_as_no_plan():
    outputs = ClaudeLLM(client=FakeClient(stop_reason="refusal", text="")).generate([MESSAGES], 0, 0, 100)
    assert outputs == [""]


def test_batch_keeps_its_order():
    client = FakeClient()
    outputs = ClaudeLLM(client=client).generate([MESSAGES] * 5, 0, 0, 100)
    assert len(outputs) == 5
    assert len(client.requests) == 5
