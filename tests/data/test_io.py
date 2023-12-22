from dataclasses import dataclass
import io
import json

from repepo.data.io import jdump


class FakeFile(io.StringIO):
    def close(self):
        pass


def test_jdump_converts_dataclasess_to_dicts() -> None:
    @dataclass
    class Foo:
        bar: str

    foos = [Foo("baz"), Foo("qux")]
    fake_file = FakeFile()
    jdump(foos, fake_file)
    assert json.loads(fake_file.getvalue()) == [{"bar": "baz"}, {"bar": "qux"}]
