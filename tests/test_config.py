import pytest
from hjmodel.config import StopCode

def test_stopcode_lookup_by_name_and_id():
    sc = StopCode.from_name("HJ")
    assert sc == StopCode.HJ
    sc2 = StopCode.from_id(sc.value)
    assert sc2 == StopCode.HJ

def test_stopcode_hex_metadata():
    assert hasattr(StopCode.HJ, "hex")
    assert isinstance(StopCode.HJ.hex, str)
    assert StopCode.HJ.hex != StopCode.WJ.hex
