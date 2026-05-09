from hj.evolution import StopCode


def test_stopcode_lookup_by_name_and_id():
    assert StopCode["HJ"] is StopCode.HJ
    assert StopCode(StopCode.HJ.value) is StopCode.HJ
