from liver_ai.real_data import patient_level_split


def test_patient_level_split_is_disjoint_and_complete():
    ids = [f"P{i:03d}" for i in range(20)]
    train, val, test = patient_level_split(ids)
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(ids)
