import random

import pytest

from training.cv import temporal_folds


def races_of(n: int) -> list[tuple[int, int]]:
    """n races over consecutive seasons of 22 rounds: (2023, 1) ... in calendar order."""
    return [(2023 + i // 22, i % 22 + 1) for i in range(n)]


def test_56_races_and_5_folds_give_the_blocks_of_the_spec():
    folds = temporal_folds(races_of(56), 5)
    assert [len(f.train_races) for f in folds] == [10, 20, 29, 38, 47]
    assert [len(f.valid_races) for f in folds] == [10, 9, 9, 9, 9]
    assert [f.number for f in folds] == [1, 2, 3, 4, 5]


def test_all_training_races_come_before_the_validation_races():
    for fold in temporal_folds(races_of(56), 5):
        assert max(fold.train_races) < min(fold.valid_races)


def test_validation_blocks_are_the_last_five_and_do_not_overlap():
    races = races_of(56)
    folds = temporal_folds(races, 5)
    validated = [race for f in folds for race in f.valid_races]
    assert len(validated) == len(set(validated)) == 46
    assert validated == races[10:]


def test_training_expands_by_one_block_and_includes_the_previous_validation():
    folds = temporal_folds(races_of(56), 5)
    for previous, current in zip(folds, folds[1:], strict=False):
        assert current.train_races == previous.train_races + previous.valid_races


def test_input_order_does_not_matter():
    races = races_of(56)
    shuffled = races[:]
    random.Random(0).shuffle(shuffled)
    assert temporal_folds(shuffled, 5) == temporal_folds(races, 5)


def test_exactly_n_folds_plus_one_races_gives_blocks_of_one():
    folds = temporal_folds(races_of(6), 5)
    assert [len(f.train_races) for f in folds] == [1, 2, 3, 4, 5]
    assert all(len(f.valid_races) == 1 for f in folds)


def test_too_few_races_raises():
    with pytest.raises(ValueError, match="at least 6 races"):
        temporal_folds(races_of(5), 5)
