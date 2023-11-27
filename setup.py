from random import shuffle, random
import inspect
from typing import Optional

class Setup:
    def __init__(self, calc, all_possible_inputs):
        self.calc = calc
        self.pairs = [(x, calc(x)) for x in all_possible_inputs]

    @property
    def func_code(self) -> str:
        return inspect.getsource(self.calc)

    def get_sample(self, train_size: int, test_label: Optional[bool] = None) -> tuple[str, str, str]:
        assert train_size < len(self.pairs), f"Requested {train_size} train pairs from a dataset with {len(self.pairs)} elements"
        assert not train_size % 2, "We're balancing inputs, so train_size must be even"
        shuffle(self.pairs)

        zeros = []
        ones = []
        for pair in self.pairs:
            if pair[1] and len(ones) < train_size / 2:
                ones.append(pair)
            elif not pair[1] and len(zeros) < train_size / 2:
                zeros.append(pair)

        train_data = zeros + ones
        assert len(train_data) == train_size, "Not enough zeros or ones in the dataset"

        shuffle(train_data)
        train_data_str = "\n".join([f"Input: {pair[0]} Label: {pair[1]}" for pair in train_data])

        if test_label is None:
            test_label = bool(random() > 0.5)
        test_input = next(pair[0] for pair in reversed(self.pairs) if pair[1] == test_label)
        assert test_input not in [x[0] for x in train_data]

        return train_data_str, test_input, str(test_label)
