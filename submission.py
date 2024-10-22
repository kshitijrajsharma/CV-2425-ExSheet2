"""Submission for exercise sheet 2

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

from typing import Callable

import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()  ## force nn.module init
        self.f = nn.Linear(2, 1)
        with torch.no_grad():  # using grad because we want directly weight assigned instead of learning
            self.f.weight = nn.Parameter(torch.tensor([weight]))
            self.f.bias = nn.Parameter(torch.tensor([bias]))

    def forward(self, x):  ## activation function
        return (self.f(x) > 0).int()


# Exercise 2.1 (AND gate)
def assignment_ex1(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    # YOUR CODE GOES HERE
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    and_gate = Gate(weight=[0.2, 0.1], bias=-0.2)
    return and_gate(inputs)


# Exercise 2.2 (OR gate)
def assignment_ex2(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    # YOUR CODE GOES HERE
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    or_gate = Gate(weight=[0.1, 0.1], bias=0.0)
    return or_gate(inputs)
