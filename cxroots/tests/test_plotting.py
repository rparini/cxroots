import os
import tempfile

import pytest
from numpy import cos, exp

from cxroots import Circle


@pytest.fixture()
def results_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_save_plot(results_dir):
    file_path = os.path.join(results_dir, "circle.pdf")
    Circle(1j, 2).show(save_file=file_path)
    assert os.path.exists(file_path)


def test_save_animation(results_dir):
    file_path = os.path.join(results_dir, "circle.gif")
    Circle(1j, 2).demo_roots(f=lambda z: cos(exp(1j * z)), save_file=file_path)
    assert os.path.exists(file_path)


def test_save_roots(results_dir):
    file_path = os.path.join(results_dir, "roots.png")
    Circle(1j, 2).roots(f=lambda z: cos(exp(1j * z))).show(save_file=file_path)
    assert os.path.exists(file_path)
