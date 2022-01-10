#!usr/bin/env python
# -*- coding: utf8 -*-
import pytest

def pytest_addoption(parser):
    parser.addoption("--smilextract-bin-path", action="store", dest='stringvalue')

@pytest.fixture
def smilextract_bin_path(request):
    return request.config.getoption("--smilextract-bin-path")