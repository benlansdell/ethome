#####################
## Setup test code ##
#####################

import pytest

def test_ffmpeg():
    from behaveml.utils import checkFFMPEG
    assert type(checkFFMPEG()) is bool