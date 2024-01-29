#####################
## Setup test code ##
#####################


def test_ffmpeg():
    from ethome.utils import checkFFMPEG

    assert type(checkFFMPEG()) is bool
