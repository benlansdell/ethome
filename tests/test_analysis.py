#####################
## Setup test code ##
#####################

import pytest

from behaveml import VideosetDataFrame, clone_metadata

#Metadata is a dictionary
def test_clone_metadata(tracking_files, label_files, metadata_params):

    metadata = clone_metadata(tracking_files, 
                                label_files = label_files, 
                                frame_length = metadata_params['frame_length'], 
                                fps = metadata_params['fps'], 
                                units = metadata_params['units'], 
                                resolution = metadata_params['resolution'])

    assert len(metadata) == 5
    assert metadata[list(metadata.keys())[0]]['fps'] == metadata_params['fps']



def test_VideoDataFrame(tracking_files, label_files, metadata, metadata_params):
    # metadata = clone_metadata(tracking_files, 
    #                             label_files = label_files, 
    #                             frame_length = metadata_params['frame_length'], 
    #                             fps = metadata_params['fps'], 
    #                             units = metadata_params['units'], 
    #                             resolution = metadata_params['resolution'])

    #Not working right now...
    # with pytest.raises(NotImplementedError):
    #     df = VideosetDataFrame(metadata)

    #Eventually, check we can make it without error
    try: df = VideosetDataFrame(metadata)
    except: assert False, "Failed to make VideosetDataFrame object"

    try: df = VideosetDataFrame({})
    except: assert False, "Failed to make empty VideosetDataFrame object"

    metadata_no_labels = clone_metadata(tracking_files, 
                                frame_length = metadata_params['frame_length'], 
                                fps = metadata_params['fps'], 
                                units = metadata_params['units'], 
                                resolution = metadata_params['resolution'])
                                
    try: df = VideosetDataFrame(metadata_no_labels)
    except: assert False, "Failed to make VideosetDataFrame object without labels"

    #Also check that improper formatted metadata raises the right exception

    #Check that 