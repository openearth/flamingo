import unittest
import datetime
import logging
import os

from flamingo import batch, filesys
from flamingo.classification import channels

logging.basicConfig(level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class TestCase(unittest.TestCase):

    def test_data_path(self):
        from pkg_resources import Requirement, resource_filename
        dirname = resource_filename(Requirement.parse("flamingo"), "data")
        # should only contain argusnl
        self.assertEqual(1, len(os.listdir(dirname)))
        logger.info(dirname)

    def test_filesys(self):
        dirname = 'argusnl'
        images = filesys.get_image_list(dirname)
        logger.info("found {:d} images in {:s}".format(len(images), dirname))

    # run only in the morning, when I have time to wait
    @unittest.skipIf(datetime.datetime.now().hour > 10, "run this test before 10am it takes a while")
    def test_batch(self):
        dirname = 'argusnl'
        batch.run_preprocessing(dirname, feat_extract=True, segmentation=True)

    def test_add_channels(self):
        """test if we extend the channels"""
        dirname = 'argusnl'
        images = filesys.get_image_list(dirname)
        filename = images[0]
        img = filesys.read_image_file(dirname, filename)
        all_channels = channels.add_channels(img, colorspace='rgb')
        # we should have quite a number of channels
        self.assertGreater(all_channels.shape[2], 10)
