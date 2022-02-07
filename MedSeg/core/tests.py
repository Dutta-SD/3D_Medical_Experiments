from dataloaders import SpleenImageDataLoader
import unittest
import config

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.dl = SpleenImageDataLoader(config.DATA_DIR, num_val_samples=1)
        self.dl.setup()
        self.vl = self.dl.val_dataloader()
        # self.tl = self.dl.train_dataloader()

    def test_val_dl(self):
        x = None
        for i in self.vl:
            x = i
            break

        self.assertEqual(x['image'].shape.__len__(), 5)


if __name__ == '__main__':
    unittest.main(
        verbosity=2,
    )
