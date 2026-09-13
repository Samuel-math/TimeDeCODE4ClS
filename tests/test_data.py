import tempfile
import unittest
from pathlib import Path
import numpy as np
from data import prepare


class ProtocolTests(unittest.TestCase):
    def test_official_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'Example'
            d.mkdir()
            (d / 'Example_TRAIN.ts').write_text('@data\n1,2,3,4:a\n2,3,4,5:b\n3,4,5,6:a\n4,5,6,7:b\n')
            (d / 'Example_TEST.ts').write_text('@data\n100,101,102,103:a\n200,201,202,203:b\n')
            train, selection, meta = prepare(tmp, 'Example', 4, 16, 42, .5, 'test_selection')
            self.assertEqual(len(train[0]), 4)
            self.assertEqual(len(selection[0]), 2)
            self.assertEqual(meta['val_indices'], [])
            self.assertEqual(meta['selection_split'], 'TEST')
            self.assertFalse(meta['independent_test'])
            self.assertAlmostEqual(float(np.asarray(meta['normalization_mean']).squeeze()), 4.)
            self.assertGreater(float(selection[0].mean()), 10.)


if __name__ == '__main__':
    unittest.main()
