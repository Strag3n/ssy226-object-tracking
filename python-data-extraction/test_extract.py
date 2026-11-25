import struct
import unittest

import numpy as np

from extract import CDR, cloud_array, image_array


class ExtractionTests(unittest.TestCase):
    def test_cdr_alignment_relative_to_payload(self):
        reader = CDR(b'\x00\x01\x00\x00' + struct.pack('<I4xd', 42, 1.25))
        self.assertEqual(reader.number('I'), 42)
        self.assertEqual(reader.number('d'), 1.25)
        reader.finish()

    def test_depth_preserves_endian_values_and_row_padding(self):
        msg = dict(encoding='16UC1', is_bigendian=True, height=2, width=2, step=6,
                   data=struct.pack('>HHHHHH', 1000, 0, 999, 2000, 3000, 999))
        np.testing.assert_array_equal(image_array(msg), [[1000, 0], [2000, 3000]])

    def test_cloud_preserves_nan_fields_and_padding(self):
        msg = dict(fields=[dict(name='x', offset=0, datatype=7, count=1)],
                   is_bigendian=False, point_step=4, row_step=12, height=2, width=2,
                   data=struct.pack('<6f', 1, float('nan'), 999, 3, 4, 999))
        result = cloud_array(msg)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.isnan(result['x'][0, 1]))
        self.assertEqual(result['x'][1, 0], 3)

    def test_truncated_image_rejected(self):
        with self.assertRaises(ValueError):
            image_array(dict(encoding='rgb8', is_bigendian=False, height=1, width=2,
                             step=6, data=b'abc'))


if __name__ == '__main__':
    unittest.main()
