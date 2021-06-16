import unittest
from src.utils import find_latest_ckpt
from unittest import mock
import os


class Test_utils_find_latest_ckpt(unittest.TestCase):
    # TODO check is ckpt a file
    def test_default_behavior(self):
        with mock.patch('os.listdir') as m_ld:
            m_ld.return_value = ['0.pt', '1.pt', '10.pt',
                                 '100.pt', '999.pt', 'a.pt', 'pt']
            r = find_latest_ckpt('test', r'(\d+).pt')
            self.assertEqual(r.path, os.path.join('test', '999.pt'))
            self.assertEqual(r.step, 999)

    def test_empty(self):
        with mock.patch('os.listdir') as m_ld:
            m_ld.return_value = []
            r = find_latest_ckpt('test', r'(\d+).pt')
            self.assertEqual(r, None)


if __name__ == '__main__':
    unittest.main()
