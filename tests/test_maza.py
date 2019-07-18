import unittest

def maza(a1, a2):
    return a1**a2

class Test_maza(unittest.TestCase):
    def test_maza1(self):
        self.assertEqual(maza(4, 3), 64)

    def test_maza2(self):
        self.assertEqual(maza(2,2), 5)
