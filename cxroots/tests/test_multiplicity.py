import unittest

from cxroots.CxDerivative import multiplicity_correct

class TestMultiplicity(unittest.TestCase):
	def setUp(self):
		self.f = lambda z: (z-2)**2*(z-1)**5 
		self.df = lambda z: 2*(z-2)*(z-1)**5 + 5*(z-2)**2*(z-1)**4

	def test_multiplicity_check1_f(self):
		self.assertFalse(multiplicity_correct(self.f, None, 2, 1))

	def test_multiplicity_check2_f(self):
		self.assertTrue(multiplicity_correct(self.f, None, 2, 2))

	def test_multiplicity_check3_f(self):
		self.assertFalse(multiplicity_correct(self.f, None, 2, 3))

	def test_multiplicity_check4_f(self):
		self.assertFalse(multiplicity_correct(self.f, None, 1, 4))

	def test_multiplicity_check5_f(self):
		self.assertTrue(multiplicity_correct(self.f, None, 1, 5))

	def test_multiplicity_check6_f(self):
		self.assertFalse(multiplicity_correct(self.f, None, 1, 6))

	def test_multiplicity_check1_df(self):
		self.assertFalse(multiplicity_correct(self.f, self.df, 2, 1))

	def test_multiplicity_check2_df(self):
		self.assertTrue(multiplicity_correct(self.f, self.df, 2, 2))

	def test_multiplicity_check3_df(self):
		self.assertFalse(multiplicity_correct(self.f, self.df, 2, 3))

	def test_multiplicity_check4_df(self):
		self.assertFalse(multiplicity_correct(self.f, self.df, 1, 4))

	def test_multiplicity_check5_df(self):
		self.assertTrue(multiplicity_correct(self.f, self.df, 1, 5))

	def test_multiplicity_check6_df(self):
		self.assertFalse(multiplicity_correct(self.f, self.df, 1, 6))

if __name__ == '__main__':
	unittest.main()
