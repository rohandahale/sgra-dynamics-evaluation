import unittest
import os
import shutil
import pandas as pd
import sys
import numpy as np

# Add src to path to import mean_image_extraction
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import mean_image_extraction
import ehtim as eh

class TestMeanImageExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'tests/test_data'
        self.output_csv = 'tests/test_output.csv'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy FITS files
        self.create_dummy_fits(os.path.join(self.test_dir, 'sample1.fits'), flux=1.0)
        self.create_dummy_fits(os.path.join(self.test_dir, 'sample2.fits'), flux=1.2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def create_dummy_fits(self, filename, flux=1.0, r0=20*eh.RADPERUAS, sigma=5*eh.RADPERUAS):
        # Create a ring image
        im = eh.image.make_empty(32, 160*eh.RADPERUAS, 0.0, 230e9)
        im = im.add_ring_m1(flux, 0.0, r0, 0.0, sigma, pol='I')
        # Add some polarization
        im.qvec = im.ivec * 0.1
        im.uvec = im.ivec * 0.1
        im.vvec = im.ivec * 0.01
        im.save_fits(filename)

    def test_extraction_directory(self):
        # Run the main function logic
        cmd = f"python3 src/mean_image_extraction.py --fits {self.test_dir} -o {self.output_csv} --ncores 1"
        exit_code = os.system(cmd)
        self.assertEqual(exit_code, 0, "Script execution failed")

        # Check output
        self.assertTrue(os.path.exists(self.output_csv), "Output CSV not created")
        
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 2, "Should have 2 rows for 2 input files")

    def test_extraction_directory_parallel(self):
        # Run with parallel cores
        cmd = f"python3 src/mean_image_extraction.py --fits {self.test_dir} -o {self.output_csv} --ncores 2"
        exit_code = os.system(cmd)
        self.assertEqual(exit_code, 0, "Parallel script execution failed")

        # Check output
        self.assertTrue(os.path.exists(self.output_csv), "Output CSV not created")
        
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 2, "Should have 2 rows for 2 input files")

    def test_extraction_single_file(self):
        single_file = os.path.join(self.test_dir, 'sample1.fits')
        cmd = f"python3 src/mean_image_extraction.py --fits {single_file} -o {self.output_csv}"
        exit_code = os.system(cmd)
        self.assertEqual(exit_code, 0, "Script execution failed for single file")
        
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 1, "Should have 1 row for single file")
        self.assertAlmostEqual(df['D'].iloc[0], 40.0, delta=5.0, msg="Diameter D should be close to 40 uas")

if __name__ == '__main__':
    unittest.main()
