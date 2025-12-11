import unittest
import os
import subprocess
import shutil

class TestVisualize(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(base_dir)
        
        self.test_dir = os.path.join(base_dir, 'test_viz_output')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        
        self.uvfits_path = os.path.join(base_dir, 'data', 'mring+hsCW_LO_onsky.uvfits')
        self.recon_path = os.path.join(base_dir, 'data', 'mring+hsCW_LO_onsky_truth.hdf5')
        self.truth_path = os.path.join(base_dir, 'data', 'mring+hsCW_LO_onsky_truth.hdf5')
        
        self.outpath_prefix = os.path.join(self.test_dir, 'viz_test')

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_visualize_run(self):
        # Run the script
        cmd = [
            'python', 'src/visualize.py',
            '-d', self.uvfits_path,
            '--input', self.recon_path,
            '--truthmv', self.truth_path,
            '-o', self.outpath_prefix,
            '--ncores', '2' # Test parallel with 2 cores
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        
        # Check outputs
        self.assertTrue(os.path.exists(f'{self.outpath_prefix}_total.gif'), "Total Intensity GIF not created")
        self.assertTrue(os.path.exists(f'{self.outpath_prefix}_lp.gif'), "LP GIF not created")
        self.assertTrue(os.path.exists(f'{self.outpath_prefix}_var.png'), "Variance Plot not created")

if __name__ == '__main__':
    unittest.main()
