import unittest
import os
from pomdpy.parsers.pomdp import parse
from pomdpy.pomdp import POMDP

class TestRegression(unittest.TestCase):

    def test_parse_tiger_pomdp(self):
        # Construct the path to the pomdp file relative to this test file
        current_dir = os.path.dirname(__file__)
        pomdp_file_path = os.path.join(current_dir, '..', 'examples', 'tiger.pomdp')

        with open(pomdp_file_path, 'r') as f:
            pomdp_content = f.read()

        # Parse the pomdp file
        pomdp_model = parse(pomdp_file_path) # parse function expects a file path

        # Assert that the returned POMDP object is not None
        self.assertIsNotNone(pomdp_model)

        # Assert that the POMDP object's states are ['tiger-left', 'tiger-right']
        self.assertEqual(pomdp_model.states, ['tiger-left', 'tiger-right'])

        # Assert that the POMDP object's actions are ['listen', 'open-left', 'open-right']
        self.assertEqual(pomdp_model.actions, ['listen', 'open-left', 'open-right'])

        # Assert that the POMDP object's observations are ['tiger-left', 'tiger-right']
        self.assertEqual(pomdp_model.observations, ['tiger-left', 'tiger-right'])

    def test_parse_kaspers_pomdp(self):
        current_dir = os.path.dirname(__file__)
        pomdp_file_path = os.path.join(current_dir, '..', 'examples', 'kaspers.pomdp')

        with open(pomdp_file_path, 'r') as f:
            pomdp_content = f.read()

        pomdp_model = parse(pomdp_file_path)

        self.assertIsNotNone(pomdp_model)
        self.assertEqual(pomdp_model.states, ['s0', 's1', 's2'])
        self.assertEqual(pomdp_model.actions, ['a0', 'a1'])
        self.assertEqual(pomdp_model.observations, ['o0', 'o1'])

    def test_parse_pierres_pomdp(self):
        current_dir = os.path.dirname(__file__)
        pomdp_file_path = os.path.join(current_dir, '..', 'examples', 'pierres.pomdp')

        with open(pomdp_file_path, 'r') as f:
            pomdp_content = f.read()

        pomdp_model = parse(pomdp_file_path)

        self.assertIsNotNone(pomdp_model)
        self.assertEqual(pomdp_model.states, ['s0', 's1', 's2', 's3'])
        self.assertEqual(pomdp_model.actions, ['a0', 'a1'])
        self.assertEqual(pomdp_model.observations, ['o0', 'o1'])

    def test_parse_revealing_tiger_pomdp(self):
        current_dir = os.path.dirname(__file__)
        pomdp_file_path = os.path.join(current_dir, '..', 'examples', 'revealing-tiger.pomdp')

        with open(pomdp_file_path, 'r') as f:
            pomdp_content = f.read()

        pomdp_model = parse(pomdp_file_path)

        self.assertIsNotNone(pomdp_model)
        self.assertEqual(pomdp_model.states, ['tiger-left', 'tiger-right'])
        self.assertEqual(pomdp_model.actions, ['listen', 'open-left', 'open-right'])
        self.assertEqual(pomdp_model.observations, ['tiger-left', 'tiger-right', 'null'])

    def test_aswin_tiger_cobuchi_tiger_left(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'  # Assumes aswin.py is in the repo root
        pomdp_file_path = os.path.join('examples', 'tiger.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'tiger.pomdp.cobuchi_tiger-left.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..') # Assuming tests directory is one level down from repo root

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 'tiger-left']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_aswin_kaspers_cobuchi_s0(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'
        pomdp_file_path = os.path.join('examples', 'kaspers.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'kaspers.pomdp.cobuchi_s0.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..')

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 's0']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_aswin_pierres_cobuchi_s5(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'
        pomdp_file_path = os.path.join('examples', 'pierres.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'pierres.pomdp.cobuchi_s5.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..')

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 's5']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_aswin_pierres_cobuchi_s5_buchi_s6(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'
        pomdp_file_path = os.path.join('examples', 'pierres.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'pierres.pomdp.cobuchi_s5_buchi_s6.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..')

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 's5', '--buchi', 's6']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_aswin_revealing_tiger_cobuchi_dead_buchi_done(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'
        pomdp_file_path = os.path.join('examples', 'revealing-tiger.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'revealing-tiger.pomdp.cobuchi_dead_buchi_done.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..')

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 'dead', '--buchi', 'done']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_aswin_revealing_tiger_cobuchi_tiger_left_tiger_right_buchi_done(self):
        import subprocess
        
        # Define paths relative to the repository root
        aswin_script_path = 'aswin.py'
        pomdp_file_path = os.path.join('examples', 'revealing-tiger.pomdp')
        expected_output_file_path = os.path.join('tests', 'expected_outputs', 'aswin', 'revealing-tiger.pomdp.cobuchi_tiger-left_tiger-right_buchi_done.txt')

        # Construct absolute paths from the perspective of this test file's location
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..')

        abs_aswin_script_path = os.path.join(repo_root, aswin_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path)

        command_parts = ['python', abs_aswin_script_path, abs_pomdp_file_path, '--cobuchi', 'tiger-left', 'tiger-right', '--buchi', 'done']

        # Execute the command
        process = subprocess.run(command_parts, capture_output=True, text=True, check=True, cwd=repo_root)
        actual_output = process.stdout.strip()

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output = f.read().strip()
        
        self.assertEqual(actual_output, expected_output)

    def test_sim_tiger_listen_open_left(self):
        import subprocess
        import os

        # Define paths relative to the repository root
        sim_script_path = 'sim.py'
        pomdp_file_path_rel = os.path.join('examples', 'tiger.pomdp')
        expected_output_file_path_rel = os.path.join('tests', 'expected_outputs', 'sim', 'tiger.pomdp.listen_open-left.txt')

        # Construct absolute paths (or paths relative to cwd if running from repo root)
        current_dir = os.path.dirname(__file__)
        repo_root = os.path.join(current_dir, '..') # Assuming tests directory is one level down

        abs_sim_script_path = os.path.join(repo_root, sim_script_path)
        abs_pomdp_file_path = os.path.join(repo_root, pomdp_file_path_rel)
        abs_expected_output_file_path = os.path.join(repo_root, expected_output_file_path_rel)

        command_parts = ['python', abs_sim_script_path, abs_pomdp_file_path, '-s']
        actions_string = "listen\nopen-left\n"

        # Execute the command
        # We expect an EOFError, so the non-zero exit code is fine.
        process = subprocess.run(command_parts, input=actions_string, capture_output=True, text=True, cwd=repo_root)
        
        raw_stdout = process.stdout
        
        # Filter the captured stdout
        filtered_lines = []
        for line in raw_stdout.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("Enter an action:") or \
               line_stripped.startswith("New observation ="):
                filtered_lines.append(line_stripped)
        
        actual_filtered_output = "\n".join(filtered_lines)

        # Read the content of the golden file
        with open(abs_expected_output_file_path, 'r') as f:
            expected_output_content = f.read().strip()
            # Also strip each line in the expected output for robust comparison
            expected_output_lines = [line.strip() for line in expected_output_content.splitlines()]
            expected_output_final = "\n".join(expected_output_lines)

        self.assertEqual(actual_filtered_output.strip(), expected_output_final)


if __name__ == '__main__':
    unittest.main()
