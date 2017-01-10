import os
import tempfile
import unittest
import unittest.mock

import hpolib
import hpolib.config


class TestConfig(unittest.TestCase):

    @unittest.mock.patch.object(hpolib.config.HPOlibConfig,
                                '_HPOlibConfig__parse_config')
    @unittest.mock.patch.object(hpolib.config.HPOlibConfig,
                                '_HPOlibConfig__check_data_dir')
    @unittest.mock.patch.object(hpolib.config.HPOlibConfig,
                                '_HPOlibConfig__create_config_file')
    @unittest.mock.patch.object(hpolib.config.HPOlibConfig,
                                '_HPOlibConfig__make_abs_path')
    @unittest.mock.patch('os.path.expanduser')
    def test_automatic_startup(self, expand_mock, abs_mock, config_mock,
                               data_mock, parser_mock):
        fixture = '~/ureazpnblvxfrcmbpdatmbpdeai'

        # Set up mocks
        expand_mock.side_effect = ['default_data_dir']

        # Do not create or check for directories
        parser_mock.return_value = True
        data_mock.return_value = True
        config_mock.return_value = True
        abs_mock.return_value = fixture

        # Initialize config
        c = hpolib.config.HPOlibConfig()

        # Check values
        self.assertEqual(c.config_file, fixture)
        self.assertEqual(c.data_dir, None)  # config has never been parsed
        self.assertEqual(c.defaults["data_dir"], 'default_data_dir')

        # Check mock call counts
        self.assertEqual(parser_mock.call_count, 1)

        # Check call arguments
        self.assertEqual(abs_mock.call_args[0][0], "~/.hpolibrc")

    @unittest.mock.patch('os.makedirs')
    @unittest.mock.patch('os.path.expanduser')
    def test_parse_config_exists(self, expand_mock, mkdir_mock):
        # Set up tmp config file with entries
        config_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        config_file.write('key = value\n')
        config_file.write('data_dir = %s' % 'new_dir')
        config_file.close()

        # Set up mock effects
        expand_mock.side_effect = [config_file.name, 'default_dir']
        mkdir_mock.return_value = True

        # Parse new config file
        hpolib._config._setup(config_file.name)

        # Asserts
        self.assertEqual(hpolib._config.config.sections(), ['FAKE_SECTION'])
        self.assertEqual(hpolib._config.config.get('FAKE_SECTION', 'key'),
                         'value')
        self.assertEqual(hpolib._config.config.get('FAKE_SECTION', 'data_dir'),
                         'new_dir')
        self.assertEqual(mkdir_mock.call_count, 1)

        # Remove file
        try:
            os.remove(config_file.name)
        except Exception:
            pass

    @unittest.mock.patch('os.path.expanduser')
    def test_parse_config_not_exist(self, expand_mock):
        # Set up tmp config file and delete it to get filename
        config_file = tempfile.NamedTemporaryFile(mode='w')
        config_file.close()

        expand_mock.side_effect = ["default_dir", config_file.name]

        # Re__init__ config
        hpolib._config.__init__()

        # Asserts
        self.assertEqual(hpolib._config.config.sections(), ['FAKE_SECTION'])
        self.assertEqual(hpolib._config.config.get('FAKE_SECTION', 'data_dir'),
                         'default_dir')
        self.assertEqual(hpolib._config.data_dir, 'default_dir')

        # Remove file
        try:
            os.remove(config_file.name)
        except Exception:
            pass
