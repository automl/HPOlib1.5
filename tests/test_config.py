import configparser
import os
import tempfile
import unittest
import unittest.mock

import hpolib.config
hpolib.config.__all__.append('_parse_config')
hpolib.config.__all__.append('_setup')


class TestConfig(unittest.TestCase):

    @unittest.mock.patch('hpolib.config.set_data_directory')
    @unittest.mock.patch('hpolib.config._parse_config')
    @unittest.mock.patch('os.mkdir')
    def test_automatic_startup(self, mkdir_mock, parser_mock, setter_mock):
        config = unittest.mock.Mock(spec=configparser.RawConfigParser)
        fixture = '~/ureazpnblvxfrcmbpdatmbpdeai'
        config.get.return_value = fixture
        parser_mock.return_value = config

        hpolib.config._setup()

        self.assertEqual(mkdir_mock.call_count, 1)
        # This is the end of the extended path to the user directory
        self.assertTrue(mkdir_mock.call_args[0][0].startswith('/home'))
        self.assertTrue(mkdir_mock.call_args[0][0].endswith('/.hpolib'))

        self.assertEqual(parser_mock.call_count, 1)
        self.assertEqual(config.get.call_count, 1)
        self.assertEqual(config.get.call_args[0], ('FAKE_SECTION', 'data_dir'))

        self.assertEqual(setter_mock.call_count, 1)
        self.assertEqual(setter_mock.call_args[0][0], fixture)

    @unittest.mock.patch('os.path.expanduser')
    def test_parse_config(self, expand_mock):
        # Config file exists
        config_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        config_file.write('key = value\n')
        config_file.write('data_dir = new_dir')
        config_file.close()
        name = config_file.name
        expand_mock.side_effect = ['default_dir', name]

        config = hpolib.config._parse_config()
        self.assertEqual(config.sections(), ['FAKE_SECTION'])
        self.assertEqual(config.get('FAKE_SECTION', 'key'), 'value')
        self.assertEqual(config.get('FAKE_SECTION', 'data_dir'), 'new_dir')
        try:
            os.remove(name)
        except Exception:
            pass

        # Config file does not exist
        config_file = tempfile.NamedTemporaryFile(mode='w')
        name = config_file.name
        config_file.close()

        expand_mock.side_effect = ['default_dir', name]

        config = hpolib.config._parse_config()
        self.assertEqual(config.sections(), ['FAKE_SECTION'])
        self.assertEqual(config.get('FAKE_SECTION', 'data_dir'), 'default_dir')
        try:
            os.remove(name)
        except Exception:
            pass
