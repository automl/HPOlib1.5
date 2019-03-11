import ast
import configparser
import logging
import os
from io import StringIO


class HPOlibConfig:

    def __init__(self):
        """ Holds configuration for HPOlib. When initialized reads (or creates)
         the config file and data directory accordingly.

         Parameters:
         -----------

        config_file: str
            Path to config file
        """

        self.logger = logging.getLogger("HPOlibConfig")
        self.config_file = "~/.hpolibrc"
        self.global_data_dir = "/var/lib/hpolib/"

        self.config = None
        self.data_dir = None
        self.socket_dir = None
        self.image_source = None
        self.use_global_data = None

        self.defaults = {'verbosity': 0,
                         'data_dir': os.path.expanduser("~/.hpolib/"),
                         'socket_dir': os.path.expanduser("~/.cache/hpolib/"),
                         'image_dir': "/tmp/hpolib-" + str(os.getuid()) + "/",
                         'image_source': "shub://automl/HPOlib2",
                         'use_global_data': True,
                         'pyro_connect_max_wait': 60,
                         'singularity_use_instances': True}

        self._setup(self.config_file)

    def _setup(self, config_file):
        """ Sets up config. Reads the config file and parses it.

        Parameters:
        -----------

        config_file: str
            Path to config file
        """

        # Change current config file to new config file
        config_file = self.__make_abs_path(config_file)

        if config_file != self.config_file:
            self.logger.debug("Change config file from %s to %s",
                              self.config_file, config_file)
            self.config_file = config_file

        # Create an empty config file if there was none so far
        if not os.path.exists(self.config_file):
            self.__create_config_file()

        # Parse config and store input in self.config
        self.__parse_config()

        # Check whether data_dir exists, if not create
        self.__check_data_dir(self.data_dir)
        self.__check_data_dir(self.socket_dir)
        self.__check_data_dir(self.image_dir)

    @staticmethod
    def __make_abs_path(path):
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def __create_config_file(self):
        try:
            self.logger.debug("Create a new config file here: %s",
                              self.config_file)
            fh = open(self.config_file, "w")
            for k in self.defaults:
                fh.write("%s=%s\n" % (k, self.defaults[k]))
            fh.close()
        except (IOError, OSError):
            raise

    def __parse_config(self):
        """Parse the config file"""
        config = configparser.RawConfigParser()

        # Cheat the ConfigParser module by adding a fake section header
        config_file_ = StringIO()
        config_file_.write("[FAKE_SECTION]\n")
        with open(self.config_file) as fh:
            for line in fh:
                config_file_.write(line)
        config_file_.seek(0)
        config.read_file(config_file_)

        self.config = config

        # Store configuration
        self.data_dir = self.__get_config_option('data_dir')
        self.socket_dir = self.__get_config_option('socket_dir')
        self.image_dir = self.__get_config_option('image_dir')
        self.image_source = self.__get_config_option('image_source')
        self.use_global_data = self.__get_config_option('use_global_data')
        if type(self.use_global_data) is str:
            self.use_global_data = ast.literal_eval(self.use_global_data)
        self.pyro_connect_max_wait = int(self.__get_config_option('pyro_connect_max_wait'))
        self.singularity_use_instances = self.__get_config_option('singularity_use_instances')
        if type(self.singularity_use_instances) is str:
            self.singularity_use_instances = ast.literal_eval(self.singularity_use_instances)

        # Use global data dir if exist
        if os.path.isdir(self.global_data_dir) and self.use_global_data:
            self.data_dir = self.global_data_dir

    def __get_config_option(self, o):
        """ Try to get config option from configuration file. If the option is not configured, use default """
        try:
            return self.config.get('FAKE_SECTION', o)
        except configparser.NoOptionError:
            return self.defaults[o]

    def __check_data_dir(self, path):
        """ Check whether data dir exists and if not create it"""
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        except (IOError, OSError):
            self.logger.debug("Could not create data directory here: %s",
                              path)
            raise


_config = HPOlibConfig()

__all__ = ['_config']
