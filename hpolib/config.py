import configparser
import os
from io import StringIO

_default_data_dir = "~/.hpolib"
_default_config_file = os.path.expanduser("~/.hpolibrc")


def _setup():
    """Setup hpolib package. Called on first import.
    Reads the config file and data directory accordingly.
    The location of the data directory can be specified by:
    hpolib.config.data_dir = DIR
    """
    global _default_data_dir

    # read config file, create cache directory
    try:
        os.mkdir(os.path.expanduser(_default_data_dir))
    except (IOError, OSError):
        # TODO add debug information
        pass
    config = _parse_config()
    new_data_dir = config.get('FAKE_SECTION', 'data_dir')
    set_data_directory(new_data_dir)


def set_data_directory(data_dir):
    """Set module-wide cache directory.
    Sets the hpolib data directory.

    Parameters
    ----------
    data_dir : str

    See also
    --------
    get_data_directory
    """

    global _default_data_dir
    _default_data_dir = data_dir

    if not os.path.exists(_default_data_dir) \
            and not os.path.isdir(_default_data_dir):
        os.mkdir(_default_data_dir)


def _parse_config():
    """Parse the config file, set up defaults.
    """
    defaults = {'verbosity': 0,
                'data_dir': _default_data_dir}

    config_file = _default_config_file
    config = configparser.RawConfigParser(defaults=defaults)

    if not os.path.exists(config_file):
        # Create an empty config file if there was none so far
        fh = open(config_file, "w")
        fh.close()
        print("Could not find a configuration file at %s. Going to "
              "create an empty file there." % config_file)

    # Cheat the ConfigParser module by adding a fake section header
    config_file_ = StringIO()
    config_file_.write("[FAKE_SECTION]\n")
    with open(config_file) as fh:
        for line in fh:
            config_file_.write(line)
    config_file_.seek(0)
    config.read_file(config_file_)

    return config


def get_data_directory():
    """Get the current data directory.

    Returns
    -------
    cachedir : string
        The current data directory.

    See also
    --------
    set_data_directory
    """
    return _default_data_dir


def get_config_file():
    """ Return the current config file

    Returns
    -------
    file : basestring
        The current configuration directory

    """
    return _default_config_file


def set_config_file(config_file):
    """Set module-wide cache file.

    Parameters
    ----------
    config_file : str

    See also
    --------
    get_config_file
    """

    global _default_config_file
    _default_config_file = config_file
    _setup()


__all__ = ["set_data_directory", 'get_data_directory',
           "set_config_directory", 'get_config_directory']


_setup()
