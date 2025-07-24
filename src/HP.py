import yaml
import sys
import ast

CONFIG_PATH = "HP.yaml"


class _HP:
    """Singleton class for storing and retrieving hyperparameters"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Check if an instance already exists,
        if it is the case, return the existing instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path):
        """Method for loading default hyperparameters from a YAML file"""
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)
        # self.config['TRAIN_ID'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    def __getitem__(self, key, default=None):
        try:
            return self.config[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        self.config[key] = value

    def keys(self):
        """Method for getting the keys of the configuration file"""
        return self.config.keys()

    def get(self, key, default=None):
        """Method for getting a value from the configuration file with a default"""
        try:
            return self.config[key]
        except KeyError:
            return default

    def values(self):
        """Method for getting the values of the configuration file"""
        return self.config.values()

    def update(self, d):
        self.config.update(d)

    def get_config(self):
        return self.config


def get_script_arguments(keys):
    """
    Parses the script input arguments into a dictionary

    Outputs
        - args (dict): Dictionnary where keys are argument names and values
        are argument values
    """

    args = {}

    # Iterate over all command line arguments (starting at the second one,
    # the first one beeing the name of the file)
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--"):
            name = arg[2:]  # Remove the '--'
            if name in keys:
                value_str = sys.argv[
                    i + 2
                ]  # The argument value is the next input after the name
                try:
                    value = ast.literal_eval(
                        value_str
                    )  # Convert the value string into its corresponding Python object
                    #  (int, float, str)
                except ValueError:
                    value = value_str  # If the value is not convertible to a
                    # Python object type, then it is a string

                args[name] = value

    return args
