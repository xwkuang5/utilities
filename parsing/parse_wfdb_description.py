import os
import re

def parse_wfdb_description(desc_filename):
    """Parse the wfdb description files by finding the names of the channels used

    Parameters:
        desc_filename   - the input description file

    Returns:
        matches         - names of the channels
    """

    if os.path.exists(desc_filename):
        try:
            with open(desc_filename, "r") as f:
                regex = r".*Description: (.*)$"
                content = f.read()
                matches = re.findall(regex, content, flags=re.MULTILINE)
                return matches
            f.close()
        except:
            # TODO: handle exception
            raise
