# ---------------------------------------------------------------------------- #
"""
    - obslog reads critical header keywords from FITS files and collates them
      into a dictionary. This is used as the basis for an observation log.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os

# astropy import
from astropy.io import fits

# json imports (for loading config file)
from json import load

# ---------------------------------------------------------------------------- #

MYNAME = 'obslog'

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
def create_obslog(fits_files, config_file, ext='Primary'):
# ---------------------------------------------------------------------------- #

    """
    For a list of FITS files, create a dictionary for header keys and values
    from the primary (or specified) header extension of the files.
    The specified config file contains the list of the header keys and the
    format of the keys. The list of header keys must start with the key
    'FILENAME'.
    """

    # Load config file (JSON input file)
    with open(config_file, 'r') as config:
        conf = load(config)

    # Set obslog list
    obslog_list = conf['log_list']
    # Set lists needed to create obslog dictionary:
    # - keywords
    key_list = [key for key, _ in obslog_list]
    # - formats
    frm_list = [frm for _, frm in obslog_list]

    # Initialise obslog dictionary
    obslog_dict = {}
    for key in key_list:
        obslog_dict[key] = []

    # Sort list of fits files
    fits_files.sort()

    # Loop for fits files...
    for fits_file in fits_files:

        # Open fits file
        with fits.open(fits_file) as hdulist:

            # Set file name without full path
            file_name = os.path.basename(fits_file)
            # Add file name
            obslog_dict['FILENAME'].append(file_name)

            # Loop for keywords and formats
            for key, frm in zip(key_list[1:], frm_list[1:]):

                # Get default value based on key format
                default = get_default(frm)
                # Get key value
                value = get_value(hdulist[ext], key, default)
                # Add value to obslog dictionary
                obslog_dict[key].append(value)

    return obslog_dict

# ---------------------------------------------------------------------------- #
def get_default(frmt):
# ---------------------------------------------------------------------------- #

    """
    Set the default value for a given format.
    """

    # - string format...
    if frmt.count('A'): 
        default = 'UNKNOWN'

    # - integer
    elif frmt.count('I') or frmt.count('J') or frmt.count('K'):
        default = -999

    # - other (float)
    else: 
        default = -999.99

    return default

# ---------------------------------------------------------------------------- #
def get_value(hdu, key, default):
# ---------------------------------------------------------------------------- #

    """
    Get the value for the key. Return default value if key does not exist or
    retrieved value is 'empty'.
    """

    try:
        value = hdu.header[key]

        # If value is a string...
        if isinstance(value, str):
            # Remove any spaces
            value = value.strip()

        # If value is 'empty'...
        if str(value).strip() == '':
            # Override value with default
            value = default

    except:
        value = default

    return value

# ---------------------------------------------------------------------------- #